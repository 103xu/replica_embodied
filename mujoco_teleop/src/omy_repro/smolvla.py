"""Pinned SmolVLA: strict pretrained transfer, small-batch fine-tuning and live rollout."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig

from .act import chunk_indices, dataset_fingerprint, seed_everything
from .language import load_language, target_height, task_status, physical_candidate
from .runtime import LanguageEnvironment, UpstreamLanguageEnvironment, PROJECT, pace


def transfer_core(policy, filename):
    """Transfer every core tensor; normalization is always recomputed for OMY."""
    tensors = load_file(str(filename), device='cpu')
    core = {key.removeprefix('model.'): value for key, value in tensors.items() if key.startswith('model.')}
    if not core:
        raise ValueError('No pretrained model tensors found')
    result = policy.model.load_state_dict(core, strict=False)
    target = policy.model.state_dict()
    # Safetensors omits tied aliases. Permit only an alias of a loaded tensor.
    unexplained = [key for key in result.missing_keys
                   if not any(target[key].data_ptr() == target[loaded].data_ptr() for loaded in core if loaded in target)]
    if unexplained or result.unexpected_keys:
        raise ValueError(f'Pretrained core mismatch: missing={unexplained}, unexpected={result.unexpected_keys}')
    return {'core_tensors_loaded': len(core), 'tied_aliases': result.missing_keys,
            'normalization': 'OMY training episodes only; base robot statistics not reused'}


class Frames:
    """Keep small vectors in memory; decode images only for the current batch."""
    def __init__(self, dataset, chunk_size, holdout_episodes):
        self.dataset = dataset
        # The old datasets NumPy formatter imports a removed torchvision
        # VideoReader even for numeric columns; plain Arrow/Python avoids it.
        vectors = dataset.hf_dataset.select_columns(['observation.state', 'action']).with_format(None)[:]
        self.states = torch.as_tensor(vectors['observation.state'], dtype=torch.float32)
        self.actions = torch.as_tensor(vectors['action'], dtype=torch.float32)
        ends = torch.empty(dataset.num_frames, dtype=torch.long)
        self.train_indices, self.eval_indices = [], []
        for episode, (start, end) in enumerate(zip(dataset.episode_data_index['from'], dataset.episode_data_index['to'])):
            start, end = int(start), int(end)
            ends[start:end] = end
            (self.eval_indices if episode >= dataset.num_episodes - holdout_episodes else self.train_indices).extend(range(start, end))
        if not self.train_indices:
            raise ValueError('No training episodes after holdout split')
        self.indices, self.padding = chunk_indices(ends, chunk_size)
        self.train_indices = torch.tensor(self.train_indices, dtype=torch.long)
        self.eval_indices = torch.tensor(self.eval_indices, dtype=torch.long)

    def stats(self):
        def summarize(values):
            selected = values[self.train_indices]
            return {'mean': selected.mean(0), 'std': selected.std(0, unbiased=False).clamp_min(1e-6),
                    'min': selected.amin(0), 'max': selected.amax(0)}
        return {'observation.state': summarize(self.states), 'action': summarize(self.actions)}

    def batch(self, indices, device):
        indices = torch.as_tensor(indices, dtype=torch.long)
        examples = [self.dataset[int(i)] for i in indices]
        result = {key: torch.stack([row[key] for row in examples]).to(device)
                  for key in ('observation.image', 'observation.wrist_image', 'observation.state')}
        result['task'] = [row['task'] for row in examples]
        result['action'] = self.actions[self.indices[indices]].to(device)
        # The pinned implementation reads this historical spelling.
        result['actions_id_pad'] = self.padding[indices].to(device)
        return result


def amp(device):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()


def load_policy(checkpoint, device):
    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Missing trained model: {checkpoint}')
    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(cfg, SmolVLAConfig):
        raise ValueError('Expected a SmolVLA checkpoint')
    cfg.device = device
    cfg.load_vlm_weights = False  # Complete trained checkpoint supplies all weights.
    local_vlm = PROJECT / '.downloads/models/SmolVLM2-500M-Video-Instruct'
    cfg.vlm_model_name = str(local_vlm.resolve())
    return SmolVLAPolicy.from_pretrained(checkpoint, config=cfg, local_files_only=True, strict=True).to(device).eval()


def diagnose(args):
    """Measure teacher-forced action error and text sensitivity with fixed noise."""
    seed_everything(args.seed)
    if args.samples < 1:
        raise ValueError('samples must be positive')
    dataset = load_language(args.dataset)
    policy = load_policy(args.checkpoint, args.device)
    cache = Frames(dataset, policy.config.chunk_size, min(2, dataset.num_episodes - 1))
    rows = []
    indices = torch.linspace(0, dataset.num_frames - 1, args.samples).long()
    for index in indices:
        batch = cache.batch([index], args.device)
        target = batch.pop('action')[0, 0]
        batch.pop('actions_id_pad')
        policy.reset()
        torch.manual_seed(args.seed)
        with torch.inference_mode(), amp(args.device):
            action = policy.select_action(batch)[0].float()
        error = (action - target).abs()
        rows.append({'frame': int(index), 'task': batch['task'][0],
                     'joint_mae_rad': float(error[:6].mean()), 'gripper_mae': float(error[-1])})
    paired = cache.batch([0], args.device)
    paired.pop('action')
    paired.pop('actions_id_pad')
    actions = []
    tokens = []
    for color in ('red', 'blue', 'red'):
        paired['task'] = [f'Place the {color} mug on the plate.']
        policy.reset()
        torch.manual_seed(args.seed)
        with torch.inference_mode(), amp(args.device):
            tokens.append(policy.prepare_language(paired)[0].cpu().tolist())
            actions.append(policy.select_action(paired)[0].float().cpu())
    result = {'checkpoint': str(args.checkpoint.resolve()), 'samples': rows,
              'teacher_forced': True, 'uses_future_action_input': False,
              'joint_mae_rad': float(np.mean([r['joint_mae_rad'] for r in rows])),
              'gripper_mae': float(np.mean([r['gripper_mae'] for r in rows])),
              'same_scene_text_probe': {'different_tokens': tokens[0] != tokens[1],
                  'red_blue_mean_action_difference': float((actions[0] - actions[1]).abs().mean()),
                  'repeated_red_max_difference': float((actions[0] - actions[2]).abs().max()),
                  'proves_task_success': False}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


def train(args):
    seed_everything(args.seed)
    if args.steps < 1 or args.batch_size < 1 or args.chunk_size < 1 or args.holdout_episodes < 0 or args.log_every < 1:
        raise ValueError('Invalid training settings')
    output = (args.output or PROJECT / 'checkpoints/smolvla_omy').resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    dataset = load_language(args.dataset or PROJECT / 'data/language_demo_020')
    cache = Frames(dataset, args.chunk_size, args.holdout_episodes)
    features = dataset_to_policy_features(dataset.meta.features)
    cfg = SmolVLAConfig(input_features={k: features[k] for k in ('observation.image', 'observation.wrist_image', 'observation.state')},
                       output_features={'action': features['action']}, chunk_size=args.chunk_size,
                       n_action_steps=args.chunk_size, device=args.device,
                       vlm_model_name=str((PROJECT / '.downloads/models/SmolVLM2-500M-Video-Instruct').resolve()),
                       load_vlm_weights=False, prefix_length=0, pad_language_to='max_length')
    policy = SmolVLAPolicy(cfg, dataset_stats=cache.stats())
    pretrained = args.init_checkpoint or PROJECT / '.downloads/models/smolvla_base'
    transfer = transfer_core(policy, Path(pretrained) / 'model.safetensors')
    policy.to(args.device)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=1e-10)
    probe_name = 'model.action_out_proj.weight'
    frozen_name = next(name for name, p in policy.named_parameters() if not p.requires_grad and 'vision_model' in name)
    before = {name: policy.state_dict()[name].detach().cpu().clone() for name in (probe_name, frozen_name)}
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'policy': 'SmolVLA', 'complete': False, 'steps': args.steps, 'batch_size': args.batch_size,
              'chunk_size': args.chunk_size, 'learning_rate': args.learning_rate, 'seed': args.seed,
              'dataset': str(dataset.root), 'dataset_sha256': dataset_fingerprint(dataset.root),
              'episodes': dataset.num_episodes, 'frames': dataset.num_frames,
              'holdout_episodes': args.holdout_episodes, 'training_frames': len(cache.train_indices),
              'evaluation_frames': len(cache.eval_indices), 'pretrained': str(pretrained), 'transfer': transfer,
              'heldout_loss_probe_frames': min(4, len(cache.eval_indices)), 'heldout_loss_is_full_eval': False,
              'trainable_parameters': sum(p.numel() for p in parameters), 'uses_language': True,
              'state_type': 'joint_angle', 'action_type': 'absolute_joint_targets', 'device': args.device}
    (output / 'training_run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        with (output / 'metrics.jsonl').open('w', encoding='utf-8', buffering=1) as log:
            for step in range(1, args.steps + 1):
                policy.train()
                selection = cache.train_indices[torch.randint(len(cache.train_indices), (args.batch_size,))]
                batch = cache.batch(selection, args.device)
                with amp(args.device):
                    loss, _ = policy(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite training loss at step {step}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 10)
                if not torch.isfinite(norm):
                    raise RuntimeError(f'Nonfinite gradient at step {step}')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % args.log_every == 0 or step == args.steps:
                    row = {'step': step, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                           'elapsed_seconds': time.perf_counter() - started,
                           'peak_cuda_mib': torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else None}
                    if len(cache.eval_indices):
                        policy.eval()
                        with torch.inference_mode(), amp(args.device):
                            # Fixed evaluation noise does not alter the training RNG.
                            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device == 'cuda' else []):
                                torch.manual_seed(12345)
                                held = cache.eval_indices[torch.linspace(0, len(cache.eval_indices) - 1, min(4, len(cache.eval_indices))).long()]
                                eval_loss, _ = policy(cache.batch(held, args.device))
                            row['heldout_flow_loss'] = float(eval_loss)
                    log.write(json.dumps(row) + '\n')
                    print('SMOLVLA_TRAIN ' + json.dumps(row), flush=True)
        policy.save_pretrained(output / 'final')
        report['parameter_checks'] = {name: {'changed_elements': int((before[name] != policy.state_dict()[name].detach().cpu()).sum()),
                                            'maximum_change': float((before[name] - policy.state_dict()[name].detach().cpu()).abs().max())}
                                      for name in before}
        if report['parameter_checks'][probe_name]['changed_elements'] == 0:
            raise RuntimeError('Action expert did not update')
        if report['parameter_checks'][frozen_name]['changed_elements'] != 0:
            raise RuntimeError('Frozen vision parameters unexpectedly changed')
        # Strict checkpoint reload is verified separately by deploy-smolvla.
        report.update(complete=True, elapsed_seconds=time.perf_counter() - started,
                      peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else None)
        print('SMOLVLA_TRAINING_COMPLETE ' + json.dumps(report), flush=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (output / 'training_run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def deploy(args):
    seed_everything(args.seed)
    if args.max_steps < 1:
        raise ValueError('max-steps must be positive')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    policy = load_policy(args.checkpoint, args.device)
    output.mkdir(parents=True)
    environment_class = UpstreamLanguageEnvironment if args.upstream_interface else LanguageEnvironment
    env = environment_class(seed=args.seed, headless=args.headless, action_type='joint_angle')
    results = []
    try:
        for color in ('red', 'blue'):
            env.reset(seed=args.seed)
            env.set_instruction(f'Place the {color} mug on the plate.')
            policy.reset()
            initial = maximum = target_height(env)
            frames, stable, preview, latencies = 0, 0, [], []
            physical_stable, max_strict_stable, max_physical_stable, lift_run, max_lift_run = 0, 0, 0, 0, 0
            with (output / f'{color}.jsonl').open('w', encoding='utf-8', buffering=1) as log:
                for step in range(args.max_steps):
                    if not env.alive():
                        break
                    start = time.perf_counter()
                    agent, wrist = env.images()
                    batch = {'observation.image': torch.from_numpy(agent.copy()).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255,
                             'observation.wrist_image': torch.from_numpy(wrist.copy()).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255,
                             'observation.state': torch.from_numpy(env.get_state()).unsqueeze(0).to(args.device),
                             'task': [env.instruction]}
                    query = time.perf_counter()
                    with torch.inference_mode(), amp(args.device):
                        command = policy.select_action(batch)[0].float().cpu().numpy()
                    latencies.append(time.perf_counter() - query)
                    if command.shape != (7,) or not np.isfinite(command).all():
                        raise ValueError(f'Invalid VLA action: {command}')
                    command[-1] = np.clip(command[-1], 0, 1)
                    env.step(command)
                    env.advance()
                    env.show(agent, wrist)
                    height = target_height(env)
                    maximum = max(maximum, height)
                    lift_run = lift_run + 1 if height - initial >= 0.03 else 0
                    max_lift_run = max(max_lift_run, lift_run)
                    status = task_status(env, initial, maximum)
                    stable = stable + 1 if status['strict_success'] else 0
                    physical_stable = physical_stable + 1 if physical_candidate(status, max_lift_run) else 0
                    max_strict_stable = max(max_strict_stable, stable)
                    max_physical_stable = max(max_physical_stable, physical_stable)
                    log.write(json.dumps({'step': step, 'instruction': env.instruction, 'action': command.tolist(),
                                          'current_lift_m': height - initial, 'max_consecutive_lift_ticks': max_lift_run,
                                          'physical_stable_frames': physical_stable, **status}) + '\n')
                    if step % 10 == 0:
                        preview.append(Image.fromarray(agent))
                    frames = step + 1
                    if (physical_stable if args.success_rule == 'physical' else stable) >= 20:
                        break
                    pace(start, args.fast)
            agent, wrist = env.images()
            Image.fromarray(agent).save(output / f'{color}_agent.png')
            Image.fromarray(wrist).save(output / f'{color}_wrist.png')
            if preview:
                preview[0].save(output / f'{color}.gif', save_all=True, append_images=preview[1:], duration=500, loop=0)
            results.append({'instruction': env.instruction, 'seed': args.seed, 'steps': frames,
                            'success': max_strict_stable >= 20, 'stable_frames': max_strict_stable, **status,
                            'physical_success': max_physical_stable >= 20, 'physical_stable_frames': max_physical_stable,
                            'max_consecutive_lift_ticks': max_lift_run,
                            'mean_action_call_ms': float(np.mean(latencies) * 1000)})
            print('SMOLVLA_ROLLOUT ' + json.dumps(results[-1]), flush=True)
        report = {'checkpoint': str(args.checkpoint.resolve()), 'controller': 'SmolVLA neural policy only',
                  'environment_interface': 'upstream_SimpleEnv2' if args.upstream_interface else 'corrected_OMY_mapping',
                  'camera_pipeline': '800x600_resized_to_256x256' if args.upstream_interface else 'native_256x256',
                  'termination_rule': args.success_rule,
                  'physical_rule': 'legacy success + lift >=3cm for >=3 consecutive ticks + final up-axis cosine >=0.85, stable for 20 ticks',
                  'uses_recorded_actions': False, 'uses_scripted_controller': False, 'uses_language': True,
                  'state_type': 'joint_angle', 'fps': 20, 'successes': sum(row['success'] for row in results),
                  'physical_successes': sum(row['physical_success'] for row in results), 'trials': results}
        (output / 'deployment_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        env.close()
