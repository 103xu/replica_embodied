"""ACT training/evaluation using the tutorial's pinned LeRobot implementation."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.configs.policies import PreTrainedConfig

from .dataset import load_dataset
from .runtime import PROJECT


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')


def chunk_indices(episode_ends, chunk_size):
    """Build future action indices with padding, without crossing episode boundaries."""
    episode_ends = torch.as_tensor(episode_ends, dtype=torch.long)
    raw = torch.arange(len(episode_ends))[:, None] + torch.arange(chunk_size)[None, :]
    padding = raw >= episode_ends[:, None]
    return torch.minimum(raw, episode_ends[:, None] - 1), padding


def dataset_fingerprint(root):
    digest = hashlib.sha256()
    root = Path(root).resolve()
    for file in sorted([*(root / 'meta').glob('*'), *(root / 'data').rglob('*.parquet')]):
        if file.is_file():
            digest.update(file.relative_to(root).as_posix().encode())
            with file.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
    return digest.hexdigest()


def make_config(dataset, device='cuda', chunk_size=10, pretrained=True):
    features = dataset_to_policy_features(dataset.meta.features)
    # Exactly the section 3 tutorial inputs: external camera + end-effector pose.
    inputs = {k: features[k] for k in ('observation.image', 'observation.state')}
    outputs = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    return ACTConfig(input_features=inputs, output_features=outputs, device=device,
                     chunk_size=chunk_size, n_action_steps=chunk_size,
                     pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if pretrained else None)


class CachedFrames:
    """Cache small local datasets on GPU as uint8; sampling retains exact image values."""
    def __init__(self, dataset, chunk_size, device):
        self.device = device
        self.size = dataset.num_frames
        images, states, actions = [], [], []
        for i in range(self.size):
            frame = dataset[i]
            images.append((frame['observation.image'] * 255).round().to(torch.uint8))
            states.append(frame['observation.state'])
            actions.append(frame['action'])
        self.images = torch.stack(images).to(device)
        self.states = torch.stack(states).float().to(device)
        self.actions = torch.stack(actions).float().to(device)
        ends = torch.empty(self.size, dtype=torch.long)
        for start, end in zip(dataset.episode_data_index['from'], dataset.episode_data_index['to']):
            ends[int(start):int(end)] = int(end)
        indices, padding = chunk_indices(ends, chunk_size)
        self.indices, self.padding = indices.to(device), padding.to(device)

    def batch(self, indices, noise=0.0):
        indices = indices.to(self.device)
        image = self.images[indices].float().div_(255)
        if noise:
            image = (image + torch.randn_like(image) * noise).clamp_(0, 1)
        return {'observation.image': image, 'observation.state': self.states[indices],
                'action': self.actions[self.indices[indices]], 'action_is_pad': self.padding[indices]}


@torch.inference_mode()
def predict_chunks(policy, inputs):
    # Ground-truth future actions must never enter the inference-time VAE encoder.
    inputs = {k: inputs[k] for k in policy.config.input_features}
    normalized = policy.normalize_inputs(inputs)
    normalized['observation.images'] = [normalized[k] for k in policy.config.image_features]
    chunks = policy.model(normalized)[0]
    return policy.unnormalize_outputs({'action': chunks})['action']


@torch.inference_mode()
def offline_metrics(policy, cache, batch_size=32):
    policy.eval()
    predictions = []
    for start in range(0, cache.size, batch_size):
        indices = torch.arange(start, min(start + batch_size, cache.size), device=cache.device)
        predictions.append(predict_chunks(policy, cache.batch(indices))[:, 0].float().cpu())
    predicted = torch.cat(predictions)
    target = cache.actions.cpu()
    error = (predicted - target).abs()
    return {'teacher_forced': True, 'future_action_input': False, 'frames': cache.size,
            'mae': float(error.mean()), 'joint_mae_rad': float(error[:, :6].mean()),
            'gripper_mae': float(error[:, 6].mean()), 'per_dimension_mae': error.mean(0).tolist()}, predicted


def load_policy(checkpoint, device='cuda', ensemble=None, action_steps=None):
    checkpoint = Path(checkpoint).resolve()
    for name in ('config.json', 'model.safetensors'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f'Missing trained checkpoint file: {checkpoint / name}')
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, ACTConfig):
        raise ValueError(f'Expected an ACT checkpoint, got {type(config).__name__}')
    config.device = device
    # All backbone weights come from this trained checkpoint; no online fallback.
    config.pretrained_backbone_weights = None
    if ensemble is not None:
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = ensemble
    elif action_steps is not None:
        config.n_action_steps = action_steps
        config.temporal_ensemble_coeff = None
    config.__post_init__()
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
    policy.eval()
    policy.reset()
    return policy


def initialize_from_checkpoint(configuration, dataset_stats, checkpoint, device):
    """Transfer all trained weights; interpolate only action-query positions if chunk length changes."""
    source = load_policy(checkpoint, device='cpu')
    if source.config.input_features != configuration.input_features:
        raise ValueError('Initial checkpoint and training feature shapes must match.')
    if source.config.chunk_size == configuration.chunk_size:
        source.config.device = device
        source.config.n_action_steps = configuration.chunk_size
        source.config.temporal_ensemble_coeff = None
        return source.to(device), {'source_chunk_size': source.config.chunk_size, 'resized_keys': []}
    configuration.pretrained_backbone_weights = None
    policy = ACTPolicy(configuration, dataset_stats=dataset_stats).to(device)
    state, target = source.state_dict(), policy.state_dict()
    if state.keys() != target.keys():
        raise ValueError('Architecture keys differ; only action chunk length transfer is supported.')
    resized = []
    for key in state:
        if state[key].shape == target[key].shape:
            continue
        if key == 'model.decoder_pos_embed.weight':
            state[key] = torch.nn.functional.interpolate(state[key].T.unsqueeze(0),
                            size=configuration.chunk_size, mode='linear', align_corners=True).squeeze(0).T
        elif key == 'model.vae_encoder_pos_enc':
            state[key] = target[key]
        else:
            raise ValueError(f'Unexpected weight shape mismatch: {key}')
        resized.append(key)
    policy.load_state_dict(state, strict=True)
    return policy, {'source_chunk_size': source.config.chunk_size,
                    'target_chunk_size': configuration.chunk_size, 'resized_keys': resized}


def train(args):
    seed_everything(args.seed)
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Choose --device cpu explicitly or check my_env.')
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError('Steps and batch size must be positive.')
    root = args.dataset or PROJECT / 'data' / 'scripted_demo_004'
    output = args.output or PROJECT / 'checkpoints' / 'act_omy_5000'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite training run: {output}. Choose a new --output.')
    dataset = load_dataset(root)
    configuration = make_config(dataset, device, args.chunk_size, pretrained=not args.no_pretrained_backbone)
    cache = CachedFrames(dataset, args.chunk_size, device)
    if args.init_checkpoint:
        policy, transfer = initialize_from_checkpoint(configuration, dataset.meta.stats, args.init_checkpoint, device)
    else:
        policy = ACTPolicy(configuration, dataset_stats=dataset.meta.stats).to(device)
        transfer = None
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    output.mkdir(parents=True)
    report = {'dataset': str(Path(root).resolve()), 'dataset_sha256': dataset_fingerprint(root),
              'episodes': dataset.num_episodes, 'frames': dataset.num_frames,
              'training_steps': args.steps, 'batch_size': args.batch_size, 'chunk_size': args.chunk_size,
              'learning_rate': args.learning_rate, 'noise_std': args.noise_std,
              'seed': args.seed, 'device': device, 'torch': torch.__version__,
              'parameters': sum(p.numel() for p in policy.parameters()),
              'single_episode_overfit': dataset.num_episodes == 1,
              'held_out_episodes': [],
              'normalization': 'initial training checkpoint statistics' if args.init_checkpoint else 'training dataset statistics',
              'initial_checkpoint': str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
              'weight_transfer': transfer,
              'amp_bfloat16': args.amp and device == 'cuda'}
    (output / 'training_run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    before, _ = offline_metrics(policy, cache)
    history = []
    start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    context = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if report['amp_bfloat16'] else nullcontext()
    best_mae = float('inf')
    with (output / 'metrics.jsonl').open('w', encoding='utf-8', buffering=1) as log:
        for step in range(1, args.steps + 1):
            policy.train()
            indices = torch.randint(cache.size, (args.batch_size,), device=device)
            batch = cache.batch(indices, args.noise_std)
            optimizer.zero_grad(set_to_none=True)
            with context():
                loss, parts = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite ACT loss at step {step}: {float(loss)}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            record = {'step': step, 'loss': float(loss.detach()), **parts}
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                record['elapsed_seconds'] = time.perf_counter() - start
                record['steps_per_second'] = step / record['elapsed_seconds']
                print(json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, predicted = offline_metrics(policy, cache)
                record['offline'] = metrics
                print('OFFLINE ' + json.dumps({'step': step, **metrics}), flush=True)
                if metrics['mae'] < best_mae:
                    best_mae = metrics['mae']
                    policy.save_pretrained(output / 'best')
                    (output / 'best' / 'evaluation.json').write_text(json.dumps({'step': step, **metrics}, indent=2), encoding='utf-8')
            history.append(record)
            log.write(json.dumps(record) + '\n')
        policy.save_pretrained(output / 'final')
    after, predictions = offline_metrics(policy, cache)
    report.update({'elapsed_seconds': time.perf_counter() - start, 'before': before, 'after': after,
                   'best_mae': best_mae, 'complete': True,
                   'peak_cuda_memory_mib': torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else 0})
    (output / 'training_run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'offline_actions.npz', predicted=predictions.numpy(), target=cache.actions.cpu().numpy())
    plot_training(output, history, predictions.numpy(), cache.actions.cpu().numpy())
    print('TRAINING_COMPLETE ' + json.dumps(report), flush=True)


def plot_training(output, history, prediction, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot([x['step'] for x in history], [x['loss'] for x in history], alpha=0.6)
    axis.set(xlabel='Training step', ylabel='ACT L1 + KL loss', yscale='log')
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / 'training_loss.png', dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
    for i, axis in enumerate(axes):
        axis.plot(target[:, i], label='Demonstration', linewidth=1)
        axis.plot(prediction[:, i], label='ACT, observation conditioned', linewidth=1, alpha=0.8)
        axis.set_ylabel('gripper' if i == 6 else f'joint {i+1} (rad)')
        axis.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel('Recorded frame (20 Hz); training episode, no held-out claim')
    figure.tight_layout()
    figure.savefig(output / 'offline_actions.png', dpi=150)
    plt.close(figure)
