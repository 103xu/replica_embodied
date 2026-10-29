"""Closed-loop ACT inference: live observations -> trained model -> MuJoCo controls."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .act import load_policy, seed_everything
from .runtime import OMYEnvironment, PROJECT, pace


def deploy(args):
    seed_everything(args.seed)
    if args.max_steps < 1 or args.success_stable_frames < 1 or args.trials < 1:
        raise ValueError('Steps, stable-success frames and trial count must be positive')
    checkpoint = args.checkpoint or PROJECT / 'checkpoints' / 'act_omy_chunk100' / 'final'
    output = args.output or PROJECT / 'outputs' / 'act_deployment'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite rollout: {output}. Choose a new --output.')
    policy = load_policy(checkpoint, args.device,
                         ensemble=args.ensemble if args.mode == 'ensemble' else None,
                         action_steps=args.action_steps if args.mode == 'chunk' else None)
    output.mkdir(parents=True)
    reports, previews, actions, positions = [], [], [], []
    env = OMYEnvironment(seed=args.seed, headless=args.headless, action_type='joint_angle')
    try:
        for trial in range(args.trials):
            if trial:
                env.reset(seed=args.seed + trial)
            policy.reset()
            initial_mug_height = float(env.get_obj_pose()[0][2])
            max_mug_height = initial_mug_height
            predicted_seconds = []
            predicate_success = False
            grasp_success = False
            stable_frames = 0
            first_predicate_step = None
            last_step = -1
            with (output / f'trial_{trial:03d}.jsonl').open('w', encoding='utf-8', buffering=1) as log:
                for step in range(args.max_steps):
                    if not env.alive():
                        break
                    start = time.perf_counter()
                    state = env.get_ee_pose().copy()
                    agent, wrist = env.images()
                    inputs = {'observation.state': torch.from_numpy(state).unsqueeze(0).to(args.device),
                              'observation.image': torch.from_numpy(agent.copy()).permute(2, 0, 1).unsqueeze(0).to(args.device).float() / 255}
                    before = time.perf_counter()
                    with torch.inference_mode():
                        # No recorded actions, task phase, timestamp or scripted controller is supplied.
                        command = policy.select_action(inputs)[0].float().cpu().numpy()
                    predicted_seconds.append(time.perf_counter() - before)
                    if command.shape != (7,) or not np.isfinite(command).all():
                        raise RuntimeError(f'Invalid ACT action at frame {step}: {command}')
                    command[-1] = np.clip(command[-1], 0, 1)
                    env.step(command)
                    env.advance()
                    env.show(agent, wrist)
                    mug, plate = env.get_obj_pose()
                    max_mug_height = max(max_mug_height, float(mug[2]))
                    predicate_success = env.check_success()
                    if predicate_success and first_predicate_step is None:
                        first_predicate_step = step
                    released_and_raised = env.gripper_cmd_scalar <= 0.10 and env.get_ee_pose()[2] >= 0.95
                    candidate = predicate_success and released_and_raised and max_mug_height - initial_mug_height >= 0.10
                    stable_frames = stable_frames + 1 if candidate else 0
                    grasp_success = stable_frames >= args.success_stable_frames
                    row = {'step': step, 'action': command.tolist(), 'state': state.tolist(),
                           'mug': mug.tolist(), 'plate': plate.tolist(),
                           'task_predicate_success': predicate_success, 'grasp_and_place_success': grasp_success,
                           'stable_success_frames': stable_frames, 'applied_gripper_command': env.gripper_cmd_scalar}
                    log.write(json.dumps(row) + '\n')
                    if trial == 0:
                        actions.append(command.copy())
                        positions.append(state)
                        if step % 5 == 0:
                            previews.append(Image.fromarray(agent))
                    last_step = step
                    if step % 100 == 0 or grasp_success:
                        print('ROLLOUT ' + json.dumps(row), flush=True)
                    if grasp_success:
                        break
                    pace(start, args.fast)
            agent, wrist = env.images()
            Image.fromarray(agent).save(output / f'trial_{trial:03d}_agent.png')
            Image.fromarray(wrist).save(output / f'trial_{trial:03d}_wrist.png')
            reports.append({'trial': trial, 'seed': args.seed + trial, 'steps': last_step + 1,
                            'task_predicate_success': predicate_success,
                            'grasp_and_place_success': grasp_success,
                            'first_upstream_predicate_step': first_predicate_step,
                            'stable_success_frames': stable_frames,
                            'applied_gripper_command': env.gripper_cmd_scalar,
                            'mug_lift_height_m': max_mug_height - initial_mug_height,
                            'mean_policy_latency_ms': float(np.mean(predicted_seconds) * 1000) if predicted_seconds else None,
                            'final_success_debug': env.get_success_debug()})
        summary = {'checkpoint': str(Path(checkpoint).resolve()), 'controller': 'ACT neural policy only',
                   'mode': args.mode, 'ensemble': args.ensemble if args.mode == 'ensemble' else None,
                   'action_steps': policy.config.n_action_steps if args.mode == 'chunk' else 1,
                   'fps': 20, 'max_steps': args.max_steps,
                   'confirmation': {'stable_frames': args.success_stable_frames,
                                    'max_gripper_command': 0.10, 'min_ee_height_m': 0.95,
                                    'min_mug_lift_m': 0.10},
                   'task_successes': sum(r['task_predicate_success'] for r in reports),
                   'grasp_successes': sum(r['grasp_and_place_success'] for r in reports),
                   'trials': reports, 'same_seed_as_training': args.seed == 0,
                   'uses_recorded_actions': False, 'uses_scripted_controller': False}
        (output / 'deployment_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        np.savez_compressed(output / 'rollout.npz', actions=np.asarray(actions), states=np.asarray(positions))
        if previews:
            previews[0].save(output / 'rollout.gif', save_all=True, append_images=previews[1:],
                             duration=250, loop=0)
        print('DEPLOYMENT_COMPLETE ' + json.dumps(summary), flush=True)
    finally:
        env.close()
