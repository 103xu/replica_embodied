import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time


def parser():
    p = argparse.ArgumentParser(description='Every-Embodied OMY: data, ACT, VLA and physical diagnostics')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Report the active Python, dependencies, CUDA and model status')
    language_collect = sub.add_parser('collect-language', help='Section 5: instruction-labelled two-cup demonstrations')
    language_collect.add_argument('--dataset', type=Path, default=None)
    language_collect.add_argument('--episodes', type=int, default=20)
    language_collect.add_argument('--seed', type=int, default=0)
    language_collect.add_argument('--color', choices=['alternate', 'red', 'blue'], default='alternate')
    language_collect.add_argument('--scripted', action='store_true')
    language_collect.add_argument('--headless', action='store_true')
    language_collect.add_argument('--fast', action='store_true')
    language_replay = sub.add_parser('replay-language', help='Section 6: replay actions with their exact instruction')
    language_replay.add_argument('--dataset', type=Path, default=None)
    language_replay.add_argument('--episode', type=int, default=0)
    language_replay.add_argument('--output', type=Path, default=None)
    language_replay.add_argument('--headless', action='store_true')
    language_replay.add_argument('--fast', action='store_true')
    language_audit = sub.add_parser('audit-language', help='Section 9: audit all recorded physical trajectories, including upright placement')
    language_audit.add_argument('--dataset', type=Path, required=True)
    language_audit.add_argument('--output', type=Path, required=True)
    smoltrain = sub.add_parser('train-smolvla', help='Section 8: fine-tune strictly loaded pretrained SmolVLA')
    smoltrain.add_argument('--dataset', type=Path, default=None)
    smoltrain.add_argument('--output', type=Path, default=None)
    smoltrain.add_argument('--init-checkpoint', type=Path, default=None)
    smoltrain.add_argument('--steps', type=int, default=20000)
    smoltrain.add_argument('--batch-size', type=int, default=1)
    smoltrain.add_argument('--chunk-size', type=int, default=5)
    smoltrain.add_argument('--holdout-episodes', type=int, default=2)
    smoltrain.add_argument('--learning-rate', type=float, default=1e-4)
    smoltrain.add_argument('--seed', type=int, default=42)
    smoltrain.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    smoltrain.add_argument('--log-every', type=int, default=100)
    smoldeploy = sub.add_parser('deploy-smolvla', help='Section 8: evaluate red/blue language instructions in live MuJoCo')
    smoldeploy.add_argument('--checkpoint', type=Path, required=True)
    smoldeploy.add_argument('--output', type=Path, required=True)
    smoldeploy.add_argument('--seed', type=int, default=0)
    smoldeploy.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    smoldeploy.add_argument('--max-steps', type=int, default=1000)
    smoldeploy.add_argument('--headless', action='store_true')
    smoldeploy.add_argument('--fast', action='store_true')
    smoldeploy.add_argument('--upstream-interface', action='store_true', help='Published-model comparison only: original SimpleEnv2 gripper mapping and integrator')
    smoldeploy.add_argument('--success-rule', choices=['physical', 'strict'], default='physical', help='Stop after 20 stable ticks under section-9 physical rule or legacy 10cm confirmation')
    smoldiagnose = sub.add_parser('diagnose-smolvla', help='Section 9: teacher-forced action error and fixed-noise language contrast')
    smoldiagnose.add_argument('--dataset', type=Path, required=True)
    smoldiagnose.add_argument('--checkpoint', type=Path, required=True)
    smoldiagnose.add_argument('--output', type=Path, required=True)
    smoldiagnose.add_argument('--samples', type=int, default=20)
    smoldiagnose.add_argument('--seed', type=int, default=42)
    smoldiagnose.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    collect = sub.add_parser('collect', help='Section 1: keyboard demonstration collection')
    collect.add_argument('--dataset', type=Path, default=None)
    collect.add_argument('--episodes', type=int, default=1)
    collect.add_argument('--seed', type=int, default=0)
    collect.add_argument('--max-seconds', type=float, default=None)
    smoke = sub.add_parser('smoke', help='Automated collection/replay plumbing test; not a grasp demonstration')
    smoke.add_argument('--dataset', type=Path, default=None)
    smoke.add_argument('--frames', type=int, default=40)
    smoke.add_argument('--headless', action='store_true')
    smoke.add_argument('--fast', action='store_true')
    demo = sub.add_parser('auto-demo', help='Run a scripted IK grasp and collect its actual physics trajectory')
    demo.add_argument('--dataset', type=Path, default=None)
    demo.add_argument('--seed', type=int, default=0)
    demo.add_argument('--headless', action='store_true')
    demo.add_argument('--fast', action='store_true')
    demo.add_argument('--output', type=Path, default=None, help='Optional preview directory; keep audit outputs separate')
    replay = sub.add_parser('replay', help='Section 2: replay local action labels in MuJoCo')
    replay.add_argument('--dataset', type=Path, default=None)
    replay.add_argument('--episode', type=int, default=0)
    replay.add_argument('--headless', action='store_true')
    replay.add_argument('--fast', action='store_true')
    replay.add_argument('--output', type=Path, default=None)
    training = sub.add_parser('train-act', help='Section 3: train the pinned LeRobot ACT policy')
    training.add_argument('--dataset', type=Path, default=None)
    training.add_argument('--output', type=Path, default=None)
    training.add_argument('--steps', type=int, default=5000)
    training.add_argument('--batch-size', type=int, default=16)
    training.add_argument('--chunk-size', type=int, default=10)
    training.add_argument('--learning-rate', type=float, default=1e-4)
    training.add_argument('--noise-std', type=float, default=0.02)
    training.add_argument('--seed', type=int, default=0)
    training.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    training.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    training.add_argument('--no-pretrained-backbone', action='store_true')
    training.add_argument('--log-every', type=int, default=50)
    training.add_argument('--eval-every', type=int, default=250)
    training.add_argument('--init-checkpoint', type=Path, default=None, help='Initialize from a trained ACT checkpoint')
    deployment = sub.add_parser('deploy-act', help='Section 4: run a trained ACT model in closed-loop MuJoCo')
    deployment.add_argument('--checkpoint', type=Path, default=None)
    deployment.add_argument('--output', type=Path, default=None)
    deployment.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    deployment.add_argument('--seed', type=int, default=0)
    deployment.add_argument('--trials', type=int, default=1)
    deployment.add_argument('--max-steps', type=int, default=1200)
    deployment.add_argument('--mode', choices=['ensemble', 'chunk'], default='chunk')
    deployment.add_argument('--ensemble', type=float, default=0.9)
    deployment.add_argument('--action-steps', type=int, default=None)
    deployment.add_argument('--headless', action='store_true')
    deployment.add_argument('--fast', action='store_true')
    deployment.add_argument('--success-stable-frames', type=int, default=20)
    return p


def doctor():
    from .runtime import PROJECT, SCENE
    import torch
    result = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'cuda_available': torch.cuda.is_available(), 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
              'scene_exists': SCENE.is_file(), 'project': str(PROJECT)}
    for name in ('mujoco', 'numpy', 'scipy', 'datasets', 'lerobot', 'pyarrow', 'pillow'):
        result[name] = importlib.metadata.version(name)
    print(json.dumps(result, indent=2))


def collect(args, smoke=False):
    import numpy as np
    from PIL import Image
    from .dataset import add_frame, create_dataset, write_run
    from .runtime import OMYEnvironment, PROJECT, pace

    if smoke and args.frames < 2:
        raise ValueError('--frames must be at least 2')
    if not smoke and args.episodes < 1:
        raise ValueError('--episodes must be positive')
    root = args.dataset or PROJECT / 'data' / ('smoke' if smoke else 'demo_data')
    dataset = create_dataset(root)
    env = None
    frames, episodes = 0, 0
    started = False
    results = []
    initial_state = None
    wall_start = time.perf_counter()
    print('Focus the MuJoCo window. WASD/R/F: move; arrows/Q/E: rotate; Space: gripper; Z: discard/reset; Esc: quit.')
    try:
        env = OMYEnvironment(seed=0 if smoke else args.seed, headless=smoke and args.headless)
        wall_start = time.perf_counter()
        while env.alive():
            start = time.perf_counter()
            if not smoke and args.max_seconds and start - wall_start >= args.max_seconds:
                break
            if smoke:
                action = np.zeros(7, dtype=np.float32)
                action[2] = 0.001 if frames < args.frames // 2 else -0.001
                reset = False
            else:
                action, reset = env.teleop_robot()
            if reset:
                dataset.clear_episode_buffer()
                env.reset(seed=args.seed)
                started = False
                initial_state = None
                continue
            started = started or bool(np.any(action != 0))
            if started and initial_state is None:
                initial_state = env.snapshot()
            # Observation is captured BEFORE preparing the current action label.
            state = env.get_ee_pose().copy()
            agent, wrist = env.images()
            env.step(action)
            if started:
                add_frame(dataset, env, agent, wrist, state)
                frames += 1
            env.advance()
            env.show(agent, wrist)
            success = env.check_success()
            if started and ((smoke and frames >= args.frames) or (not smoke and success)):
                dataset.save_episode()
                states = Path(root).resolve() / 'initial_states'
                states.mkdir(exist_ok=True)
                np.savez_compressed(states / f'episode_{episodes:06d}.npz', state=initial_state)
                episodes += 1
                results.append({'episode': episodes - 1, 'success': bool(success), 'smoke': smoke})
                print(f'Saved episode {episodes}; physical task success={success}; smoke={smoke}')
                if smoke or episodes >= args.episodes:
                    if smoke:
                        output = PROJECT / 'outputs'
                        output.mkdir(exist_ok=True)
                        Image.fromarray(agent).save(output / 'agent.png')
                        Image.fromarray(wrist).save(output / 'wrist.png')
                    break
                env.reset(seed=args.seed)
                started = False
                initial_state = None
            pace(start, smoke and args.fast)
    finally:
        if env is not None:
            env.close()
        dataset.stop_image_writer()
        write_run(root, {'mode': 'smoke' if smoke else 'keyboard', 'saved_episodes': episodes,
                         'recorded_frames_including_unsaved': frames, 'results': results,
                         'observation_timing': 'before_command', 'action_type': 'absolute_joint_targets',
                         'fps': 20})
    print(f'Dataset: {Path(root).resolve()}; saved episodes: {episodes}')
    if episodes == 0:
        print('No successful episode saved. An unfinished episode is intentionally not training data.')


def replay(args):
    import numpy as np
    import torch
    from PIL import Image
    from .dataset import load_dataset
    from .runtime import OMYEnvironment, PROJECT, pace

    root = args.dataset or PROJECT / 'data' / 'demo_data'
    dataset = load_dataset(root)
    if not 0 <= args.episode < dataset.num_episodes:
        raise ValueError(f'Episode index must be in [0, {dataset.num_episodes})')
    first = int(dataset.episode_data_index['from'][args.episode])
    last = int(dataset.episode_data_index['to'][args.episode])
    env = OMYEnvironment(seed=0, headless=args.headless, action_type='joint_angle')
    errors, count = [], 0
    try:
        data = dataset[first]
        initial = data['obj_init'].numpy()
        env.set_obj_pose(initial[:3], initial[3:])
        snapshot = Path(root) / 'initial_states' / f'episode_{args.episode:06d}.npz'
        if snapshot.exists():
            with np.load(snapshot, allow_pickle=False) as archive:
                env.restore(archive['state'])
        for i in range(first, last):
            if not env.alive():
                break
            start = time.perf_counter()
            data = dataset[i]
            # Dataset access stays in the main process on Windows (no DataLoader workers).
            action = data['action'].numpy().astype(np.float32)
            env.step(action)
            env.advance()
            observed = env.get_ee_pose()
            if i + 1 < last:
                target = dataset[i + 1]['observation.state'].numpy()
                errors.append(float(np.linalg.norm(observed[:3] - target[:3])))
            agent = (data['observation.image'].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            wrist = (data['observation.wrist_image'].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            env.show(agent, wrist)
            count += 1
            pace(start, args.fast)
        final_agent, final_wrist = env.images()
        report = {'dataset': str(Path(root).resolve()), 'episode': args.episode,
                  'expected_frames': last - first, 'replayed_frames': count,
                  'complete': count == last - first, 'final_task_success': env.check_success(),
                  'max_next_observation_position_error_m': max(errors, default=0),
                  'torch': torch.__version__}
        out = args.output or PROJECT / 'outputs' / 'replay_report.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        Image.fromarray(final_agent).save(out.with_name(out.stem + '_agent.png'))
        Image.fromarray(final_wrist).save(out.with_name(out.stem + '_wrist.png'))
        print(json.dumps(report, indent=2))
    finally:
        env.close()


def scripted_collect(args):
    import numpy as np
    from PIL import Image
    from .dataset import add_frame, create_dataset, write_run
    from .runtime import OMYEnvironment, PROJECT, pace
    from .scripted import scripted_actions

    root = args.dataset or PROJECT / 'data' / 'scripted_demo'
    dataset = create_dataset(root)
    env = None
    report = {'mode': 'scripted_ik', 'seed': args.seed, 'saved_episodes': 0,
              'not_a_learned_policy': True, 'fps': 20, 'state_type': 'ee_pose',
              'action_type': 'absolute_joint_targets', 'observation_timing': 'before_command'}
    frames = 0
    preview = []
    phase_starts = {}
    try:
        env = OMYEnvironment(seed=args.seed, headless=args.headless)
        initial_state = env.snapshot()
        initial_mug_height = float(env.get_obj_pose()[0][2])
        max_mug_height = initial_mug_height
        for phase, action in scripted_actions(env):
            phase_starts.setdefault(phase, frames)
            if not env.alive():
                raise RuntimeError('Window closed before scripted demonstration completed.')
            start = time.perf_counter()
            state = env.get_ee_pose().copy()
            agent, wrist = env.images()
            env.step(action)
            add_frame(dataset, env, agent, wrist, state)
            env.advance()
            env.show(agent, wrist)
            frames += 1
            if frames % 5 == 0:
                preview.append(Image.fromarray(agent))
            max_mug_height = max(max_mug_height, float(env.get_obj_pose()[0][2]))
            pace(start, args.fast)
        report.update({'frames': frames, 'task_predicate_success': env.check_success(),
                       'mug_lift_height_m': max_mug_height - initial_mug_height,
                       'phase_start_frames': phase_starts,
                       'success_debug': env.get_success_debug()})
        if not env.check_success() or report['mug_lift_height_m'] < 0.10:
            raise RuntimeError('Scripted grasp did not meet both lift and final task criteria; no successful episode saved.')
        dataset.save_episode()
        states = Path(root).resolve() / 'initial_states'
        states.mkdir(exist_ok=True)
        np.savez_compressed(states / 'episode_000000.npz', state=initial_state)
        report['saved_episodes'] = 1
        output = args.output or PROJECT / 'outputs'
        output.mkdir(parents=True, exist_ok=True)
        agent, wrist = env.images()
        Image.fromarray(agent).save(output / 'scripted_agent.png')
        Image.fromarray(wrist).save(output / 'scripted_wrist.png')
        if preview:
            preview[0].save(output / 'scripted_preview.gif', save_all=True,
                            append_images=preview[1:], duration=250, loop=0)
        print(json.dumps(report, indent=2))
    finally:
        if env is not None:
            env.close()
        dataset.stop_image_writer()
        write_run(root, report)


def main():
    args = parser().parse_args()
    if args.command == 'doctor':
        doctor()
    elif args.command in ('train-smolvla', 'deploy-smolvla', 'diagnose-smolvla'):
        from . import smolvla
        {'train-smolvla': smolvla.train, 'deploy-smolvla': smolvla.deploy,
         'diagnose-smolvla': smolvla.diagnose}[args.command](args)
    elif args.command in ('collect-language', 'replay-language', 'audit-language'):
        from . import language
        if args.command == 'collect-language':
            if args.headless and not args.scripted:
                raise ValueError('Keyboard collection requires a window; remove --headless or use --scripted.')
            language.collect(args)
        elif args.command == 'replay-language':
            language.replay(args)
        else:
            language.audit(args)
    elif args.command == 'collect':
        collect(args)
    elif args.command == 'smoke':
        collect(args, smoke=True)
    elif args.command == 'replay':
        replay(args)
    elif args.command == 'auto-demo':
        scripted_collect(args)
    elif args.command == 'train-act':
        from .act import train
        train(args)
    elif args.command == 'deploy-act':
        from .deploy_act import deploy
        deploy(args)
