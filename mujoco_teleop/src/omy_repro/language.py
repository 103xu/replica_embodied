"""Real two-cup collection/replay with instruction labels; never a learned policy."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from .dataset import FEATURES, write_run
from .runtime import LanguageEnvironment, PROJECT, pace
from .scripted import scripted_actions
from .preview import OverviewPreview

REPO_ID = 'datawhale_eai_pnp_language'


def load_language(root):
    root = Path(root).resolve()
    provenance = root / 'reproduction.json'
    if not provenance.is_file() or json.loads(provenance.read_text(encoding='utf-8')).get('state_type') != 'joint_angle':
        raise ValueError('Expected a language dataset with explicitly recorded joint-angle state semantics.')
    return LeRobotDataset(repo_id=REPO_ID, root=root, video_backend='pyav')


def task_status(env, initial_height, max_height):
    red, blue, plate = env.get_obj_pose()
    target = red if env.obj_target == 'body_obj_mug_5' else blue
    upright_cosine = float(env.env.get_R_body(env.obj_target)[2, 2])
    return {'target': env.obj_target, 'target_xy_distance_m': float(np.linalg.norm(target[:2] - plate[:2])),
            'target_z_distance_m': float(abs(target[2] - plate[2])),
            'target_up_cosine': upright_cosine, 'upright': upright_cosine >= 0.85,
            'legacy_success': bool(env.check_success()),
            'mug_lift_height_m': max_height - initial_height,
            'strict_success': bool(np.linalg.norm(target[:2] - plate[:2]) < 0.10
                                   and abs(target[2] - plate[2]) < 0.12
                                   and env.gripper_cmd_scalar <= 0.10
                                   and env.get_ee_pose()[2] >= 0.95
                                   and max_height - initial_height >= 0.10)}


def target_height(env):
    return float(env.env.get_p_body(env.obj_target)[2])


def physical_candidate(status, consecutive_lift_ticks):
    """Section-9 physical rule, distinct from the older 10cm ACT confirmation."""
    return bool(status['legacy_success'] and status['upright'] and consecutive_lift_ticks >= 3)


def collect(args):
    root = (args.dataset or PROJECT / 'data' / 'language_demo').resolve()
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite {root}; choose a new --dataset.')
    if args.episodes < 1:
        raise ValueError('episodes must be positive')
    features = {**FEATURES, 'obj_init': {'dtype': 'float32', 'shape': (9,), 'names': ['obj_init']}}
    root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(repo_id=REPO_ID, root=root, robot_type='omy', fps=20,
                                    features=features, use_videos=False,
                                    image_writer_processes=0, image_writer_threads=2)
    env = None
    overview = None
    results = []
    report = {'mode': 'scripted_ik' if args.scripted else 'keyboard', 'state_type': 'joint_angle',
              'action_type': 'absolute_joint_targets', 'observation_timing': 'before_command',
              'fps': 20, 'not_a_learned_policy': True, 'episodes': results, 'saved_episodes': 0}
    try:
        env = LanguageEnvironment(seed=args.seed, headless=args.headless)
        if getattr(args, 'overview_preview', False):
            overview = OverviewPreview(env)
        for episode in range(args.episodes):
            seed = args.seed + episode // 2
            env.reset(seed=seed)
            color = ['red', 'blue'][episode % 2] if args.color == 'alternate' else args.color
            env.set_instruction(f'Place the {color} mug on the plate.')
            snapshot = env.snapshot()
            initial_height = maximum = target_height(env)
            preview, frames, stable = [], 0, 0
            started = args.scripted
            stream = scripted_actions(env) if args.scripted else None
            while env.alive():
                start = time.perf_counter()
                if stream is not None:
                    try:
                        _, action = next(stream)
                    except StopIteration:
                        break
                else:
                    action, reset = env.teleop_robot()
                    if reset:
                        dataset.clear_episode_buffer()
                        env.reset(seed=seed)
                        env.set_instruction(f'Place the {color} mug on the plate.')
                        snapshot = env.snapshot()
                        initial_height = maximum = target_height(env)
                        started, frames, stable = False, 0, 0
                        continue
                    started = started or bool(np.any(action != 0))
                state = env.get_state()
                agent, wrist = env.images()
                env.step(action)
                if started:
                    dataset.add_frame({'observation.image': agent, 'observation.wrist_image': wrist,
                                       'observation.state': state, 'action': env.get_commanded_joint_action(),
                                       'obj_init': env.obj_init_pose.copy()}, task=env.instruction)
                    frames += 1
                    if frames % 10 == 0:
                        preview.append(overview.frame() if overview else Image.fromarray(agent))
                env.advance()
                env.show(agent, wrist)
                maximum = max(maximum, target_height(env))
                status = task_status(env, initial_height, maximum)
                stable = stable + 1 if status['strict_success'] else 0
                if stream is None and started and stable >= 20:
                    break
                pace(start, args.fast)
            status = task_status(env, initial_height, maximum)
            if not started or not status['strict_success']:
                raise RuntimeError(f'Unsuccessful {color} demonstration; not saved: {status}')
            dataset.save_episode()
            states = root / 'initial_states'
            states.mkdir(exist_ok=True)
            np.savez_compressed(states / f'episode_{episode:06d}.npz', state=snapshot)
            if preview:
                preview[0].save(states / f'episode_{episode:06d}.gif', save_all=True,
                                append_images=preview[1:], duration=500, loop=0)
            results.append({'episode': episode, 'seed': seed, 'instruction': env.instruction,
                            'frames': frames, 'stable_frames_at_end': stable, **status})
            report['saved_episodes'] = len(results)
            print('LANGUAGE_EPISODE ' + json.dumps(results[-1]), flush=True)
    finally:
        if overview is not None:
            overview.close()
        if env is not None:
            env.close()
        dataset.stop_image_writer()
        write_run(root, report)
    print('LANGUAGE_COLLECTION_COMPLETE ' + json.dumps(report), flush=True)


def replay(args):
    root = args.dataset or PROJECT / 'data' / 'language_demo'
    dataset = load_language(root)
    if not 0 <= args.episode < dataset.num_episodes:
        raise ValueError('episode outside dataset')
    first, last = [int(dataset.episode_data_index[key][args.episode]) for key in ('from', 'to')]
    output = args.output or PROJECT / 'outputs' / 'language_replay'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    env = LanguageEnvironment(seed=0, headless=args.headless, action_type='joint_angle')
    overview = None
    errors, frames, maximum, preview = [], 0, 0, []
    try:
        if getattr(args, 'overview_preview', False):
            overview = OverviewPreview(env)
        env.set_instruction(dataset[first]['task'])
        with np.load(Path(root) / 'initial_states' / f'episode_{args.episode:06d}.npz', allow_pickle=False) as archive:
            env.restore(archive['state'])
        initial = maximum = target_height(env)
        for index in range(first, last):
            if not env.alive():
                break
            start = time.perf_counter()
            frame = dataset[index]
            env.step(frame['action'].numpy().astype(np.float32))
            env.advance()
            maximum = max(maximum, target_height(env))
            if index + 1 < last:
                errors.append(float(np.linalg.norm(env.get_state() - dataset[index + 1]['observation.state'].numpy())))
            agent, wrist = env.images()
            env.show(agent, wrist)
            if frames % 10 == 0:
                preview.append(overview.frame() if overview else Image.fromarray(agent))
            frames += 1
            pace(start, args.fast)
        agent, wrist = env.images()
        Image.fromarray(agent).save(output / 'agent.png')
        Image.fromarray(wrist).save(output / 'wrist.png')
        if preview:
            preview[0].save(output / 'replay.gif', save_all=True, append_images=preview[1:], duration=500, loop=0)
        report = {'dataset': str(Path(root).resolve()), 'episode': args.episode, 'instruction': env.instruction,
                  'complete': frames == last - first, 'frames': frames, 'state_type': 'joint_angle',
                  'max_next_joint_state_error_rad': max(errors, default=0), **task_status(env, initial, maximum)}
        (output / 'replay_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if overview is not None:
            overview.close()
        env.close()


def audit(args):
    """Replay all recorded joint labels without decoding images; audit physical outcomes."""
    import pyarrow.parquet as pq
    root = args.dataset.resolve()
    load_language(root)  # Check the explicit state/action contract.
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    tasks = {row['task_index']: row['task'] for row in
             (json.loads(line) for line in (root / 'meta/tasks.jsonl').read_text(encoding='utf-8').splitlines())}
    env = LanguageEnvironment(seed=0, headless=True, action_type='joint_angle')
    results = []
    try:
        for episode in range(info['total_episodes']):
            relative = info['data_path'].format(episode_chunk=episode // info['chunks_size'], episode_index=episode)
            rows = pq.read_table(root / relative, columns=['action', 'observation.state', 'task_index']).to_pylist()
            env.reset(seed=0)
            env.set_instruction(tasks[rows[0]['task_index']])
            with np.load(root / 'initial_states' / f'episode_{episode:06d}.npz', allow_pickle=False) as archive:
                env.restore(archive['state'])
            initial = maximum = target_height(env)
            errors, lift_ticks, lift_run, max_lift_run = [], 0, 0, 0
            for index, row in enumerate(rows):
                if row['task_index'] != rows[0]['task_index']:
                    raise ValueError('An episode has inconsistent language labels')
                env.step(np.asarray(row['action'], dtype=np.float32))
                env.advance()
                height = target_height(env)
                maximum = max(maximum, height)
                lift_ticks += height - initial >= 0.03
                lift_run = lift_run + 1 if height - initial >= 0.03 else 0
                max_lift_run = max(max_lift_run, lift_run)
                if index + 1 < len(rows):
                    errors.append(float(np.linalg.norm(env.get_state() - rows[index + 1]['observation.state'])))
            status = task_status(env, initial, maximum)
            result = {'episode': episode, 'instruction': env.instruction, 'frames': len(rows),
                      'max_next_joint_state_error_rad': max(errors, default=0), 'lift_ticks_above_3cm': int(lift_ticks),
                      'max_consecutive_lift_ticks': int(max_lift_run),
                      **status, 'physical_success': physical_candidate(status, max_lift_run)}
            results.append(result)
            print('AUDIT_EPISODE ' + json.dumps(result), flush=True)
        report = {'dataset': str(root), 'uses_learned_policy': False, 'episodes': results,
                  'strict_successes': sum(row['strict_success'] for row in results),
                  'physical_successes': sum(row['physical_success'] for row in results),
                  'physical_rule': 'legacy success + lift >=3cm for >=3 ticks + final up-axis cosine >=0.85',
                  'max_next_joint_state_error_rad': max(row['max_next_joint_state_error_rad'] for row in results)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        env.close()
