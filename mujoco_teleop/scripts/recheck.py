"""Rerun bounded training/inference/data checks in a NEW isolated output directory.

This does not retrain the 5,000/20,000-step experiments. Training smoke tests
verify real parameter updates; published-model rollouts verify task behavior.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise SystemExit('Use existing my_env Python 3.10.')
    tag = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = (args.output or ROOT / 'outputs' / f'recheck_{tag}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PYTHONIOENCODING='utf-8')
    report = {'output': str(output), 'scope': 'bounded end-to-end checks; no full retraining', 'steps': []}

    def save():
        (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def run(name, argv, stdin=None):
        started = time.perf_counter()
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run([str(value) for value in argv], cwd=WORKSPACE,
                                       env=environment, input=stdin, text=True, encoding='utf-8',
                                       stdout=log, stderr=subprocess.STDOUT)
        result = {'name': name, 'exit_code': completed.returncode,
                  'elapsed_seconds': time.perf_counter() - started, 'argv': [str(value) for value in argv]}
        report['steps'].append(result)
        save()
        print('RECHECK ' + json.dumps(result), flush=True)
        if completed.returncode:
            raise RuntimeError(f'{name} failed; inspect {output / (name + ".log")}')

    cli = [sys.executable, ROOT / 'run.py']
    py05 = ROOT / '.venv-pi05/Scripts/python.exe'
    single = output / 'single_data'
    language = output / 'language_data'
    try:
        run('doctor', cli + ['doctor'])
        run('unit_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', ROOT / 'tests', '-v'])
        run('pi05_interface_tests', [py05, ROOT / 'scripts/pi05_data.py', 'test'])
        legacy = "import runpy,sys;runpy.run_path('mujoco_teleop/run.py',run_name='bootstrap');sys.argv=['mujoco_demo/demo_embodied_mujoco.py','--headless','--fast','--autoplay','--autoplay-rounds','1'];runpy.run_path(sys.argv[0],run_name='__main__')"
        run('legacy_demo', [sys.executable, '-c', legacy], stdin='q\n')
        run('collect_single', cli + ['auto-demo', '--dataset', single, '--output', output / 'single_preview', '--headless', '--fast'])
        run('replay_single', cli + ['replay', '--dataset', single, '--output', output / 'single_replay.json', '--headless', '--fast'])
        run('train_act_smoke', cli + ['train-act', '--dataset', single, '--output', output / 'act_smoke', '--steps', '2', '--batch-size', '2', '--eval-every', '2'])
        run('deploy_act', cli + ['deploy-act', '--checkpoint', ROOT / 'checkpoints/act_omy_chunk100/final', '--output', output / 'act_rollout', '--headless', '--fast'])
        run('collect_language', cli + ['collect-language', '--dataset', language, '--episodes', '2', '--scripted', '--headless', '--fast'])
        for episode in (0, 1):
            run(f'replay_language_{episode}', cli + ['replay-language', '--dataset', language, '--episode', str(episode), '--output', output / f'language_replay_{episode}', '--headless', '--fast'])
        run('audit_language', cli + ['audit-language', '--dataset', language, '--output', output / 'language_audit.json'])
        run('train_smolvla_smoke', cli + ['train-smolvla', '--dataset', language, '--output', output / 'smol_smoke', '--steps', '5', '--batch-size', '1', '--holdout-episodes', '0', '--log-every', '1'])
        run('reload_smolvla_smoke', cli + ['deploy-smolvla', '--checkpoint', output / 'smol_smoke/final', '--output', output / 'smol_reload', '--max-steps', '5', '--headless', '--fast'])
        run('diagnose_smolvla', cli + ['diagnose-smolvla', '--checkpoint', output / 'smol_smoke/final', '--dataset', language, '--output', output / 'smol_diagnosis.json', '--samples', '4'])
        run('published_smolvla', cli + ['deploy-smolvla', '--checkpoint', ROOT / '.downloads/models/omy_pnp_smolvla', '--output', output / 'published_rollout', '--upstream-interface', '--headless', '--fast'])
        run('convert_pi05', [py05, ROOT / 'scripts/pi05_data.py', 'prepare', '--source', single, '--destination', output / 'pi05_v3', '--labels-verified'])
        run('pi_runtime_check', [sys.executable, ROOT / 'scripts/check_vla.py', '--output', output / 'pi_readiness.json'])
        load = lambda path: json.loads(path.read_text(encoding='utf-8'))
        outcomes = {'single_replay': load(output / 'single_replay.json')['final_task_success'],
                    'act_successes': load(output / 'act_rollout/deployment_report.json')['grasp_successes'],
                    'language_replay_errors': [load(output / f'language_replay_{e}/replay_report.json')['max_next_joint_state_error_rad'] for e in (0, 1)],
                    'published_physical_successes': load(output / 'published_rollout/deployment_report.json')['physical_successes'],
                    'act_training_complete': load(output / 'act_smoke/training_run.json')['complete'],
                    'smol_training_complete': load(output / 'smol_smoke/training_run.json')['complete']}
        report['outcomes'] = outcomes
        report['checks_passed'] = bool(outcomes['single_replay'] and outcomes['act_successes'] == 1
                                       and outcomes['published_physical_successes'] == 2
                                       and max(outcomes['language_replay_errors']) < 1e-6
                                       and outcomes['act_training_complete'] and outcomes['smol_training_complete'])
        if not report['checks_passed']:
            raise RuntimeError('Commands finished but behavior checks failed; inspect outcomes')
        report['complete'] = True
    except Exception as error:
        report.update(complete=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    print('RECHECK_COMPLETE ' + json.dumps(report['outcomes']), flush=True)


if __name__ == '__main__':
    main()
