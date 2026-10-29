"""Check upstream VLA configs and optionally model access without downloading large weights."""
import argparse
import json
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/vla_preflight.json')
    args = parser.parse_args()
    runpy.run_path(str(ROOT / 'run.py'), run_name='bootstrap')
    if args.online:
        # HF reads this flag at import time; set it before config imports.
        os.environ['HF_HUB_OFFLINE'] = '0'
    import torch
    from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs import parser as config_parser
    from lerobot.configs.train import TrainPipelineConfig
    from unittest.mock import patch
    import sys

    @config_parser.wrap()
    def check(cfg: TrainPipelineConfig):
        cfg.validate()
        return {'policy': cfg.policy.type, 'chunk_size': cfg.policy.chunk_size,
                'batch_size': cfg.batch_size, 'steps': cfg.steps, 'configuration_parses': True,
                'training_verified': False}

    results = {'scope': 'configuration and model access only; not training success', 'policies': []}
    for policy in ('pi0', 'smolvla'):
        config = ROOT / 'vendor/every_embodied' / f'{policy}_datawhale_eai.yaml'
        with patch.object(sys, 'argv', [sys.argv[0], '--config_path', str(config)]):
            results['policies'].append(check())
    results['gpu_total_gib'] = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
    results['pi05_upstream_short_check_peak_gib'] = 12.85
    results['pi05_full_training_locally_verified'] = False
    if args.online:
        from huggingface_hub import HfApi
        api = HfApi()
        access = []
        for repo in ('lerobot/pi0', 'google/paligemma-3b-pt-224', 'lerobot/pi05_base'):
            try:
                info = api.model_info(repo, files_metadata=True)
                entry = {'repo_id': repo, 'revision': info.sha, 'gated': info.gated,
                         'weight_bytes': sum(f.size or 0 for f in info.siblings if f.rfilename.endswith('.safetensors'))}
                # A metadata listing does not prove gated-file access.
                import urllib.request
                url = f'https://huggingface.co/{repo}/resolve/{info.sha}/config.json'
                try:
                    with urllib.request.urlopen(url, timeout=30) as response:
                        entry['anonymous_config_http_status'] = response.status
                        response.read(100000)
                except Exception as error:
                    entry['anonymous_config_http_status'] = getattr(error, 'code', None)
                    entry['config_access_error'] = type(error).__name__
                access.append(entry)
            except Exception as error:
                access.append({'repo_id': repo, 'error': type(error).__name__})
        results['model_access'] = access
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
