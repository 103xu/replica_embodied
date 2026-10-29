"""Check upstream VLA configs and optionally model access without downloading large weights."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import importlib


def pi0_runtime_check(torch):
    """Count the real pinned architecture on meta, without downloading/allocating weights."""
    try:
        module = importlib.import_module('lerobot.common.policies.pi0.modeling_pi0')
    except Exception as error:
        return {'model_import_ok': False, 'error_type': type(error).__name__, 'error': str(error)}
    from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
    result = {'model_import_ok': True, 'meta_only': True, 'actual_training_attempted': False,
              'estimate_excludes': ['activations', 'CUDA workspaces', 'allocator overhead'], 'configurations': []}
    for expert_only in (False, True):
        with torch.device('meta'):
            model = module.PI0FlowMatching(PI0Config(device='cpu', chunk_size=5, n_action_steps=5,
                                                     train_expert_only=expert_only))
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        # The expert lm_head is unused in the pinned custom forward: do not
        # count its nonexistent gradients/optimizer moments.
        gradients = sum(p.numel() * p.element_size() for name, p in model.named_parameters()
                        if p.requires_grad and 'gemma_expert.lm_head' not in name)
        result['configurations'].append({'train_expert_only': expert_only,
            'parameters': sum(p.numel() for p in model.parameters()), 'weight_gib': weights / 2**30,
            'estimated_weights_gradients_adam_gib': (weights + 3 * gradients) / 2**30,
            'optimizer_assumption': 'two Adam moment tensors matching parameter dtype'})
        del model
    return result


def pi05_runtime_check():
    python = ROOT / '.venv-pi05' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        return {'environment_exists': False}
    code = '''
import importlib.metadata,json,torch,pathlib,sys
from packaging.requirements import Requirement
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.import_utils import require_package
r={'environment_exists':True,'purpose':'CPU data/tests only','torch':torch.__version__,
   'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
pin=next(Requirement(line.strip()) for line in pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line.startswith('torchcodec=='))
version=next(iter(pin.specifier)).version
conflicts=[]
for line in importlib.metadata.requires('lerobot') or []:
    req=Requirement(line)
    if req.name=='torchcodec' and (req.marker is None or req.marker.evaluate({'extra':'dataset'})) and version not in req.specifier:
        conflicts.append({'pinned':str(pin),'required_by_lerobot_dataset':str(req),'platform':sys.platform})
r['upstream_requirements_conflicts']=conflicts
try:
    # This is the actual policy constructor's dependency gate. Construct the
    # core on meta afterwards: Policy.__init__ would try moving meta to CPU.
    require_package('transformers',extra='pi')
    with torch.device('meta'):
        model=PI05Pytorch(PI05Config(device='cpu',dtype='bfloat16',train_expert_only=True))
    r['meta_construction_ok']=True
    r['parameters']=sum(p.numel() for p in model.parameters())
except Exception as e:
    r.update(meta_construction_ok=False,error_type=type(e).__name__,error=str(e))
print(json.dumps(r))
'''
    completed = subprocess.run([str(python), '-c', code, str(ROOT / 'vendor/every_embodied/requirements_pi05.txt')], capture_output=True, text=True)
    if completed.returncode:
        return {'environment_exists': True, 'check_exit_code': completed.returncode, 'error': completed.stderr[-1500:]}
    return json.loads(completed.stdout.strip().splitlines()[-1])

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
    results['pi0_runtime'] = pi0_runtime_check(torch)
    results['pi05_runtime'] = pi05_runtime_check()
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
        from huggingface_hub import get_token
        tokenizer_access = {'repo_id': 'google/paligemma-3b-pt-224', 'credential_available': bool(get_token()),
                            'probe': 'actual AutoTokenizer.from_pretrained call used by pinned PI0Policy'}
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')
            tokenizer_access['accessible'] = True
        except Exception as error:
            status, cause = None, error
            for _ in range(8):
                if getattr(cause, 'response', None) is not None:
                    status = cause.response.status_code
                cause = cause.__cause__ or cause.__context__
                if cause is None:
                    break
            tokenizer_access.update(accessible=False, error_type=type(error).__name__, http_status=status)
        results['pi0_tokenizer_access'] = tokenizer_access
    default_estimate = results['pi0_runtime'].get('configurations', [{}])[0].get('estimated_weights_gradients_adam_gib')
    results['pi0_ready_under_tutorial_defaults'] = bool(default_estimate is not None
        and default_estimate < results['gpu_total_gib']
        and results.get('pi0_tokenizer_access', {}).get('accessible') is True)
    pi05_environment = results['pi05_runtime']
    results['pi05_gpu_environment_ready'] = bool(pi05_environment.get('cuda_available')
                                                and pi05_environment.get('meta_construction_ok'))
    # Even a complete GPU environment still needs a real memory/training check.
    results['pi05_ready_under_tutorial_defaults'] = None if results['pi05_gpu_environment_ready'] else False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
