"""Patch script — fixes HF datasets import collision with local lens_datasets package."""
from pathlib import Path
import sys

def patch_hf_import(filepath):
    path = Path(filepath)
    code = path.read_text(encoding='utf-8')

    # Find and replace the HF_AVAILABLE try/except block
    old_patterns = [
        (
            'try:\n    from datasets import load_dataset, IterableDataset as HFIterableDataset\n    HF_AVAILABLE = True\nexcept ImportError:\n    HF_AVAILABLE = False\n    log.warning("pip install datasets huggingface_hub  \u2190 required for HF streaming")',
            '''try:
    # Remove local ml/ paths temporarily to avoid collision with lens_datasets/
    import sys as _sys
    _removed = []
    for _p in list(_sys.path):
        if 'isafe2' in _p and ('ml' in _p.lower() or 'lens_datasets' in _p.lower()):
            _sys.path.remove(_p)
            _removed.append(_p)
    from datasets import load_dataset, IterableDataset as HFIterableDataset
    for _p in _removed:
        _sys.path.insert(0, _p)
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False'''
        ),
        (
            'try:\n    from datasets import load_dataset\n    HF_AVAILABLE = True\nexcept ImportError:\n    HF_AVAILABLE = False',
            '''try:
    import sys as _sys
    _removed = []
    for _p in list(_sys.path):
        if 'isafe2' in _p and ('ml' in _p.lower() or 'lens_datasets' in _p.lower()):
            _sys.path.remove(_p)
            _removed.append(_p)
    from datasets import load_dataset
    for _p in _removed:
        _sys.path.insert(0, _p)
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False'''
        ),
    ]

    changed = False
    for old, new in old_patterns:
        if old in code:
            code = code.replace(old, new)
            changed = True
            break

    if changed:
        path.write_text(code, encoding='utf-8')
        print(f"Patched: {path.name}")
    else:
        # Show relevant lines for debugging
        print(f"Pattern not found in {path.name}. Relevant lines:")
        for i, line in enumerate(code.split('\n'), 1):
            if 'HF_AVAILABLE' in line or ('from datasets' in line and 'lens_datasets' not in line):
                print(f"  L{i}: {repr(line)}")


patch_hf_import('d:/isafe2/ml/lens_datasets/remote/hf_streaming_loader.py')
patch_hf_import('d:/isafe2/ml/lens_datasets/remote/hf_audio_loader.py')

# Verify fix
sys.path.insert(0, 'd:/isafe2/ml')
from lens_datasets.remote.hf_streaming_loader import HF_AVAILABLE, HF_DATASETS
print(f"HF_AVAILABLE = {HF_AVAILABLE}")
print(f"Datasets registered: {list(HF_DATASETS.keys())}")
