#!/usr/bin/env python3
"""Download the pinned public checkpoint and describe the installed Liger code.

This prepares local paths only. It does not load the model or run an experiment.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    model = 'Qwen/Qwen2.5-0.5B'
    revision = '060db6499f32faf8b98477b0a26969ef7d8b9987'
    spec = importlib.util.find_spec('liger_kernel')
    if spec is None or spec.origin is None:
        parser.error('Install requirements-real-model.txt in this environment first.')
    package = Path(spec.origin).resolve().parent
    snapshot = snapshot_download(
        model, revision=revision,
        cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None,
        allow_patterns=['*.json', '*.safetensors', 'tokenizer*', 'merges.txt',
                        'vocab.json', '*.model'])
    sources = {}
    for name in ['cross_entropy', 'fused_linear_cross_entropy']:
        path = package / 'ops' / (name + '.py')
        sources[name] = {'path': str(path),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    record = {
        'model': model, 'revision': revision, 'snapshot': snapshot,
        'deps': str(package.parent),
        'liger_version': importlib.metadata.version('liger-kernel'),
        'sources': sources,
        'config': json.loads((Path(snapshot) / 'config.json').read_text())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'prepared': str(args.out.resolve()), 'model': model,
                      'revision': revision, 'liger': record['liger_version']}))


if __name__ == '__main__':
    main()
