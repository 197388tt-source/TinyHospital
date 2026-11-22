"""Restore deduplicated inputs with SHA-256 checks; never overwrite changed files."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(root=ROOT, check=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'assets_manifest.json').read_text(encoding='utf-8'))
    outputs = {}

    def safe_path(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'路径越界：{relative}')
        return path

    for dest, spec in manifest.get('compressed', {}).items():
        packed = safe_path(spec['source']).read_bytes()
        if digest(packed) != spec['source_sha256']:
            raise ValueError(f'压缩源文件校验失败：{spec["source"]}')
        outputs[dest] = (gzip.decompress(packed), spec['sha256'])
    for dest, spec in manifest['subsets'].items():
        source = outputs[spec['source']][0] if spec['source'] in outputs else safe_path(spec['source']).read_bytes()
        if digest(source) != spec['source_sha256']:
            raise ValueError(f'源数据校验失败：{spec["source"]}')
        lines = source.splitlines(keepends=True)
        if any(type(i) is not int or not 1 <= i <= len(lines) for i in spec['lines']):
            raise ValueError(f'子集行号无效：{dest}')
        outputs[dest] = (b''.join(lines[i - 1] for i in spec['lines']), spec['sha256'])
    for dest, spec in manifest['copies'].items():
        source = outputs[spec['source']][0] if spec['source'] in outputs else safe_path(spec['source']).read_bytes()
        outputs[dest] = (source, spec['sha256'])
    # Validate every destination before making any changes.
    for dest, (raw, expected) in outputs.items():
        path = safe_path(dest)
        if digest(raw) != expected:
            raise ValueError(f'待恢复文件校验失败：{dest}')
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f'保留已修改文件，停止恢复：{dest}。请在干净解压目录准备资源。')
    count = 0
    for dest, (raw, _) in outputs.items():
        path = safe_path(dest)
        if not path.exists() and not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw)
            count += 1
    print(f'校验通过：{len(outputs)} 个资源；新恢复 {count} 个。')
    return len(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='仅验证源文件与恢复方案，不写入')
    args = parser.parse_args()
    prepare(check=args.check)
