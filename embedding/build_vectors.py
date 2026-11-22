"""python -m embedding.build_vectors build/inspect --help"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .encoder import EmbeddingConfig, Vectorizer
from .records import load_records
from .store import VectorStore, inspect_store


def main(argv=None):
    parser = argparse.ArgumentParser(description='独立语义向量模块：原始数据与 Agent 流程不变')
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='只读查看向量库摘要，不加载模型')
    inspect.add_argument('--output', type=Path, required=True)
    build = commands.add_parser('build', help='增量编码，保存到独立 SQLite 向量库')
    build.add_argument('--source', type=Path, required=True)
    build.add_argument('--kind', choices=('dialog', 'experience', 'reflection', 'text'), required=True)
    build.add_argument('--namespace', required=True, help='例如 train-dialog；测试数据应单独命名')
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--backend', choices=('local', 'api'), required=True)
    build.add_argument('--model', required=True, help='本地模型目录/Hugging Face ID 或 API 模型名')
    build.add_argument('--model-version', required=True, help='固定模型版本标记；权重变更必须更新')
    build.add_argument('--revision', default='', help='本地后端的远端模型提交版本')
    build.add_argument('--allow-download', action='store_true', help='允许本地后端下载模型，默认仅使用本地文件')
    build.add_argument('--device', default='cpu')
    build.add_argument('--endpoint', default='', help='API 后端的完整 embeddings URL')
    build.add_argument('--api-key-env', default='EMBEDDING_API_KEY')
    build.add_argument('--timeout', type=float, default=120)
    build.add_argument('--batch-size', type=int, default=32)
    build.add_argument('--chunk-chars', type=int, default=256)
    build.add_argument('--prefix', default='', help='按模型说明设置的编码前缀；会计入缓存指纹')
    build.add_argument('--limit', type=int, help='仅导入前 N 条；不删除已有记录')
    build.add_argument('--prune', action='store_true', help='同步完整源文件，移除本命名空间中已删除的记录')
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            result = inspect_store(args.output)
        else:
            if args.limit is not None and args.limit < 1:
                raise ValueError('limit 必须大于零')
            if args.limit is not None and args.prune:
                raise ValueError('limit 与 prune 不能同时使用，避免误删未导入记录')
            config = EmbeddingConfig(args.backend, args.model, args.model_version,
                                     chunk_chars=args.chunk_chars, prefix=args.prefix,
                                     endpoint=args.endpoint, revision=args.revision,
                                     device=args.device, local_files_only=not args.allow_download)
            records = load_records(args.source, args.kind)
            if args.limit is not None:
                records = records[:args.limit]
            encoder = Vectorizer(config, args.batch_size, args.api_key_env, args.timeout)
            with VectorStore(args.output, config) as store:
                result = store.sync(args.namespace, records, encoder, args.prune)
            result['output'] = str(args.output.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f'向量化失败：{exc}\n')


if __name__ == '__main__':
    main()
