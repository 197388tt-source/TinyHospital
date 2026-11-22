"""最小 embeddings API 连通性测试。

示例：
  export EMBEDDING_API_KEY='...'
  python -m embedding.test_api \
      --endpoint https://api.openai.com/v1/embeddings \
      --model text-embedding-3-small

也支持不带 /v1 的自建 OpenAI 兼容网关，但 endpoint 必须写完整的
embeddings 路径。脚本不会保存密钥、文本或返回向量。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="测试 embeddings API 是否可调用")
    parser.add_argument("--endpoint", required=True,
                        help="完整接口地址，例如 https://api.openai.com/v1/embeddings")
    parser.add_argument("--model", required=True, help="服务商提供的 embedding 模型名")
    parser.add_argument("--key-env", default="EMBEDDING_API_KEY",
                        help="API 密钥所在环境变量，默认 EMBEDDING_API_KEY")
    parser.add_argument("--text", default="患者连续三天咳嗽。",
                        help="测试文本；不要放入真实敏感病例")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    parsed = urlsplit(args.endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("--endpoint 必须是完整的 http/https URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("--endpoint 不应包含用户名、密码、查询参数或片段")
    if not args.model.strip() or not args.text.strip():
        parser.error("--model 和 --text 不能为空")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")

    api_key = os.environ.get(args.key_env)
    if not api_key:
        print(f"缺少环境变量 {args.key_env}；请先 export {args.key_env}='你的密钥'",
              file=sys.stderr)
        return 2

    body = json.dumps({
        "model": args.model,
        "input": [args.text],
        "encoding_format": "float",
    }, ensure_ascii=False).encode("utf-8")
    request = Request(args.endpoint, method="POST", data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })

    try:
        with urlopen(request, timeout=args.timeout) as response:
            value = json.load(response)
    except HTTPError as exc:
        # Do not print the response body: gateways sometimes echo credentials or prompts.
        print(f"API 请求失败：HTTP {exc.code}。请检查 endpoint、模型名和密钥。",
              file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"API 连接失败：{exc.reason}。请检查网络和 endpoint。", file=sys.stderr)
        return 1
    except (TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"API 响应读取失败：{exc}", file=sys.stderr)
        return 1

    data = value.get("data") if isinstance(value, dict) else None
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        print("API 返回格式不符合预期：需要 data 数组且只包含一个结果。", file=sys.stderr)
        return 1
    item = data[0]
    vector = item.get("embedding")
    if (not isinstance(vector, list) or not vector
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
        print("API 返回格式不符合预期：embedding 必须是非空数值数组。", file=sys.stderr)
        return 1
    if item.get("index") not in (None, 0):
        print("API 返回的 embedding index 不是 0。", file=sys.stderr)
        return 1

    result = {
        "ok": True,
        "model": value.get("model", args.model),
        "dimension": len(vector),
        "first_values": [round(float(x), 8) for x in vector[:5]],
        "usage": value.get("usage", "服务未返回 usage"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
