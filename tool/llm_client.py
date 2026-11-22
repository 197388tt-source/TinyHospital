"""Minimal OpenAI-compatible chat client for the school model gateway."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from tool.api_retry import request_json


DEFAULT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://llmapi.paratera.com/v1/")
DEFAULT_MODEL = os.getenv("CHAT_MODEL", "DeepSeek-V4.1-Flash-1017")


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("DLUT_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.last_usage = TokenUsage()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise ValueError("请设置 DLUT_API_KEY 或 OPENAI_API_KEY；仓库不包含密钥")
        if not messages:
            raise ValueError("messages 不能为空")
        payload = json.dumps(
            {"model": self.model, "messages": messages}, ensure_ascii=False
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            result = request_json(urlopen, request, self.timeout)
        except HTTPError as exc:
            raise RuntimeError(f"学校 API 请求失败（HTTP {exc.code}）") from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError('学校聊天 API 连接超时') from None
            raise RuntimeError(f"无法连接学校 API：{exc.reason}") from None

        try:
            reply = result["choices"][0]["message"]["content"]
            usage = result.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("学校 API 返回格式不符合 Chat Completions 约定") from exc
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("学校 API 返回了空回复")

        self.last_usage = TokenUsage(prompt_tokens, completion_tokens, total_tokens)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        return reply.strip()
