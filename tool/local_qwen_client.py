"""Local Qwen3 non-thinking chat adapter for the Doctor agent."""

import os
from pathlib import Path
import re

from tool.llm_client import TokenUsage


DEFAULT_MODEL_PATH = Path(os.getenv('QWEN_MODEL_PATH', str(Path(__file__).resolve().parents[1] / 'models/Qwen3-8B')))


class LocalQwenClient:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, max_new_tokens=256, timeout=120):
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise ValueError(f'本地模型目录不存在：{model_path}')
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens 必须大于 0')
        try:
            import torch
            import accelerate  # noqa: F401 -- device_map dependency
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('请在 tiny-hospital 环境安装 torch、transformers>=4.51.0 和 accelerate') from exc
        self.torch = torch
        self.model_path = str(model_path)
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.last_usage = TokenUsage()
        self.last_raw_reply = ''
        print(f'正在加载本地 Doctor：{model_path}', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype='auto', device_map='auto', local_files_only=True,
        )
        self.model.eval()

    def chat(self, messages):
        self.last_usage = TokenUsage()
        self.last_raw_reply = ''
        if not messages:
            raise ValueError('messages 不能为空')
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(
            [prompt], return_tensors='pt', add_special_tokens=False,
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, max_time=self.timeout,
                # Clear sampling settings inherited from generation_config.json.
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_tokens = inputs['input_ids'].shape[1]
        generated = output[0, prompt_tokens:]
        completion_tokens = len(generated)
        self.last_usage = TokenUsage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)
        reply = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        self.last_raw_reply = reply
        if not reply:
            raise ValueError('本地 Qwen3 返回空回复')
        if reply.startswith('```'):
            reply = re.sub(r'^```(?:json)?\s*|\s*```$', '', reply, flags=re.IGNORECASE).strip()
        return reply
