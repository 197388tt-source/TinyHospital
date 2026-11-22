import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool.llm_client import LLMClient, TokenUsage  # noqa: E402


class LLMClientTests(unittest.TestCase):
    def test_chat_request_and_accumulated_usage(self):
        response = {
            "choices": [{"message": {"content": "流行性感冒、普通感冒"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        }
        payload = json.dumps(response).encode("utf-8")
        client = LLMClient(api_key="test-key", base_url="https://example.test/v1", model="test-model")
        messages = [{"role": "user", "content": "患者发热"}]
        with patch("tool.llm_client.urlopen", side_effect=lambda *_args, **_kwargs: io.BytesIO(payload)) as opened:
            self.assertEqual(client.chat(messages), "流行性感冒、普通感冒")
            self.assertEqual(client.chat(messages), "流行性感冒、普通感冒")
        request = opened.call_args.args[0]
        self.assertEqual(opened.call_args.kwargs["timeout"], 120)
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data)["model"], "test-model")
        self.assertEqual(client.last_usage, TokenUsage(11, 4, 15))
        self.assertEqual((client.prompt_tokens, client.completion_tokens, client.total_tokens), (22, 8, 30))

    def test_missing_credentials_fail_before_network(self):
        with patch.dict('os.environ', {}, clear=True), patch('tool.llm_client.urlopen') as opened:
            with self.assertRaisesRegex(ValueError, 'DLUT_API_KEY'):
                LLMClient().chat([{'role': 'user', 'content': 'test'}])
            opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
