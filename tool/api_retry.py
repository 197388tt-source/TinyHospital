"""Bounded retries for transient transport errors; never log credentials."""
import json
import os
import time
from http.client import RemoteDisconnected, IncompleteRead
from urllib.error import HTTPError, URLError


def request_json(opener, request, timeout):
    retries = int(os.environ.get('TINY_HOSPITAL_API_RETRIES', '2'))
    if not 0 <= retries <= 5:
        raise ValueError('TINY_HOSPITAL_API_RETRIES 必须为 0 到 5')
    for attempt in range(retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, ConnectionError,
                RemoteDisconnected, IncompleteRead) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == retries:
                raise
            delay = min(2 ** (attempt + 1), 30)
            print(f'[API 重试] {type(exc).__name__}；{delay} 秒后重试 {attempt + 1}/{retries}', flush=True)
            time.sleep(delay)
