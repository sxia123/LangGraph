import pytest
from src.core.local_llm import LocalLLMClient, LocalLLMConfig


def test_get_openai_url_normalization():
    # Standard base URL with /v1
    client = LocalLLMClient(LocalLLMConfig(base_url="http://127.0.0.1:8000/v1"))
    assert client._get_openai_url("chat/completions") == "http://127.0.0.1:8000/v1/chat/completions"

    # Base URL without /v1
    client = LocalLLMClient(LocalLLMConfig(base_url="http://127.0.0.1:8000"))
    assert client._get_openai_url("chat/completions") == "http://127.0.0.1:8000/v1/chat/completions"

    # Base URL with trailing slash
    client = LocalLLMClient(LocalLLMConfig(base_url="http://127.0.0.1:8000/v1/"))
    assert client._get_openai_url("chat/completions") == "http://127.0.0.1:8000/v1/chat/completions"

    # Base URL already including /chat/completions
    client = LocalLLMClient(LocalLLMConfig(base_url="http://127.0.0.1:8000/v1/chat/completions"))
    assert client._get_openai_url("chat/completions") == "http://127.0.0.1:8000/v1/chat/completions"

    # Base URL with /chatcompletions typo
    client = LocalLLMClient(LocalLLMConfig(base_url="http://127.0.0.1:8000/chatcompletions"))
    assert client._get_openai_url("chat/completions") == "http://127.0.0.1:8000/v1/chat/completions"
