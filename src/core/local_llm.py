import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel


class Message(BaseModel):
    id: str
    sender: str
    role: str
    content: str
    timestamp: str


class LLMResponse(BaseModel):
    content: str
    thought: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LocalLLMConfig(BaseModel):
    provider: str = "lm-studio"
    base_url: str = "http://localhost:1234/v1"
    model_name: str = "qwen2.5-coder-7b-instruct"
    api_key: Optional[str] = ""
    temperature: float = 0.2


class LocalLLMClient:
    def __init__(self, config: Optional[LocalLLMConfig] = None):
        if config is None:
            config = LocalLLMConfig(
                provider=os.getenv("LLM_PROVIDER", "lm-studio"),
                base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
                model_name=os.getenv("OPENAI_MODEL_NAME", "qwen2.5-coder-7b-instruct"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
            )
        self.config = config

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k) and v is not None:
                setattr(self.config, k, v)

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _get_openai_url(self, endpoint: str) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/{endpoint.lstrip('/')}"
        return f"{base}/v1/{endpoint.lstrip('/')}"

    def ping(self) -> Dict[str, Any]:
        if self.config.provider == "mock":
            return {
                "ok": True,
                "message": "Mock engine active (Offline visual simulation)",
                "models": ["mock-llama3.2", "mock-qwen2.5-coder", "mock-deepseek-r1"],
            }

        try:
            if self.config.provider == "ollama":
                base = self.config.base_url.rstrip("/")
                res = requests.get(f"{base}/api/tags", headers=self._get_headers(), timeout=5)
                if res.status_code == 200:
                    models = [m.get("name") for m in res.json().get("models", [])]
                    return {
                        "ok": True,
                        "message": f"Connected to Ollama ({len(models)} models detected)",
                        "models": models,
                    }

            models_url = self._get_openai_url("models")
            res = requests.get(models_url, headers=self._get_headers(), timeout=5)
            if res.status_code == 200:
                models = [m.get("id") for m in res.json().get("data", [])]
                return {
                    "ok": True,
                    "message": f"Connected to {self.config.provider} ({len(models)} models available)",
                    "models": models,
                }
            return {"ok": False, "message": f"Server returned HTTP {res.status_code}"}
        except Exception as err:
            return {
                "ok": False,
                "message": f"Could not connect to {self.config.base_url}: {str(err)}",
            }

    def generate_completion(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        available_tools: Optional[List[str]] = None,
    ) -> LLMResponse:
        if self.config.provider == "mock":
            return self._generate_mock_response(system_prompt, messages, available_tools)

        try:
            if self.config.provider == "ollama":
                return self._call_ollama_api(system_prompt, messages)
            else:
                return self._call_openai_compatible_api(system_prompt, messages)
        except Exception as err:
            mock_res = self._generate_mock_response(system_prompt, messages, available_tools)
            mock_res.content = f"[Notice: Local LLM fallback ({str(err)}). Showing simulated response]\n\n{mock_res.content}"
            return mock_res

    def _call_ollama_api(self, system_prompt: str, messages: List[Dict[str, Any]]) -> LLMResponse:
        formatted = [{"role": "system", "content": system_prompt}]
        for m in messages:
            formatted.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        base = self.config.base_url.rstrip("/")
        res = requests.post(
            f"{base}/api/chat",
            headers=self._get_headers(),
            json={
                "model": self.config.model_name,
                "messages": formatted,
                "stream": False,
                "options": {"temperature": self.config.temperature},
            },
            timeout=30,
        )
        res.raise_for_status()
        raw = res.json().get("message", {}).get("content", "")
        return self._parse_response(raw)

    def _call_openai_compatible_api(
        self, system_prompt: str, messages: List[Dict[str, Any]]
    ) -> LLMResponse:
        formatted = [{"role": "system", "content": system_prompt}]
        for m in messages:
            formatted.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        url = self._get_openai_url("chat/completions")
        res = requests.post(
            url,
            headers=self._get_headers(),
            json={
                "model": self.config.model_name,
                "messages": formatted,
                "temperature": self.config.temperature,
            },
            timeout=30,
        )
        res.raise_for_status()
        choices = res.json().get("choices", [])
        raw = choices[0].get("message", {}).get("content", "") if choices else ""
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> LLMResponse:
        thought = None
        content = raw

        think_match = re.search(r"<think>([\s\S]*?)</think>", raw)
        if think_match:
            thought = think_match.group(1).strip()
            content = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

        tool_calls = []
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "tool" in parsed or "name" in parsed:
                    tool_calls.append(
                        {
                            "id": f"call_{int(time.time() * 1000)}",
                            "name": parsed.get("tool") or parsed.get("name"),
                            "args": parsed.get("args") or parsed.get("parameters") or {},
                        }
                    )
            except Exception:
                pass

        return LLMResponse(
            content=content, thought=thought, tool_calls=tool_calls if tool_calls else None
        )

    def _generate_mock_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        available_tools: Optional[List[str]] = None,
    ) -> LLMResponse:
        time.sleep(0.3)
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        sys_lower = system_prompt.lower()
        if "supervisor" in sys_lower:
            if any(k in last_user.lower() for k in ["code", "script", "python", "react"]):
                return LLMResponse(
                    content="coder",
                    thought="User requests technical code. Routing control to Coder node.",
                )
            elif any(k in last_user.lower() for k in ["search", "find", "research"]):
                return LLMResponse(
                    content="researcher",
                    thought="User requests domain research. Routing control to Researcher node.",
                )
            return LLMResponse(content="FINISH", thought="Task complete. Finalizing execution.")

        if "researcher" in sys_lower:
            return LLMResponse(
                content=f"### Research Summary\nAnalyzed context for: '{last_user}'.\n- Local inference active.\n- LangGraph state channel verified.",
                thought="Gathered domain context for team state.",
            )

        if "coder" in sys_lower or "developer" in sys_lower:
            return LLMResponse(
                content=f"```python\n# Solution generated by Coder Node (Python LangGraph)\ndef execute_task():\n    return {{'status': 'success', 'input': '{last_user}'}}\n```",
                thought="Generated clean Python code module.",
            )

        if "critic" in sys_lower or "reviewer" in sys_lower:
            return LLMResponse(
                content="### Review\n- ✅ Type Safety & Correctness\n\nVerdict: APPROVED.",
                thought="Audited solution code.",
            )

        return LLMResponse(
            content=f"Processed response for task: {last_user[:50]}...",
            thought="Agent step complete.",
        )
