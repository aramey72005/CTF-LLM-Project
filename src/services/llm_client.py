from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


class LLMClient:
    """
    Lightweight client for talking to an Ollama-compatible HTTP endpoint.

    Supports two calling modes:
      - generate()  single-turn via /api/generate  (used by state-aware LLM mode)
      - chat()      multi-turn via /api/chat        (used by conversational mode)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "phi3",
        timeout: int = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Single-turn — used by state-aware LLM mode (unchanged)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> str:
        """
        Send a single prompt to /api/generate and return the text response.
        Used exclusively by the state-aware LLM planner (mode='llm').
        Do not change this method — the state-aware path depends on it.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "format": "json",
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        url = f"{self.base_url}/api/generate"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Failed to call Ollama generate endpoint.")
            raise RuntimeError(f"Failed to call Ollama generate endpoint: {exc}") from exc

        if stream:
            return self._collect_streaming_response(response)

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            logger.exception("Ollama response was not valid JSON.")
            raise RuntimeError("Ollama response was not valid JSON.") from exc

        if "response" not in data:
            raise RuntimeError(f"Ollama response missing 'response' field: {data}")

        return str(data["response"]).strip()

    # ------------------------------------------------------------------
    # Multi-turn — used by conversational mode (new)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        stream: bool = False,
    ) -> str:
        """
        Send a multi-turn conversation to /api/chat and return the assistant reply.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: Sampling temperature
            stream: Whether to use streaming mode

        Returns:
            The assistant's reply text

        Used by conversational mode (mode='llm_nostate') to simulate true
        multi-turn dialogue where the model reasons step-by-step and each
        turn builds on the previous exchange.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "format": "json",
            "options": {"temperature": temperature},
        }

        url = f"{self.base_url}/api/chat"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Failed to call Ollama chat endpoint.")
            raise RuntimeError(f"Failed to call Ollama chat endpoint: {exc}") from exc

        if stream:
            return self._collect_streaming_chat_response(response)

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            logger.exception("Ollama chat response was not valid JSON.")
            raise RuntimeError("Ollama chat response was not valid JSON.") from exc

        # /api/chat returns {"message": {"role": "assistant", "content": "..."}}
        message = data.get("message", {})
        content = message.get("content", "")
        if not content:
            raise RuntimeError(f"Ollama chat response missing content: {data}")

        return str(content).strip()

    # ------------------------------------------------------------------
    # Health / model listing
    # ------------------------------------------------------------------

    def list_models(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.exception("Failed to fetch model list from Ollama.")
            raise RuntimeError(f"Failed to fetch model list from Ollama: {exc}") from exc

    def health_check(self) -> bool:
        try:
            self.list_models()
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_streaming_response(self, response: requests.Response) -> str:
        """Collect /api/generate streaming chunks."""
        chunks = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid streaming JSON chunk: %s", line)
                continue
            if "response" in data:
                chunks.append(data["response"])
        return "".join(chunks).strip()

    def _collect_streaming_chat_response(self, response: requests.Response) -> str:
        """Collect /api/chat streaming chunks."""
        chunks = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid streaming chat chunk: %s", line)
                continue
            content = data.get("message", {}).get("content", "")
            if content:
                chunks.append(content)
        return "".join(chunks).strip()
