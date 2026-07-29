"""Gemini LLM provider — implements `LLMProvider` port.

Single forced function-call. The `submit_plan` tool returns the structured
plan as the function-call args dict.

Anthropic-style cache_control system blocks are flattened to a single
`system_instruction` string. Gemini does not expose prompt caching at the
chat-completions level for this SDK version.

Used in the plan-generation failover chain alongside Anthropic and OpenAI.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.features.plan_generation.application.ports import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)

logger = logging.getLogger(__name__)


# Per-million-token pricing (USD). Gemini publishes input + output rates.
# Long-context tier (>200k tokens) is more expensive but rare in plan-gen
# (we cap at 50 exercises); the standard rate is what gets billed almost always.
_GEMINI_PRICING: dict[str, tuple[float, float]] = {
    # CURRENT (alias) ids. Every pinned 2.x id below was retired by Google on
    # 2026-07-22 - all returning 404 "no longer available" on the same day - so
    # the aliases are what production actually runs, and they must be priced or
    # every call falls through to _DEFAULT_GEMINI_PRICING and cost telemetry
    # quietly becomes fiction.
    "gemini-pro-latest": (1.25, 10.0),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-3-flash-preview": (0.30, 2.50),
    # RETIRED, kept deliberately: historical rows in the plan ledger still carry
    # these model ids, and re-costing an old plan must not silently use today's
    # rates for a model it never ran on.
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-exp": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
}
_DEFAULT_GEMINI_PRICING = (0.30, 2.50)


_WRAPPER_KEYS = ("plan", "rehabilitation_plan", "rehabilitationPlan", "data")


class GeminiLLMProvider(LLMProvider):
    """Plan-generation LLM provider over Google Gemini.

    Implements the same `LLMProvider` Protocol the Anthropic and OpenAI
    providers do, so the use case stays provider-agnostic.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int | None = None,
        client: genai.Client | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("Gemini api_key is required")
        if not model:
            raise RuntimeError("Gemini model id is required")
        self._client = client or genai.Client(api_key=api_key)
        self._model = model
        self._timeout = float(timeout_seconds if timeout_seconds is not None else 90)

    async def preflight(self) -> dict[str, Any]:
        """One-token validation call against the CONFIGURED model + key.

        Same contract and same reason as the OpenAI and Anthropic providers: a
        provider without `preflight()` is invisible to the boot gate, and a check
        that never runs is indistinguishable from one that passes. See the note
        in openai_provider.preflight — that hole was found live on 2026-07-22.

        Uses the async `client.aio` surface and `asyncio.wait_for`, exactly as
        `call_with_tool` does — the SDK offers no per-request timeout option, and
        preflight must exercise the same code path a real generation takes.

        Google's SDK does not expose a clean per-status exception hierarchy, so
        classification reads the error text. Deliberately conservative: only
        clearly deterministic signals (auth, missing model, invalid argument) are
        treated as boot-blocking. Anything unrecognised is transient — a boot
        block is a serious action and should never rest on a guessed string match.
        """
        started = time.monotonic()
        base: dict[str, Any] = {"provider": "gemini", "model": self._model}

        def done(ok: bool, transient: bool, error_type: str | None, message: str) -> dict[str, Any]:
            return {
                **base,
                "ok": ok,
                "transient": transient,
                "error_type": error_type,
                "message": message[:300] if message else "",
                "latency_ms": int((time.monotonic() - started) * 1000),
            }

        try:
            from app.features.plan_generation.settings import get_plan_v2_settings

            timeout_s = float(get_plan_v2_settings().PLAN_V2_PREFLIGHT_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001 — never let a settings problem block boot
            timeout_s = 20.0

        try:
            await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._model,
                    contents="ping",
                    config=genai_types.GenerateContentConfig(max_output_tokens=1),
                ),
                timeout=timeout_s,
            )
            return done(True, False, None, "ok")
        except asyncio.TimeoutError:
            return done(
                False, True, "transient_upstream", f"preflight timed out after {timeout_s}s"
            )
        except Exception as e:  # noqa: BLE001 — classified below, never re-raised
            text = f"{type(e).__name__}: {e}"
            low = text.lower()
            if any(
                s in low
                for s in (
                    "api key not valid",
                    "api_key_invalid",
                    "unauthenticated",
                    "permission_denied",
                    "permission denied",
                )
            ):
                return done(False, False, "auth", text)
            if any(
                s in low for s in ("not found", "notfound", "is not supported", "unknown model")
            ):
                return done(False, False, "model_not_found", text)
            if "invalid_argument" in low or "invalidargument" in low:
                return done(False, False, "bad_request", text)
            # Rate limits, 5xx, connection resets and anything unclassified.
            return done(False, True, "transient_upstream", text)

    async def call_with_tool(self, req: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()

        system_text = _flatten_system_blocks(req.system)

        function_declaration = genai_types.FunctionDeclaration(
            name=req.tool_name,
            description="Submit the final structured rehabilitation plan.",
            parameters=_strip_unsupported_schema(req.tool_schema),
        )
        tool = genai_types.Tool(function_declarations=[function_declaration])

        config = genai_types.GenerateContentConfig(
            system_instruction=system_text or None,
            tools=[tool],
            tool_config=genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[req.tool_name],
                )
            ),
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
        )

        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._model,
                    contents=req.user_message,
                    config=config,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"Gemini call timed out after {self._timeout:.0f}s") from e

        latency_ms = int((time.monotonic() - t0) * 1000)

        tool_input = _extract_tool_input(response, req.tool_name)
        raw_text = _safe_text(response)

        usage = getattr(response, "usage_metadata", None)
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

        cost = _estimate_cost(self._model, in_tok, out_tok)

        logger.info(
            "gemini_call model=%s in=%d out=%d latency_ms=%d cost_usd=%.4f",
            self._model,
            in_tok,
            out_tok,
            latency_ms,
            cost,
        )

        return LLMResponse(
            tool_input=tool_input,
            raw_text=raw_text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=latency_ms,
            model=self._model,
            estimated_cost_usd=cost,
        )


def _flatten_system_blocks(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n\n".join(parts)


def _strip_unsupported_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON-schema fields Gemini's function declaration doesn't accept.

    Gemini rejects keys like `additionalProperties`, `$schema`, `definitions`,
    and `default` on most properties; quietly drop them while keeping the
    structure (type, properties, required, items, enum, description).
    """
    if not isinstance(schema, dict):
        return schema
    drop = {"$schema", "additionalProperties", "definitions", "$defs", "default"}
    cleaned: dict[str, Any] = {}
    for k, v in schema.items():
        if k in drop:
            continue
        if isinstance(v, dict):
            cleaned[k] = _strip_unsupported_schema(v)
        elif isinstance(v, list):
            cleaned[k] = [
                _strip_unsupported_schema(item) if isinstance(item, dict) else item for item in v
            ]
        else:
            cleaned[k] = v
    return cleaned


def _extract_tool_input(response: Any, tool_name: str) -> dict[str, Any]:
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fn = getattr(part, "function_call", None)
            if fn is None or getattr(fn, "name", "") != tool_name:
                continue
            args = getattr(fn, "args", None)
            if args is None:
                continue
            parsed = dict(args)
            if (
                len(parsed) == 1
                and next(iter(parsed)) in _WRAPPER_KEYS
                and isinstance(next(iter(parsed.values())), dict)
            ):
                return next(iter(parsed.values()))
            return parsed
    raise RuntimeError(f"Gemini response did not contain expected function_call for {tool_name!r}")


def _safe_text(response: Any) -> str:
    try:
        return response.text or ""
    except Exception:
        return ""


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _GEMINI_PRICING.get(model, _DEFAULT_GEMINI_PRICING)
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
