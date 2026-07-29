"""OpenAI LLM provider — implements `LLMProvider` port.

Single forced function-call. The `submit_plan` tool returns the structured plan
as `tool_calls[0].function.arguments` (JSON string). No tool-use rounds.

Used as failover when Anthropic Claude is unavailable, returns 5xx, or hits
rate limits (see `FallbackLLMProvider`). Quality is lower than Claude on
multi-morbid clinical reasoning; downstream verdict classifier still applies
the same `requires_doctor_approval` gating.

Anthropic-specific request fields (system cache_control blocks) are flattened
to a single system string. OpenAI does not support cache_control at the API
level for chat-completions.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.features.plan_generation.application.ports import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)

logger = logging.getLogger(__name__)


_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_per_million_usd, output_per_million_usd)
    "gpt-5.4": (5.0, 20.0),
    "gpt-5": (5.0, 20.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_OPENAI_PRICING = (5.0, 20.0)


_WRAPPER_KEYS = ("plan", "rehabilitation_plan", "rehabilitationPlan", "data")


# Preflight budget. Must exceed what a REASONING model burns before it can emit
# anything: gpt-5.5 rejects 1 with a 400 ("output limit was reached") and accepts
# 8. 16 leaves headroom without paying for a real answer — preflight validates
# auth and model resolution, not output.
_PREFLIGHT_MAX_TOKENS = 16


def _is_output_limit_error(text: str) -> bool:
    """True when a 400 means "ran out of tokens", not "bad configuration".

    Reaching the output limit proves the request authenticated and the model
    resolved, so it is a preflight PASS. Treating it as a config error marked a
    healthy provider misconfigured in production on 2026-07-22.
    """
    low = text.lower()
    return "output limit" in low or ("max_tokens" in low and "reached" in low)


class OpenAILLMProvider(LLMProvider):
    """Plan-generation LLM provider over OpenAI chat-completions.

    Implements the same `LLMProvider` Protocol the Anthropic provider does, so
    the use case is provider-agnostic.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        # Admin-supplied key (from `llm_provider_configs.api_key_encrypted`)
        # wins; fall back to env for the bootstrap path.
        resolved_key = api_key or settings.OPENAI_API_KEY
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        # max_retries=0: retries are owned by the application (use-case budget +
        # cooldown guard), not stacked invisibly inside the SDK. Consistent with the
        # Anthropic provider.
        self._client = client or AsyncOpenAI(
            api_key=resolved_key,
            max_retries=0,
        )
        self._model = model or settings.OPENAI_GENERATION_MODEL
        self._timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "LLM_TIMEOUT_OPENAI", 90)
        )

    async def preflight(self) -> dict[str, Any]:
        """One-token validation call against the CONFIGURED model + key.

        WHY THIS EXISTS. Without it, `run_llm_preflight`'s
        ``getattr(provider, "preflight", None)`` was None and the boot gate
        skipped this provider entirely — the same hole D-03 closed for the
        fallback chain. It surfaced in production on 2026-07-22: after OpenAI was
        promoted to primary, /health reported

            "primary 'openai' exposes no preflight() — model/key NOT validated at boot"

        so the platform booted without anyone having checked that the OpenAI
        model or key worked at all. A check that never runs is indistinguishable
        from a check that passes, which is precisely what makes it dangerous.

        Mirrors the Anthropic contract: deterministic errors (bad key, phantom
        model, malformed request) are ``transient=False`` and block boot; upstream
        wobble (rate limit, 5xx, timeout, connection) is ``transient=True`` and
        does not.

        `max_completion_tokens` — not `max_tokens` — because the configured model
        may be a reasoning model (gpt-5.x), which rejects the legacy field.

        THE BUDGET IS LOAD-BEARING. A first version of this method sent 1, and
        production immediately marked a healthy provider as misconfigured:

            preflight: primary 'openai' failed preflight (bad_request)

        Measured against the live account on gpt-5.5:

            max_completion_tokens=1  -> HTTP 400 "Could not finish the message
                                        because max_tokens or model output limit
                                        was reached"
            max_completion_tokens=8  -> HTTP 200

        A reasoning model spends tokens thinking before it emits anything, so a
        budget of 1 cannot complete and the API rejects it. That 400 was then
        classified `bad_request` -> non-transient -> BOOT BLOCKED. Had OpenAI been
        the only configured provider, this check would have taken the platform
        down over a provider that works perfectly.

        Two independent guards now:
          1. the budget is large enough for a reasoning model to finish, and
          2. an output-limit 400 is treated as SUCCESS, because reaching the token
             limit proves the request authenticated and the model resolved — which
             is the entire question preflight asks. It does not assert output.
        """
        started = time.monotonic()
        base: dict[str, Any] = {"provider": "openai", "model": self._model}

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
            import openai as _openai
        except Exception as e:  # noqa: BLE001 — SDK import problems are not config errors
            return done(False, True, "sdk_unavailable", str(e))

        try:
            from app.features.plan_generation.settings import get_plan_v2_settings

            timeout_s = float(get_plan_v2_settings().PLAN_V2_PREFLIGHT_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001 — never let a settings problem block boot
            timeout_s = 20.0

        try:
            await self._client.with_options(
                timeout=timeout_s, max_retries=0
            ).chat.completions.create(
                model=self._model,
                max_completion_tokens=_PREFLIGHT_MAX_TOKENS,
                messages=[{"role": "user", "content": "ping"}],
            )
            return done(True, False, None, "ok")
        except (_openai.AuthenticationError, _openai.PermissionDeniedError) as e:
            # 401/403 — bad, revoked, or insufficient-permission key.
            return done(False, False, "auth", str(e))
        except _openai.NotFoundError as e:
            # 404 — the configured model does not exist on this account.
            return done(False, False, "model_not_found", str(e))
        except (_openai.BadRequestError, _openai.UnprocessableEntityError) as e:
            text = str(e)
            if _is_output_limit_error(text):
                # The request authenticated and the model resolved; it simply ran
                # out of room to answer. That is exactly what preflight asks, so
                # this is a PASS, not a config error. See the budget note above.
                return done(True, False, None, "ok (output limit reached)")
            # 400/422 — malformed model id or request. Deterministic.
            return done(False, False, "bad_request", text)
        except (
            _openai.RateLimitError,
            _openai.InternalServerError,
            _openai.APITimeoutError,
            _openai.APIConnectionError,
        ) as e:
            return done(False, True, "transient_upstream", str(e))
        except Exception as e:  # noqa: BLE001 — unknown failure must not block a boot
            return done(False, True, "unknown", f"{type(e).__name__}: {e}")

    async def call_with_tool(self, req: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()

        system_text = _flatten_system_blocks(req.system)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": req.tool_name,
                    "description": "Submit the final structured rehabilitation plan.",
                    "parameters": req.tool_schema,
                },
            }
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": req.user_message},
            ],
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": req.tool_name},
            },
            "max_completion_tokens": req.max_output_tokens,
            "timeout": self._timeout,
        }

        response = await self._client.chat.completions.create(**kwargs)

        latency_ms = int((time.monotonic() - t0) * 1000)

        choice = response.choices[0]
        message = choice.message
        tool_input = _extract_tool_input(message, req.tool_name)
        raw_text = (message.content or "") if hasattr(message, "content") else ""

        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0

        cost = _estimate_cost(self._model, in_tok, out_tok)

        logger.info(
            "openai_call model=%s in=%d out=%d latency_ms=%d cost_usd=%.4f finish=%s",
            self._model,
            in_tok,
            out_tok,
            latency_ms,
            cost,
            choice.finish_reason,
        )

        # Anthropic-style cache token fields don't apply to OpenAI; report 0
        # so the LLMResponse contract is satisfied without misleading numbers.
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


def _extract_tool_input(message: Any, tool_name: str) -> dict[str, Any]:
    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        fn = getattr(call, "function", None)
        if fn is None or getattr(fn, "name", "") != tool_name:
            continue
        raw_args = getattr(fn, "arguments", "") or ""
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"OpenAI tool_call arguments not valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI tool_call arguments must decode to a dict")
        if (
            len(parsed) == 1
            and next(iter(parsed)) in _WRAPPER_KEYS
            and isinstance(next(iter(parsed.values())), dict)
        ):
            return next(iter(parsed.values()))
        return parsed
    raise RuntimeError(f"OpenAI response did not contain expected tool_call for {tool_name!r}")


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _OPENAI_PRICING.get(model, _DEFAULT_OPENAI_PRICING)
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
