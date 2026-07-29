"""Anthropic LLM provider — implements `LLMProvider` port.

Single tool-use call. Forces the model to call our `submit_plan` tool, returning
the structured plan as the tool input. No tool rounds beyond this.

Prompt caching is engaged via `cache_control: ephemeral` on system blocks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import anthropic
import httpx

from app.core.config import get_settings
from app.features.plan_generation.application.ports import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)
from app.features.plan_generation.domain.errors import ToolCallMissingError
from app.features.plan_generation.settings import get_plan_v2_settings

logger = logging.getLogger(__name__)


# Per-million-token pricing (USD). Opus 4.5–4.8 = $5/$25 (verified 2026). Tuple:
# (input, output, cache_write=1.25x, cache_read=0.1x). Default = current Opus rate.
# INVARIANT: every model the platform can actually run (config.CLAUDE_MODEL /
# settings.PLAN_V2_MODEL and their shipped defaults) MUST appear here — a missing
# entry silently falls through to _DEFAULT_PRICING, so a genuinely-misconfigured
# model is billed indistinguishably from a priced one (no signal). The pricing-
# drift guard test (tests/.../test_cost_table_covers_running_model.py) fails CI if
# a configured model is not listed, and _estimate_cost log-warns at runtime.
_COST_TABLE: dict[str, tuple[float, float, float, float]] = {
    # model_id: (input, output, cache_write, cache_read)
    "claude-opus-4-8": (5.0, 25.0, 6.25, 0.5),
    "claude-opus-4-7": (5.0, 25.0, 6.25, 0.5),
    "claude-opus-4-6": (5.0, 25.0, 6.25, 0.5),
    "claude-opus-4-5": (5.0, 25.0, 6.25, 0.5),
    "claude-sonnet-4-6": (3.0, 15.0, 3.75, 0.3),
    "claude-sonnet-4-5": (3.0, 15.0, 3.75, 0.3),
    "claude-haiku-4-5-20251001": (0.8, 4.0, 1.0, 0.08),
    "claude-haiku-4-5": (0.8, 4.0, 1.0, 0.08),
}
_DEFAULT_PRICING = (5.0, 25.0, 6.25, 0.5)

# Models already warned-about, so an unpriced model logs once (not per call).
_UNPRICED_WARNED: set[str] = set()


def _thinking_effort_tier(request_effort: str | None, thinking_budget: int) -> str:
    """Resolve the adaptive-thinking effort tier for a request.

    A per-request ``request_effort`` (set by the use case from case complexity)
    wins; otherwise fall back to the budget-derived tier (≤4k=low, ≤16k=medium,
    >16k=high). Pure so the mapping is unit-tested without the Anthropic client.
    """
    if request_effort in ("low", "medium", "high"):
        return request_effort  # type: ignore[return-value]
    if thinking_budget <= 4096:
        return "low"
    if thinking_budget <= 16384:
        return "medium"
    return "high"


class AnthropicLLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        settings = get_settings()
        v2_settings = get_plan_v2_settings()
        # Admin-supplied key (from `llm_provider_configs.api_key_encrypted`)
        # wins; otherwise fall through to the env bootstrap.
        resolved_key = api_key or settings.ANTHROPIC_API_KEY
        # max_retries=0: retries are OWNED BY THE APPLICATION (the use case's retry
        # budget + the failure-cooldown guard), not hidden inside the SDK. A hidden
        # SDK retry burns its backoff sleep INSIDE the app-owned per-call deadline
        # (asyncio.wait_for below), eating the budget invisibly. This does not affect
        # the streaming-timeout path (verified: a mid-stream read timeout is not
        # retried by the SDK anyway) — it only removes the connection/5xx/429 retry
        # stacking, which the app handles deterministically.
        self._client = client or anthropic.AsyncAnthropic(
            api_key=resolved_key,
            max_retries=0,
            timeout=float(v2_settings.PLAN_V2_PROVIDER_TIMEOUT_SECONDS),
        )
        # PLAN_V2_MODEL stays as the env-bootstrap default; admin overrides
        # it per row via `llm_provider_configs.default_model`.
        self._model = model or v2_settings.PLAN_V2_MODEL
        self._thinking_enabled = v2_settings.PLAN_V2_THINKING_ENABLED
        self._thinking_budget = v2_settings.PLAN_V2_THINKING_BUDGET_TOKENS
        # Real per-CALL wall-clock deadline. The SDK client `timeout` above is a
        # per-READ socket timeout only: on a streaming request it resets on every
        # SSE event, and the Anthropic API emits keep-alive `ping` events far more
        # often than this value — so a long/stalled generation NEVER trips it and
        # the call is otherwise bounded only by the worker's 540s job watchdog,
        # which discards the whole job (JOB_TIMEOUT → doctor gets nothing). We
        # enforce the intended per-call budget ourselves with asyncio.wait_for so
        # PLAN_V2_PROVIDER_TIMEOUT_SECONDS actually bounds a call — which is the
        # premise the settings' `_validate_timeout_budget` (OUTCOME ≥ 2×PROVIDER)
        # invariant depends on.
        self._call_deadline_s = float(v2_settings.PLAN_V2_PROVIDER_TIMEOUT_SECONDS)

    async def preflight(self) -> dict[str, Any]:
        """One-token validation call against the CONFIGURED model + key.

        This exercises the exact auth + model-resolution path a real generation
        takes, so a phantom model (e.g. ``claude-sonnet-4-8``) or a bad/revoked
        key is caught at startup rather than mid-plan. Returns a structured
        result; the startup gate fails the boot on a *deterministic* error
        (``transient=False``) and tolerates a *transient* upstream one
        (overload / timeout / network → ``transient=True``).
        """
        started = time.monotonic()
        base: dict[str, Any] = {"provider": "anthropic", "model": self._model}
        v2 = get_plan_v2_settings()
        timeout_s = float(v2.PLAN_V2_PREFLIGHT_TIMEOUT_SECONDS)

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
            await self._client.with_options(timeout=timeout_s, max_retries=0).messages.create(
                model=self._model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return done(True, False, None, "ok")
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            # 401/403 — bad/revoked/insufficient-permission key. Deterministic.
            return done(False, False, "auth", str(e))
        except anthropic.NotFoundError as e:
            # 404 — the configured model does not exist. THE outage root cause.
            return done(False, False, "model_not_found", str(e))
        except (anthropic.BadRequestError, anthropic.UnprocessableEntityError) as e:
            # 400/422 — malformed model id or request. Deterministic config error.
            return done(False, False, "bad_request", str(e))
        except (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
        ) as e:
            # 429/5xx/timeout/network — transient; must NOT block boot.
            #
            # DO NOT add `anthropic.OverloadedError` here. Its existence is
            # VERSION-DEPENDENT: absent in anthropic 0.98.1, present in 0.117.0, and
            # the pin `anthropic>=0.87.0,<1.0.0` in requirements.txt spans both — so
            # both are legitimate resolves of the same requirement.
            #
            # On a resolve that lacks it, naming it in an except tuple raises
            # AttributeError when the clause is EVALUATED — i.e. the first time a
            # transient error reaches this handler — and that error escapes the whole
            # try statement, so the `except Exception` below does NOT catch it. The
            # handler whose job is "never block boot" then becomes the thing that
            # breaks on a transient upstream error.
            #
            # Classifying 529 by STATUS CODE below is correct on every version in the
            # range; depending on the typed class is correct only on some. See
            # MYREHAB_DEFECT_LEDGER D-02 and test_preflight_transient_and_delegation.py.
            return done(False, True, "transient", str(e))
        except anthropic.APIStatusError as e:
            # Numeric fallback for any status without a typed transient subclass —
            # notably 529 (overloaded) and 408/425. Mirrors _TRANSIENT_STATUS in
            # fallback_provider.py, which already classified 529 correctly.
            code = int(getattr(e, "status_code", 0) or 0)
            transient = code >= 500 or code in (408, 425, 429)
            return done(False, transient, f"status_{code}", str(e))
        except Exception as e:  # noqa: BLE001 — any unexpected error is treated as transient
            return done(False, True, "unknown", str(e))

    async def call_with_tool(self, req: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()

        # Anthropic constraint: extended-thinking is incompatible with ANY
        # forcing tool_choice (both `tool` and `any` count as forced). With
        # thinking on we use `tool_choice: auto` — but `auto` lets the model
        # answer in prose and skip the tool, which raises ToolCallMissingError.
        # The use case recovers by retrying with req.force_tool=True; on that
        # retry we MUST disable thinking so the forced tool_choice is accepted
        # and the `submit_plan` call is guaranteed.
        thinking_active = self._thinking_enabled and not req.force_tool
        if thinking_active:
            tool_choice_block: dict[str, Any] = {"type": "auto"}
        else:
            tool_choice_block = {"type": "tool", "name": req.tool_name}

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": req.max_output_tokens,
            "system": req.system,  # cache_control already set on each block
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": req.user_message}],
                }
            ],
            "tools": [
                {
                    "name": req.tool_name,
                    "description": "Submit the final structured rehabilitation plan.",
                    "input_schema": req.tool_schema,
                }
            ],
            "tool_choice": tool_choice_block,
        }
        # Extended thinking for clinical reasoning.
        # Opus 4.7+ uses `thinking.type=adaptive` + `output_config.effort`
        # (legacy `type=enabled` + `budget_tokens` was the older API).
        if thinking_active:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {
                "effort": _thinking_effort_tier(req.thinking_effort, self._thinking_budget)
            }

        # `temperature` is deprecated for Opus 4.7+; rely on tool-use determinism.
        # Streaming avoids the client-side request-timeout window for long runs —
        # but that same property means the SDK `timeout` does NOT bound the call,
        # so we impose the real per-call deadline here. On timeout asyncio.wait_for
        # cancels the coroutine (aborting the stream via the async context manager);
        # we surface it as a builtin TimeoutError, which the FallbackLLMProvider
        # classifies as retryable and the use case treats as a retryable attempt
        # (bounded, budget-guarded) rather than an instant hard-fail.
        async def _stream_final() -> Any:
            async with self._client.messages.stream(**kwargs) as stream:
                return await stream.get_final_message()

        # Per-request grace (auto-tool cutover) overrides the provider default,
        # but can never exceed it — the provider timeout is the hard ceiling.
        deadline_s = self._call_deadline_s
        if req.deadline_s is not None:
            deadline_s = min(req.deadline_s, self._call_deadline_s)
        try:
            message = await asyncio.wait_for(_stream_final(), timeout=deadline_s)
        except (
            asyncio.TimeoutError,
            TimeoutError,
            anthropic.APITimeoutError,
            httpx.TimeoutException,
        ) as exc:
            logger.warning(
                "anthropic_call_deadline_exceeded model=%s deadline_s=%.0f elapsed_s=%.1f "
                "force_tool=%s",
                self._model,
                deadline_s,
                time.monotonic() - t0,
                req.force_tool,
            )
            raise TimeoutError(
                f"Anthropic generation exceeded per-call deadline of {deadline_s:.0f}s"
            ) from exc

        latency_ms = int((time.monotonic() - t0) * 1000)
        tool_input = _extract_tool_input(message, req.tool_name)
        raw_text = _extract_raw_text(message)

        usage = message.usage
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0

        cost = _estimate_cost(self._model, in_tok, out_tok, cache_create, cache_read)

        logger.info(
            "anthropic_call model=%s in=%d out=%d cache_read=%d cache_write=%d "
            "latency_ms=%d cost_usd=%.4f stop=%s",
            self._model,
            in_tok,
            out_tok,
            cache_read,
            cache_create,
            latency_ms,
            cost,
            message.stop_reason,
        )

        return LLMResponse(
            tool_input=tool_input,
            raw_text=raw_text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
            latency_ms=latency_ms,
            model=self._model,
            estimated_cost_usd=cost,
        )


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


_WRAPPER_KEYS = (
    "plan",
    "plan_object",
    "rehab_plan",
    "rehabilitation_plan",
    "rehabilitationPlan",
    "submit_plan",
    "data",
)


def _extract_tool_input(message: Any, tool_name: str) -> dict[str, Any]:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == tool_name:
            input_value = getattr(block, "input", None)
            if isinstance(input_value, dict):
                # Some models wrap the plan in a `plan` / `rehabilitation_plan`
                # key despite the schema demanding top-level keys. Unwrap once
                # so downstream Pydantic validation sees the canonical shape.
                if (
                    len(input_value) == 1
                    and next(iter(input_value)) in _WRAPPER_KEYS
                    and isinstance(next(iter(input_value.values())), dict)
                ):
                    return next(iter(input_value.values()))
                return input_value
    raise ToolCallMissingError(
        f"Anthropic response did not contain expected tool_use block for {tool_name!r}"
    )


def _extract_raw_text(message: Any) -> str:
    parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(parts)


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int,
    cache_read: int,
) -> float:
    pricing = _COST_TABLE.get(model)
    if pricing is None:
        # Unpriced model: cost is estimated at the default Opus rate but flagged
        # so an unrecognized/misconfigured model can't spend silently. Warn once
        # per model id to keep the log signal (not per generation call).
        if model not in _UNPRICED_WARNED:
            _UNPRICED_WARNED.add(model)
            logger.warning(
                "cost_table_miss model=%s not in _COST_TABLE; billing at default "
                "Opus rate %s. Add it to _COST_TABLE if the rate differs.",
                model,
                _DEFAULT_PRICING,
            )
        pricing = _DEFAULT_PRICING
    in_rate, out_rate, cw_rate, cr_rate = pricing
    # Anthropic billing: input_tokens already excludes cache_read_input_tokens.
    return (
        (input_tokens / 1_000_000) * in_rate
        + (output_tokens / 1_000_000) * out_rate
        + (cache_creation / 1_000_000) * cw_rate
        + (cache_read / 1_000_000) * cr_rate
    )
