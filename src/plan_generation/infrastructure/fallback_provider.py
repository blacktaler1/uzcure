"""FallbackLLMProvider — primary then secondary on retryable failure.

Wraps two `LLMProvider` instances. Tries the primary first; if it raises a
retryable error (network, 5xx, rate-limit, timeout), falls through to the
secondary. Domain errors raised AFTER the LLM call (schema validation,
language compliance) are not in scope here — they live in the use case and
get retried inside Anthropic on the same provider, then surface upward.

The use case wraps every LLM exception in `LLMUpstreamError`, so this
provider is the only place where provider-specific retryability is checked.
"""

from __future__ import annotations

import logging
from typing import Iterable

from app.services.circuit_breaker import RedisCircuitBreaker
from app.features.plan_generation.application.ports import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)

logger = logging.getLogger(__name__)


# Anthropic + OpenAI exception class names that signal a transient failure
# worth failing over for. We match by class name to avoid hard imports of
# both SDKs in this module — keeps the file cheap to import in tests.
# Exception *class names* that signal a genuinely TRANSIENT failure worth retrying
# / failing over. These are matched against the full MRO of the raised exception
# (see _is_retryable), so each name must be a class that is ONLY ever transient.
#
# CRITICAL: do NOT add `APIStatusError` here. It is the common base class for ALL
# HTTP status errors in both the Anthropic and OpenAI SDKs — every 4xx (400 bad
# request, 401 auth, 403 permission, 404 not-found, 422 unprocessable) inherits
# from it. Listing the base would make every client error retryable, masking real
# defects and wasting a second LLM call on errors that can never succeed. Only the
# specific transient subclasses below belong here:
#   APIConnectionError  — network failure; request never reached the API
#                         (APITimeoutError is a subclass, covered transitively)
#   InternalServerError — 5xx server error
#   RateLimitError      — 429; server explicitly says "retry later"
#   OverloadedError     — 529; server temporarily overloaded
# (This matches the per-type non-transient classification in
# services/llm_error_classifier.py — 4xx are non-transient, 5xx/429/timeouts are.)
_RETRYABLE_NAMES: frozenset[str] = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "TimeoutError",
        "OverloadedError",
        # Raw httpx transport timeouts. On a STREAMING request the SDK does not
        # wrap a mid-stream read timeout into anthropic.APITimeoutError — it
        # surfaces the bare httpx exception (verified against the SDK). Without
        # these names a streaming timeout is misclassified as non-retryable, so a
        # configured failover never fires. All httpx.TimeoutException subclasses.
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }
)

# Transient HTTP status codes. Used by _is_retryable as a numeric fallback for a
# bare APIStatusError that carries a status_code but no typed transient subclass.
# 408 request-timeout, 425 too-early, 429 rate-limit, 5xx server, 529 overloaded.
# Deliberately EXCLUDES other 4xx (400/401/403/404/422) — those are permanent
# client errors and must not trigger a wasted failover. Mirrors the transient set
# in services/llm_error_classifier.py.
_TRANSIENT_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


class FallbackLLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        primary: LLMProvider,
        fallback: LLMProvider,
        primary_name: str = "primary",
        fallback_name: str = "fallback",
        primary_breaker: RedisCircuitBreaker | None = None,
        fallback_breaker: RedisCircuitBreaker | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_name = primary_name
        self._fallback_name = fallback_name
        # Circuit breaker for the primary leaf. The fallback's own breaker is
        # carried here only when the fallback is a LEAF (not another Fallback,
        # which owns its breaker internally). See get_status_map().
        self._primary_breaker = primary_breaker
        self._fallback_breaker = fallback_breaker

    async def call_with_tool(self, req: LLMRequest) -> LLMResponse:
        # Circuit-breaker gate: if the primary's circuit is open, skip it and go
        # straight to the fallback (record nothing — we did not call the primary).
        if self._primary_breaker is not None and self._primary_breaker.is_open:
            logger.warning(
                "llm_primary_circuit_open_skipping primary=%s fallback=%s",
                self._primary_name,
                self._fallback_name,
            )
            return await self._call_fallback(req, primary_exc=None)
        try:
            resp = await self._primary.call_with_tool(req)
            if self._primary_breaker is not None:
                self._primary_breaker.record_success()
            return resp
        except Exception as exc:  # noqa: BLE001 — we re-raise non-retryable
            if not _is_retryable(exc):
                raise
            if self._primary_breaker is not None:
                self._primary_breaker.record_failure()
            logger.warning(
                "llm_primary_failed_failing_over primary=%s fallback=%s exc=%s",
                self._primary_name,
                self._fallback_name,
                f"{type(exc).__name__}: {exc}"[:300],
            )
            return await self._call_fallback(req, primary_exc=exc)

    async def _call_fallback(
        self, req: LLMRequest, *, primary_exc: BaseException | None
    ) -> LLMResponse:
        """Call the fallback leg, recording its breaker outcome when the fallback
        is a leaf (a nested Fallback records its own internally)."""
        try:
            resp = await self._fallback.call_with_tool(req)
            if self._fallback_breaker is not None:
                self._fallback_breaker.record_success()
            return resp
        except Exception as fallback_exc:  # noqa: BLE001
            if self._fallback_breaker is not None and _is_retryable(fallback_exc):
                self._fallback_breaker.record_failure()
            logger.error(
                "llm_fallback_failed primary=%s fallback=%s primary_exc=%s fallback_exc=%s",
                self._primary_name,
                self._fallback_name,
                type(primary_exc).__name__ if primary_exc else "circuit_open",
                f"{type(fallback_exc).__name__}: {fallback_exc}"[:300],
            )
            raise

    async def preflight(self) -> dict:
        """Preflight the CHAIN, not just the primary leg.

        HISTORY, AND WHY THIS CHANGED TWICE.

        First this method did not exist, so `run_llm_preflight`'s
        ``getattr(provider, "preflight", None)`` was always None: the boot gate
        silently skipped while reporting ok=True — a check that never ran,
        indistinguishable from one that passed (D-03).

        It was then added delegating ONLY to the primary, reasoning that a
        degraded BACKUP must not fail the boot. That half was right. The mirror
        case was not, and it took production down on 2026-07-22:

            INFO   llm_fallback_chain_active order=['anthropic', 'openai']
            ERROR  [CRITICAL] LLM preflight FAILED  model=claude-opus-4-8
                   error=bad_request  'Your credit balance is too low...'
            RuntimeError: Deployment blocked

        A working anthropic→openai chain was built, with OpenAI keyed and
        healthy, and the app then refused to boot because the PRIMARY's billing
        balance was empty. Authentication, patient records and every clinical
        screen were unreachable for a reason the runtime is explicitly designed
        to survive. The container crash-looped every ~3 minutes.

        THE RULE NOW: the boot gate exists to catch a deploy that CANNOT SERVE.
        A chain with at least one healthy leg can serve, so it boots — loudly
        marked degraded. Only a chain where EVERY leg fails is the deterministic
        misconfiguration the gate was built for, and only that blocks boot.

        What this deliberately does NOT change: which model authors a clinical
        plan. Request-time failover keeps its existing rules (owner decisions
        2026-07-01, reversed 2026-07-07) — a non-transient primary error still
        surfaces rather than silently re-routing a clinical plan to a backup.
        This method governs BOOTING, not authorship.

        Walks nested chains: Fallback(A, Fallback(B, C)) tries A, then B, then C.
        """

        async def _leg(provider, name: str) -> dict:
            pf = getattr(provider, "preflight", None)
            if not callable(pf):
                return {
                    "ok": False,
                    "transient": True,
                    "provider": type(provider).__name__,
                    "model": getattr(provider, "_model", None),
                    "error_type": "preflight_unavailable",
                    "message": (
                        f"'{name}' exposes no preflight() — model/key NOT validated at boot"
                    ),
                    "latency_ms": 0,
                }
            try:
                return await pf()
            except Exception as e:  # noqa: BLE001 — a raising leg must not abort the sweep
                return {
                    "ok": False,
                    "transient": True,
                    "provider": type(provider).__name__,
                    "model": getattr(provider, "_model", None),
                    "error_type": "preflight_raised",
                    "message": f"'{name}' preflight raised: {e}",
                    "latency_ms": 0,
                }

        primary_pf = await _leg(self._primary, self._primary_name)
        if primary_pf.get("ok"):
            return primary_pf

        # Primary is down. Can anything else serve? A nested Fallback answers for
        # its whole subtree, so this covers chains of any depth.
        fallback_pf = await _leg(self._fallback, self._fallback_name)
        if fallback_pf.get("ok"):
            # Carry forward the legs the FALLBACK already reported dead (it may
            # itself be a nested chain that skipped past its own primary), then
            # prepend ours. Reading this from primary_pf instead lost the inner
            # chain's casualties, so a three-leg outage named only the outermost.
            degraded_from = list(fallback_pf.get("degraded_from") or [])
            degraded_from.insert(
                0,
                {
                    "provider": self._primary_name,
                    "error_type": primary_pf.get("error_type"),
                    "message": primary_pf.get("message"),
                },
            )
            out = dict(fallback_pf)
            out["ok"] = True
            out["degraded"] = True
            out["degraded_from"] = degraded_from
            out["message"] = (
                f"primary '{self._primary_name}' failed preflight "
                f"({primary_pf.get('error_type')}); serving via "
                f"'{self._fallback_name}'. Plan generation is DEGRADED."
            )
            return out

        # Every leg is unusable — this is the deploy the gate exists to stop.
        return {
            "ok": False,
            # Non-transient only when BOTH legs are non-transient: one transient
            # failure means a retry could still recover the chain, and a boot
            # block is the wrong response to that.
            "transient": bool(primary_pf.get("transient") or fallback_pf.get("transient")),
            "provider": type(self._primary).__name__,
            "model": primary_pf.get("model"),
            "error_type": primary_pf.get("error_type") or "chain_unavailable",
            "message": (
                f"no usable provider: '{self._primary_name}' "
                f"({primary_pf.get('error_type')}: {primary_pf.get('message')}); "
                f"'{self._fallback_name}' "
                f"({fallback_pf.get('error_type')}: {fallback_pf.get('message')})"
            ),
            "latency_ms": 0,
        }

    def get_status_map(self) -> dict:
        """Per-provider circuit-breaker status for the whole nested chain.
        Walks the structure: this node reports its primary's breaker, then merges
        the fallback's map (nested Fallback) or the fallback leaf's breaker."""
        out: dict = {}
        if self._primary_breaker is not None:
            out[self._primary_name] = self._primary_breaker.get_status()
        fb_map = getattr(self._fallback, "get_status_map", None)
        if callable(fb_map):
            out.update(self._fallback.get_status_map())
        elif self._fallback_breaker is not None:
            out[self._fallback_name] = self._fallback_breaker.get_status()
        return out


def _is_retryable(exc: BaseException) -> bool:
    for cls in _walk_classes(type(exc)):
        if cls.__name__ in _RETRYABLE_NAMES:
            return True
    # Numeric HTTP-status fallback. The class-name match above misses a transient
    # error that surfaces as a *bare* APIStatusError carrying only a status_code
    # (e.g. a 529 overload that the SDK did not map to the typed OverloadedError —
    # seen on some proxy/streaming paths). Both the Anthropic and OpenAI SDKs put
    # the HTTP status on `.status_code`. Classify by the transient-status set so a
    # real 529/503/500/429/etc. fails over even without a typed subclass — while
    # 4xx client errors (400/401/403/404/422) stay NON-retryable. Mirrors the
    # transient-status set in services/llm_error_classifier.py.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    # asyncio.TimeoutError before Python 3.11 isn't a TimeoutError subclass
    return isinstance(exc, TimeoutError)


def _walk_classes(cls: type) -> Iterable[type]:
    seen: set[type] = set()
    stack: list[type] = [cls]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        yield c
        stack.extend(c.__bases__)
