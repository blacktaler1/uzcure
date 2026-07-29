"""Factory: build the active `LLMProvider` for plan generation.

Reads admin-managed `llm_provider_configs` rows (is_enabled=True, ordered
by `priority` ascending) and returns a failover CHAIN: the highest-priority
provider is primary, each lower-priority provider is its backup.

FAILOVER POLICY (owner decision, 2026-07-01 REVERSED 2026-07-07): clinical
plans are authored by the primary; on a genuinely TRANSIENT primary failure
(network / 5xx / 429 rate-limit / 529 overload / timeout) the next provider
takes over so the doctor still gets a plan instead of a hard failure. Only
transient failures fail over — non-transient errors (4xx, schema/language
compliance) still surface, so no silent downgrade on a real defect. The
downstream verdict / requires_doctor_approval gating applies identically
regardless of which provider authored the plan. Failover is off whenever only
one provider is enabled (a single row, or the empty-table bootstrap).

Bootstrap: if the table is empty (fresh install before the admin seeds it),
uses the env-configured primary (Anthropic). Cache TTL = 60s so admin edits
take effect within a minute without per-request DB pressure.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.features.plan_generation.application.ports import LLMProvider
from app.features.plan_generation.infrastructure.anthropic_provider import (
    AnthropicLLMProvider,
)
from app.features.plan_generation.infrastructure.fallback_provider import (
    FallbackLLMProvider,
)
from app.features.plan_generation.infrastructure.deepseek_provider import (
    DeepSeekLLMProvider,
)
from app.features.plan_generation.infrastructure.gemini_provider import (
    GeminiLLMProvider,
)
from app.features.plan_generation.infrastructure.openai_provider import (
    OpenAILLMProvider,
)
from app.models.llm_provider_config import LlmProviderConfig
from app.services.circuit_breaker import RedisCircuitBreaker, get_redis_client_for_cb

logger = logging.getLogger(__name__)


_CACHE_TTL_SECONDS = 60.0
_cache: dict[str, tuple[float, LLMProvider]] = {}

# Per-provider circuit breakers, keyed by provider name. Module-level so the
# same breaker instance (and its in-memory trip state) survives chain rebuilds
# every cache cycle. Redis-backed when available; in-memory fallback otherwise.
_breakers: dict[str, RedisCircuitBreaker] = {}


def _breaker_for(name: str) -> RedisCircuitBreaker:
    cb = _breakers.get(name)
    if cb is None:
        cb = RedisCircuitBreaker(name, redis_client=get_redis_client_for_cb())
        _breakers[name] = cb
    return cb


# Provider name → constructor adapter. Each adapter takes (api_key, model)
# and returns an `LLMProvider` instance. Adding a new provider is a one-line
# registration here.
_PROVIDER_BUILDERS: dict[str, Callable[[str, str], LLMProvider]] = {
    "anthropic": lambda key, model: AnthropicLLMProvider(api_key=key or None, model=model),
    "openai": lambda key, model: OpenAILLMProvider(api_key=key or None, model=model),
    "gemini": lambda key, model: GeminiLLMProvider(api_key=key, model=model),
    # DeepSeek is OpenAI-API-compatible; `key or None` falls back to env like the
    # other two that support a bootstrap path.
    "deepseek": lambda key, model: DeepSeekLLMProvider(api_key=key or None, model=model),
}


def build_llm_provider() -> LLMProvider:
    """Return the active provider chain.

    Cached for `_CACHE_TTL_SECONDS` so a stable workflow (plan-gen burst)
    doesn't hammer the DB. Cache is keyed on a single string so any update
    to the table invalidates the whole entry on the next read.
    """
    cached = _cache.get("chain")
    if cached and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    db = SessionLocal()
    try:
        provider = _build_from_db(db)
    finally:
        db.close()

    _cache["chain"] = (time.monotonic(), provider)
    return provider


def invalidate_cache() -> None:
    """Drop the cached chain. Call after admin-side config writes so the
    next plan-generation request rebuilds with the new settings."""
    _cache.clear()


def _build_from_db(db: Session) -> LLMProvider:
    rows = (
        db.query(LlmProviderConfig)
        .filter(LlmProviderConfig.is_enabled.is_(True))
        .order_by(LlmProviderConfig.priority.asc(), LlmProviderConfig.name.asc())
        .all()
    )

    instances: list[tuple[str, LLMProvider]] = []
    for row in rows:
        builder = _PROVIDER_BUILDERS.get(row.name)
        if builder is None:
            logger.warning("llm_provider_unknown_skipped name=%s", row.name)
            continue
        api_key = row.api_key_encrypted or ""
        try:
            instances.append((row.name, builder(api_key, row.default_model)))
        except Exception as e:  # noqa: BLE001 — bad row shouldn't kill chain
            logger.warning(
                "llm_provider_init_failed name=%s model=%s reason=%s",
                row.name,
                row.default_model,
                e,
            )

    if not instances:
        logger.info("llm_provider_db_empty falling_back_to_env_bootstrap")
        return _bootstrap_from_env()

    if len(instances) == 1:
        return instances[0][1]

    # FAILOVER (owner decision 2026-07-01 reversed 2026-07-07): build a nested
    # FallbackLLMProvider chain, primary first. On a TRANSIENT primary failure
    # the next provider authors the plan; non-transient errors still surface.
    chain = _chain_with_fallback(instances)
    logger.info("llm_fallback_chain_active order=%s", [n for n, _ in instances])
    return chain


def _chain_with_fallback(
    instances: list[tuple[str, LLMProvider]],
) -> LLMProvider:
    """Fold ordered (name, provider) pairs into a nested FallbackLLMProvider.

    instances[0] is the primary; each subsequent entry is the next backup.
    Built right-to-left so ``Fallback(A, Fallback(B, C))`` tries A→B→C. A
    per-provider circuit breaker is attached to each leg; a `fallback_breaker`
    is passed only when the fallback is a LEAF (a nested Fallback owns its own
    breakers via get_status_map — matching FallbackLLMProvider's contract).
    """
    node_name, node = instances[-1]
    node_is_leaf = True
    for name, provider in reversed(instances[:-1]):
        node = FallbackLLMProvider(
            primary=provider,
            fallback=node,
            primary_name=name,
            fallback_name=node_name,
            primary_breaker=_breaker_for(name),
            fallback_breaker=_breaker_for(node_name) if node_is_leaf else None,
        )
        node_is_leaf = False
        node_name = name
    return node


def _bootstrap_from_env() -> LLMProvider:
    """Fresh-install wiring before the admin seeds `llm_provider_configs`.

    Primary-only by clinical-safety policy (owner decision, 2026-07-01): a
    clinical plan is authored by the primary (Anthropic) model only — never a
    backup. If it is unavailable the job fails closed and the doctor retries.
    """
    return AnthropicLLMProvider()


def get_plan_chain_status() -> dict:
    """Circuit-breaker status for the LIVE plan-generation provider chain,
    in the same shape as llm_router.stats['circuit_breakers'] so health/monitoring
    endpoints can report the chain plan generation actually uses. Returns {} when
    the chain is a single provider (no Fallback, no breakers) or not yet built."""
    cached = _cache.get("chain")
    chain = cached[1] if cached else None
    get_map = getattr(chain, "get_status_map", None)
    if callable(get_map):
        return chain.get_status_map()
    return {}


async def run_llm_preflight(timeout_s: float) -> dict:
    """Validate the ACTIVE plan-generation provider (env- or DB-resolved) with a
    tiny real call. Returns a structured result:
        {ok, transient, provider, model, error_type, message, latency_ms}
    Never raises — the caller decides whether to fail the boot based on `ok`
    and `transient`. A provider without a `preflight()` method is reported as
    ok=False / transient=True (`preflight_unavailable`): visible as a real gap,
    but never boot-blocking, since "could not validate" is weaker evidence than
    "validated and wrong".
    """
    import asyncio

    try:
        provider = build_llm_provider()
    except Exception as e:  # noqa: BLE001 — provider construction itself failed
        # TRANSIENT: failing to even BUILD the provider (e.g. a DB blip reading
        # llm_provider_configs) is an infrastructure error, NOT a model/key
        # misconfiguration — it must never block boot on preflight grounds.
        return {
            "ok": False,
            "transient": True,
            "provider": "unknown",
            "model": None,
            "error_type": "provider_init",
            "message": str(e)[:300],
            "latency_ms": 0,
        }

    pf = getattr(provider, "preflight", None)
    if not callable(pf):
        # FAIL-VISIBLE, NOT FAIL-OPEN. This previously returned ok=True, so a check
        # that never ran was indistinguishable from a check that passed — and that
        # is exactly what production did (verified live 2026-07-21). ok=False makes
        # the gap visible in /health and app.state.llm_preflight; transient=True
        # keeps it non-boot-blocking, because "we could not validate" is not the
        # same evidence as "the model/key are wrong". See DEFECT_LEDGER D-03.
        return {
            "ok": False,
            "transient": True,
            "provider": type(provider).__name__,
            "model": getattr(provider, "_model", None),
            "error_type": "preflight_unavailable",
            "message": (
                f"{type(provider).__name__} exposes no preflight() — "
                "model/key NOT validated at boot"
            ),
            "latency_ms": 0,
        }

    try:
        # Hard wall-clock guard so a hung upstream can never block boot forever.
        return await asyncio.wait_for(pf(), timeout=timeout_s + 3.0)
    except (asyncio.TimeoutError, TimeoutError):
        return {
            "ok": False,
            "transient": True,
            "provider": type(provider).__name__,
            "model": getattr(provider, "_model", None),
            "error_type": "timeout",
            "message": f"preflight exceeded {timeout_s + 3.0:.0f}s",
            "latency_ms": int((timeout_s + 3.0) * 1000),
        }
