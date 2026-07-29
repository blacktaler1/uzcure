"""Redis-backed cooldown guard with in-memory fallback.

When a generation fails (timeout/error), the idem_key is recorded with a TTL.
A repeat request within the cooldown window is rejected at the use-case
boundary to prevent retry storms.
"""

from __future__ import annotations

import logging
import time

from app.features.plan_generation.application.ports import CooldownGuard

logger = logging.getLogger(__name__)


_REDIS_PREFIX = "myrehab:plan_v2_failure:"
DEFAULT_COOLDOWN_SECONDS = 90
_FALLBACK_CLEANUP_THRESHOLD = 200


class FailureCooldownGuard(CooldownGuard):
    def __init__(self, *, cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS) -> None:
        self._cooldown = cooldown_seconds
        self._fallback: dict[str, float] = {}

    async def is_cooling_down(self, *, key: str) -> bool:
        client = _redis_client()
        if client is not None:
            try:
                return bool(client.exists(_REDIS_PREFIX + key))
            except Exception:
                logger.debug("cooldown_redis_check_failed key=%s", key[:12])
        return self._fallback_active(key)

    async def remaining_seconds(self, *, key: str) -> int:
        client = _redis_client()
        if client is not None:
            try:
                ttl = client.ttl(_REDIS_PREFIX + key)
                return int(ttl) if ttl and ttl > 0 else 0
            except Exception:
                logger.debug("cooldown_redis_ttl_failed key=%s", key[:12])
        ts = self._fallback.get(key)
        if ts is None:
            return 0
        return max(0, int(self._cooldown - (time.monotonic() - ts)))

    async def record_failure(self, *, key: str) -> None:
        client = _redis_client()
        if client is not None:
            try:
                client.setex(_REDIS_PREFIX + key, self._cooldown, "1")
                return
            except Exception:
                logger.debug("cooldown_redis_set_failed key=%s", key[:12])
        self._fallback[key] = time.monotonic()
        if len(self._fallback) > _FALLBACK_CLEANUP_THRESHOLD:
            self._sweep_fallback()

    def _fallback_active(self, key: str) -> bool:
        ts = self._fallback.get(key)
        if ts is None:
            return False
        if time.monotonic() - ts > self._cooldown:
            self._fallback.pop(key, None)
            return False
        return True

    def _sweep_fallback(self) -> None:
        cutoff = time.monotonic() - 5 * self._cooldown
        for key, ts in list(self._fallback.items()):
            if ts < cutoff:
                self._fallback.pop(key, None)


def _redis_client():
    try:
        from app.services.cache_service import get_redis_client

        return get_redis_client()
    except Exception:
        return None
