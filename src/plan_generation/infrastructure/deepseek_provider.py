"""DeepSeek LLM provider — implements `LLMProvider` port.

DeepSeek exposes an OpenAI-compatible chat-completions API, so this is a thin
subclass of `OpenAILLMProvider`: same forced single function-call, same response
parsing, same preflight contract. Only the endpoint, the credential, the model
default, the pricing table and the reported provider name differ. Reimplementing
the request/response handling would mean two copies of the tool-call extraction
logic drifting apart.

WHY IT EXISTS. As of 2026-07-22 the plan-generation chain had three legs and only
ONE that could actually serve: Anthropic was out of credit balance (HTTP 400) and
Google was over its free-tier quota (HTTP 429). A three-leg chain where two legs
are dead is decorative — production had no working failover at all. A
`DEEPSEEK_API_KEY` was already configured on App Service but no provider existed
to use it.

Verified against the live account before this was written:

    simple call      -> 200 OK  finish=stop        content='OK'
    forced tool call -> 200 OK  finish=tool_calls  tool=submit_plan
                        args={"diagnosis": "Post-operative anterior cruciate
                              ligament (ACL) reconstruction...", ...}

That second line is the one that matters: it is the exact shape
`call_with_tool` sends, so the inherited implementation is known to work rather
than assumed to.

CLINICAL NOTE. Like OpenAI, this is a FAILOVER leg. Which model authors a
clinical plan is an owner decision (2026-07-01, revised 2026-07-07); adding a
provider here changes what is available, not that policy.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.features.plan_generation.infrastructure.openai_provider import OpenAILLMProvider

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Per-million-token pricing (USD). Kept separate from _OPENAI_PRICING so cost
# telemetry does not silently bill DeepSeek traffic at OpenAI rates.
_DEEPSEEK_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}
_DEFAULT_DEEPSEEK_PRICING = (0.27, 1.10)


class DeepSeekLLMProvider(OpenAILLMProvider):
    """Plan-generation provider over DeepSeek's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        # Admin-supplied key (llm_provider_configs.api_key_encrypted) wins; fall
        # back to env for the bootstrap path, matching the other providers.
        resolved_key = api_key or getattr(settings, "DEEPSEEK_API_KEY", "")
        if not resolved_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

        # max_retries=0: retries are owned by the application (use-case budget +
        # cooldown guard), never stacked invisibly inside the SDK.
        self._client = client or AsyncOpenAI(
            api_key=resolved_key,
            base_url=DEEPSEEK_BASE_URL,
            max_retries=0,
        )
        self._model = model or getattr(settings, "DEEPSEEK_GENERATION_MODEL", "deepseek-chat")
        self._timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "LLM_TIMEOUT_OPENAI", 90)
        )

    async def preflight(self) -> dict[str, Any]:
        """Inherit OpenAI's preflight, but report the correct provider name.

        The parent hard-codes ``provider: "openai"``. Left uncorrected, a DeepSeek
        outage would be attributed to OpenAI in /health and in the chain's
        `degraded_from` list — sending an operator to the wrong vendor during an
        incident, which is precisely the confusion the 2026-07-22 outage was made
        worse by.
        """
        result = await super().preflight()
        result["provider"] = "deepseek"
        return result


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """DeepSeek pricing. Exposed for the same reason the OpenAI one is: cost
    telemetry is only useful if it is attributed to the right vendor's rates."""
    in_rate, out_rate = _DEEPSEEK_PRICING.get(model, _DEFAULT_DEEPSEEK_PRICING)
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
