from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, ServicesDep
from app.api.schemas import UsageOut

router = APIRouter(prefix="/api/usage", tags=["usage"])


@router.get("", response_model=UsageOut)
def my_usage(
    user: CurrentUser,
    services: ServicesDep,
    hours: Annotated[int, Query(ge=1, le=720, description="Look back this many hours")] = 24,
):
    """How many questions, model calls and tokens the signed-in user used recently."""
    totals = services.traces.usage(user.id, hours)
    s = services.settings
    priced = s.llm_input_price_per_million or s.llm_output_price_per_million
    cost = (
        (totals["input_tokens"] * s.llm_input_price_per_million + totals["output_tokens"] * s.llm_output_price_per_million)
        / 1_000_000
        if priced
        else None
    )
    return UsageOut(hours=hours, estimated_cost=cost, **totals)
