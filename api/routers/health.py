"""
نقاط فحص الحالة والإحصائيات
"""
from fastapi import APIRouter

from api.schemas import HealthResponse, MetricsResponse
from api.dependencies import get_predictor, get_model_name, get_uptime
from api.metrics import get_metrics
from api.model_manager import get_model_manager, AVAILABLE_MODELS
from api import __version__


# ============================================================
# تعريف router — يجب أن يكون قبل أي @router.get
# ============================================================
router = APIRouter(tags=["Health"])


# ============================================================
# /health
# ============================================================
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="فحص حالة الخادم",
    description="يعيد حالة الخادم والنموذج المحمّل",
)
async def health_check():
    """فحص حالة الخادم."""
    try:
        get_predictor()
        model_loaded = True
        model_name = get_model_name()
    except Exception:
        model_loaded = False
        model_name = None

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        version=__version__,
        model_loaded=model_loaded,
        model_name=model_name,
        uptime_seconds=round(get_uptime(), 2),
    )


# ============================================================
# /metrics
# ============================================================
@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="إحصائيات الاستخدام",
    description="يعرض عدد الطلبات والتنبؤات ومتوسط الاستجابة",
)
async def metrics():
    """إحصائيات الـ API."""
    m = get_metrics().get_stats()
    manager = get_model_manager()

    return MetricsResponse(
        uptime_seconds=m["uptime_seconds"],
        total_requests=m["total_requests"],
        total_predictions=m["total_predictions"],
        loaded_models=manager.get_loaded_models(),
        available_models=len(AVAILABLE_MODELS),
        average_latency_ms=m["average_latency_ms"],
        requests_per_model=m["requests_per_model"],
    )