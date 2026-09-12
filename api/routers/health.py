"""
نقاط فحص الحالة
"""
from fastapi import APIRouter, Depends
from api.schemas import HealthResponse
from api.dependencies import get_predictor, get_model_name, get_uptime
from api import __version__


router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="فحص حالة الخادم",
    description="يعيد حالة الخادم والنموذج المحمّل"
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
        uptime_seconds=round(get_uptime(), 2)
    )


@router.get(
    "/",
    summary="معلومات الـ API",
    description="الصفحة الرئيسية للـ API"
)
async def root():
    """الصفحة الرئيسية."""
    return {
        "name": "Arabic Sentiment Analyzer API",
        "version": __version__,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
        },
        "labels": ["negative", "neutral", "positive"]
    }