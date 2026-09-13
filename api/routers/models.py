"""
نقاط إدارة النماذج
"""
from fastapi import APIRouter, HTTPException

from api.schemas import ModelsListResponse, ModelInfo
from api.model_manager import get_model_manager, AVAILABLE_MODELS, DEFAULT_MODEL


# ============================================================
# تعريف router — قبل أي @router
# ============================================================
router = APIRouter(prefix="/models", tags=["Models"])


# ============================================================
# GET /models — قائمة النماذج
# ============================================================
@router.get(
    "",
    response_model=ModelsListResponse,
    summary="قائمة النماذج المتاحة",
    description="يعرض جميع النماذج مع حالتها",
)
async def list_models():
    """قائمة النماذج المتاحة."""
    manager = get_model_manager()
    models_dict = manager.list_models()

    models = []
    for key, info in models_dict.items():
        models.append(ModelInfo(key=key, **info))

    return ModelsListResponse(
        models=models,
        default_model=DEFAULT_MODEL,
        count=len(models),
    )


# ============================================================
# POST /models/{model_name}/load — تحميل نموذج
# ============================================================
@router.post(
    "/{model_name}/load",
    summary="تحميل نموذج",
    description="يحمّل نموذجاً في الذاكرة (اختياري — يحدث تلقائياً)",
)
async def load_model(model_name: str):
    """تحميل نموذج."""
    manager = get_model_manager()

    try:
        loaded = manager.load_model(model_name)
        return {
            "status": "loaded",
            "model": model_name,
            "name": loaded["full_name"],
            "load_time_sec": round(loaded["load_time"], 3),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ============================================================
# GET /models/loaded — النماذج المحمّلة
# ============================================================
@router.get(
    "/loaded",
    summary="النماذج المحمّلة",
    description="يعرض النماذج المحمّلة في الذاكرة",
)
async def loaded_models():
    """النماذج المحمّلة."""
    manager = get_model_manager()
    loaded = manager.get_loaded_models()

    return {
        "loaded": loaded,
        "count": len(loaded),
    }