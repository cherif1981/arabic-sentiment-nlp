"""
نقاط التنبؤ
"""
import time
from fastapi import APIRouter, HTTPException, Path

from api.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ProbabilityItem,
)
from api.model_manager import get_model_manager, AVAILABLE_MODELS, DEFAULT_MODEL
from api.metrics import get_metrics


# ✅ router معرّف أولاً
router = APIRouter(prefix="/predict", tags=["Predictions"])


# ============================================================
# التنبؤ بالنموذج الافتراضي
# ============================================================
@router.post(
    "",
    response_model=PredictResponse,
    summary="تحليل بنموذج افتراضي (Optimized)",
    description=f"يستخدم النموذج الافتراضي '{DEFAULT_MODEL}' (الأدق في الفئة الكلاسيكية)"
)
async def predict_default(request: PredictRequest):
    """التنبؤ بالنموذج الافتراضي."""
    return await _predict_with_model(request.text, DEFAULT_MODEL)


# ============================================================
# التنبؤ بنموذج محدد
# ============================================================
@router.post(
    "/{model_name}",
    response_model=PredictResponse,
    summary="تحليل بنموذج محدد",
    description="اختر النموذج: optimized, logistic, svm, nb, arabert"
)
async def predict_with_model(
    request: PredictRequest,
    model_name: str = Path(
        ...,
        description="اسم النموذج",
        examples=["optimized", "logistic", "svm", "nb"],
    ),
):
    """التنبؤ بنموذج محدد."""
    return await _predict_with_model(request.text, model_name)


# ============================================================
# الدالة المشتركة
# ============================================================
async def _predict_with_model(text: str, model_name: str) -> PredictResponse:
    """التنبؤ بنموذج معين (مشترك بين النقاط)."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ")

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"النموذج '{model_name}' غير موجود. "
                   f"المتاح: {list(AVAILABLE_MODELS.keys())}"
        )

    manager = get_model_manager()
    metrics = get_metrics()

    try:
        # تسجيل الطلب
        metrics.record_request()

        # التنبؤ
        start = time.time()
        result = manager.predict(text, model_name=model_name)
        latency_ms = (time.time() - start) * 1000

        # تسجيل التنبؤ
        metrics.record_prediction(model_name, latency_ms)

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في التحليل: {e}")

    # بناء الاحتمالات
    probabilities = []
    if "probabilities" in result:
        try:
            predictor = manager.get_predictor(model_name)
            if hasattr(predictor.model, "classes_"):
                classes = list(predictor.model.classes_)
                from api.dependencies import format_probabilities
                probabilities = [
                    ProbabilityItem(**p)
                    for p in format_probabilities(result["probabilities"], classes)
                ]
        except Exception:
            pass

    return PredictResponse(
        text=result["text"],
        cleaned_text=result.get("cleaned", ""),
        sentiment=str(result["label"]),
        sentiment_ar=result["label_ar"],
        confidence=float(result.get("confidence", 0.0)),
        probabilities=probabilities,
        model_name=model_name,
    )


# ============================================================
# Batch (يدعم نموذج محدد)
# ============================================================
@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="تحليل عدة نصوص دفعة واحدة",
    description="يحلّل قائمة من النصوص (حتى 100) بنموذج واحد"
)
async def predict_batch(request: BatchPredictRequest):
    """تحليل عدة نصوص."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="قائمة النصوص فارغة")

    model_name = DEFAULT_MODEL
    manager = get_model_manager()
    metrics = get_metrics()

    try:
        start = time.time()
        results = []

        for text in request.texts:
            if not text.strip():
                continue

            result = manager.predict(text, model_name=model_name)
            latency_ms = (time.time() - start) * 1000

            probabilities = []
            if "probabilities" in result:
                try:
                    predictor = manager.get_predictor(model_name)
                    if hasattr(predictor.model, "classes_"):
                        classes = list(predictor.model.classes_)
                        from api.dependencies import format_probabilities
                        probabilities = [
                            ProbabilityItem(**p)
                            for p in format_probabilities(result["probabilities"], classes)
                        ]
                except Exception:
                    pass

            results.append(PredictResponse(
                text=result["text"],
                cleaned_text=result.get("cleaned", ""),
                sentiment=str(result["label"]),
                sentiment_ar=result["label_ar"],
                confidence=float(result.get("confidence", 0.0)),
                probabilities=probabilities,
                model_name=model_name,
            ))

        elapsed_ms = (time.time() - start) * 1000
        metrics.record_prediction(model_name, elapsed_ms / len(request.texts))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في التحليل: {e}")

    return BatchPredictResponse(
        results=results,
        count=len(results),
        processing_time_ms=round(elapsed_ms, 2),
    )