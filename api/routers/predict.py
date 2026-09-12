"""
نقاط التنبؤ
"""
from fastapi import APIRouter, HTTPException, Depends
from api.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ProbabilityItem
)
from api.dependencies import get_predictor, get_model_name, format_probabilities

import time


router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post(
    "",
    response_model=PredictResponse,
    summary="تحليل مشاعر نص واحد",
    description="يحلّل نصاً عربياً ويعيد التصنيف مع نسبة الثقة",
    responses={
        200: {"description": "تم التحليل بنجاح"},
        400: {"description": "نص فارغ أو غير صالح"},
        500: {"description": "خطأ داخلي"},
    }
)
async def predict_sentiment(request: PredictRequest):
    """
    تحليل مشاعر نص عربي واحد.
    
    - **text**: النص العربي (1 إلى 5000 حرف)
    - **mode**: وضع المعالجة (`classic` أو `light`)
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="النص فارغ")

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        start = time.time()
        result = predictor.predict(request.text)
        elapsed_ms = (time.time() - start) * 1000
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في التحليل: {e}")

    # بناء الاحتمالات
    probabilities = []
    if "probabilities" in result:
        # الحصول على الفئات من النموذج
        classes = list(predictor.model.classes_)
        probabilities = [
            ProbabilityItem(**p)
            for p in format_probabilities(result["probabilities"], classes)
        ]

    return PredictResponse(
        text=result["text"],
        cleaned_text=result.get("cleaned", ""),
        sentiment=str(result["label"]),
        sentiment_ar=result["label_ar"],
        confidence=result.get("confidence", 0.0),
        probabilities=probabilities,
        model_name=get_model_name()
    )


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="تحليل عدة نصوص دفعة واحدة",
    description="يحلّل قائمة من النصوص (حتى 100) ويعيد النتائج"
)
async def predict_batch(request: BatchPredictRequest):
    """
    تحليل مشاعر عدة نصوص دفعة واحدة.
    
    - **texts**: قائمة نصوص (1 إلى 100)
    - **mode**: وضع المعالجة
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="قائمة النصوص فارغة")

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        start = time.time()
        classes = list(predictor.model.classes_)
        results = []

        for text in request.texts:
            if not text.strip():
                continue

            result = predictor.predict(text)

            probabilities = []
            if "probabilities" in result:
                probabilities = [
                    ProbabilityItem(**p)
                    for p in format_probabilities(result["probabilities"], classes)
                ]

            results.append(PredictResponse(
                text=result["text"],
                cleaned_text=result.get("cleaned", ""),
                sentiment=str(result["label"]),
                sentiment_ar=result["label_ar"],
                confidence=result.get("confidence", 0.0),
                probabilities=probabilities,
                model_name=get_model_name()
            ))

        elapsed_ms = (time.time() - start) * 1000
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في التحليل: {e}")

    return BatchPredictResponse(
        results=results,
        count=len(results),
        processing_time_ms=round(elapsed_ms, 2)
    )