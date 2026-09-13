"""
Pydantic schemas — توثيق تلقائي للـ API
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

# ✅ Python 3.7 fallback
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# ============================================================
# Input Schemas
# ============================================================
class PredictRequest(BaseModel):
    """طلب تحليل نص واحد."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="النص العربي المراد تحليله",
        examples=["هذا المنتج ممتاز جداً وأنا سعيد بشرائه"]
    )
    mode: Optional[Literal["classic", "light"]] = Field(
        default="classic",
        description="وضع المعالجة"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "هذا المنتج رائع",
                "mode": "classic"
            }
        }
    )


class BatchPredictRequest(BaseModel):
    """طلب تحليل عدة نصوص."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="قائمة النصوص (بحد أقصى 100 نص)"
    )
    mode: Optional[Literal["classic", "light"]] = Field(default="classic")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "هذا المنتج ممتاز",
                    "خدمة سيئة جداً",
                    "المنتج عادي"
                ]
            }
        }
    )


# ============================================================
# Output Schemas
# ============================================================
class ProbabilityItem(BaseModel):
    """احتمال فئة واحدة."""
    label: str = Field(..., description="اسم الفئة")
    label_ar: str = Field(..., description="الاسم بالعربية")
    probability: float = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    """نتيجة تحليل نص واحد."""
    text: str = Field(..., description="النص الأصلي")
    cleaned_text: str = Field(..., description="النص بعد المعالجة")
    sentiment: str = Field(..., description="التصنيف (negative/neutral/positive)")
    sentiment_ar: str = Field(..., description="التصنيف بالعربية")
    confidence: float = Field(..., ge=0, le=1, description="نسبة الثقة")
    probabilities: List[ProbabilityItem] = Field(
        default_factory=list,
        description="توزيع الاحتمالات لكل فئة"
    )
    model_name: str = Field(..., description="اسم النموذج المستخدم")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "هذا المنتج رائع",
                "cleaned_text": "المنتج رائع",
                "sentiment": "positive",
                "sentiment_ar": "إيجابي 😊",
                "confidence": 0.97,
                "probabilities": [
                    {"label": "negative", "label_ar": "سلبي", "probability": 0.01},
                    {"label": "neutral", "label_ar": "محايد", "probability": 0.02},
                    {"label": "positive", "label_ar": "إيجابي", "probability": 0.97}
                ],
                "model_name": "Linear SVM"
            }
        }
    )


class BatchPredictResponse(BaseModel):
    """نتيجة تحليل عدة نصوص."""
    results: List[PredictResponse]
    count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """حالة الخادم."""
    status: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """رد الخطأ."""
    error: str
    detail: Optional[str] = None
    status_code: int

# ============================================================
# Multi-Model Schemas
# ============================================================
class ModelInfo(BaseModel):
    """معلومات نموذج واحد."""
    key: str
    name: str
    description: str
    accuracy: str
    speed: str
    available: bool
    default: bool
    loaded: bool


class ModelsListResponse(BaseModel):
    """قائمة النماذج المتاحة."""
    models: List[ModelInfo]
    default_model: str
    count: int


class MultiModelPredictRequest(BaseModel):
    """طلب تحليل مع اختيار النموذج."""
    text: str = Field(..., min_length=1, max_length=5000)
    model_name: Optional[str] = Field(
        default=None,
        description="اسم النموذج (اختياري، الافتراضي: optimized)"
    )


class MetricsResponse(BaseModel):
    """إحصائيات الـ API."""
    uptime_seconds: float
    total_requests: int
    total_predictions: int
    loaded_models: List[str]
    available_models: int
    average_latency_ms: float
    requests_per_model: dict