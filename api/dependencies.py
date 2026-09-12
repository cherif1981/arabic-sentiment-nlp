"""
التبعيات المشتركة (Dependency Injection)
"""
import os
import sys
import time
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from predict import SentimentPredictor, LABEL_MAP


# وقت بدء الخادم
START_TIME = time.time()

# النموذج المحمّل
_predictor = None


def get_predictor() -> SentimentPredictor:
    """
    تحميل النموذج (Singleton).
    يُحمَّل مرة واحدة عند أول استدعاء.
    """
    global _predictor
    if _predictor is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "sentiment_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"النموذج غير موجود في {model_path}\n"
                f"شغّل: python src/train.py"
            )
        _predictor = SentimentPredictor(model_path=model_path)
    return _predictor


def get_model_name() -> str:
    """اسم النموذج الحالي."""
    try:
        predictor = get_predictor()
        return type(predictor.model).__name__
    except Exception:
        return "unknown"


def get_uptime() -> float:
    """زمن تشغيل الخادم بالثواني."""
    return time.time() - START_TIME


def format_probabilities(probs: list, classes: list) -> list:
    """تحويل الاحتمالات إلى صيغة موحّدة."""
    result = []
    for label, prob in zip(classes, probs):
        result.append({
            "label": str(label),
            "label_ar": LABEL_MAP.get(label, str(label)),
            "probability": round(float(prob), 4)
        })
    return result