"""
مدير النماذج — يدير تحميل وتخزين جميع النماذج
"""
import os
import sys
import time
import joblib
from typing import Optional, Dict
from threading import Lock

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


# ============================================================
# معلومات النماذج المتاحة
# ============================================================
AVAILABLE_MODELS = {
    "optimized": {
        "name": "TF-IDF + Logistic Regression (Optimized)",
        "path": os.path.join(PROJECT_ROOT, "models", "tfidf_logistic_optimized.pkl"),
        "desc": "النموذج المُحسَّن بـ Grid Search — الأدق في الفئة الكلاسيكية",
        "accuracy": "~61%",
        "speed": "⚡⚡⚡",
        "type": "sklearn",
    },
    "logistic": {
        "name": "TF-IDF + Logistic Regression",
        "path": os.path.join(PROJECT_ROOT, "models", "model_logistic_regression.pkl"),
        "desc": "النموذج الأساسي — متوازن وسريع",
        "accuracy": "~58%",
        "speed": "⚡⚡⚡",
        "type": "sklearn",
    },
    "svm": {
        "name": "TF-IDF + Linear SVM",
        "path": os.path.join(PROJECT_ROOT, "models", "model_linear_svm.pkl"),
        "desc": "يدعم التصنيف الخطي بسرعة عالية",
        "accuracy": "~58%",
        "speed": "⚡⚡⚡",
        "type": "sklearn",
    },
    "nb": {
        "name": "TF-IDF + Naive Bayes",
        "path": os.path.join(PROJECT_ROOT, "models", "model_naive_bayes.pkl"),
        "desc": "الأسرع في التدريب",
        "accuracy": "~42%",
        "speed": "⚡⚡⚡",
        "type": "sklearn",
    },
    "arabert": {
        "name": "AraBERT",
        "path": os.path.join(PROJECT_ROOT, "models", "arabert"),
        "desc": "نموذج Transformer — الأدق (يحتاج GPU)",
        "accuracy": "~85%+",
        "speed": "🐢",
        "type": "arabert",
    },
}

DEFAULT_MODEL = "optimized"


# ============================================================
# Model Manager
# ============================================================
class ModelManager:
    """
    مدير النماذج — Singleton thread-safe.
    يحمّل كل نموذج مرة واحدة فقط ويخزّنه في الذاكرة.
    """

    def __init__(self):
        self._models: Dict[str, dict] = {}
        self._lock = Lock()
        self._default_model = DEFAULT_MODEL

    def list_models(self) -> dict:
        """قائمة النماذج المتاحة."""
        result = {}
        for key, info in AVAILABLE_MODELS.items():
            exists = os.path.exists(info["path"])
            result[key] = {
                "name": info["name"],
                "description": info["desc"],
                "accuracy": info["accuracy"],
                "speed": info["speed"],
                "available": exists,
                "default": key == self._default_model,
                "loaded": key in self._models,
            }
        return result

    def load_model(self, model_name: str) -> dict:
        """
        تحميل نموذج (مع التخزين المؤقت).
        
        Returns:
            dict: {"name": ..., "type": ..., "model": ..., "load_time": ...}
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"النموذج '{model_name}' غير معروف. "
                f"المتاح: {list(AVAILABLE_MODELS.keys())}"
            )

        # إذا كان محمّلاً مسبقاً
        with self._lock:
            if model_name in self._models:
                return self._models[model_name]

        info = AVAILABLE_MODELS[model_name]

        # التحقق من وجود الملف
        if not os.path.exists(info["path"]):
            raise FileNotFoundError(
                f"النموذج '{model_name}' غير موجود في: {info['path']}"
            )

        # التحميل
        start = time.time()

        if info["type"] == "sklearn":
            model = joblib.load(info["path"])
        elif info["type"] == "arabert":
            from arabert_predict import AraBERTPredictor
            model = AraBERTPredictor(model_dir=info["path"])
        else:
            raise ValueError(f"نوع نموذج غير معروف: {info['type']}")

        load_time = time.time() - start

        loaded = {
            "name": model_name,
            "full_name": info["name"],
            "type": info["type"],
            "model": model,
            "path": info["path"],
            "load_time": load_time,
        }

        # تخزين مؤقت
        with self._lock:
            self._models[model_name] = loaded

        return loaded

    def get_predictor(self, model_name: str = None):
        """الحصول على النموذج الجاهز للتنبؤ."""
        if model_name is None:
            model_name = self._default_model

        loaded = self.load_model(model_name)

        if loaded["type"] == "sklearn":
            # استخدم SentimentPredictor لضمان نفس preprocessing
            from predict import SentimentPredictor
            predictor = SentimentPredictor(
                model_path=loaded["path"],
                mode="classic"
            )
            return predictor
        elif loaded["type"] == "arabert":
            return loaded["model"]

        raise ValueError(f"نوع غير مدعوم: {loaded['type']}")

    def predict(self, text: str, model_name: str = None) -> dict:
        """
        التنبؤ بـ TF-IDF + Logistic Regression.
        
        Args:
            text: النص العربي
            model_name: اسم النموذج (اختياري)
        """
        predictor = self.get_predictor(model_name)
        return predictor.predict(text)

    def is_loaded(self, model_name: str) -> bool:
        """هل النموذج محمّل؟"""
        return model_name in self._models

    def get_loaded_models(self) -> list:
        """أسماء النماذج المحمّلة."""
        return list(self._models.keys())

    def unload_model(self, model_name: str):
        """إلغاء تحميل نموذج."""
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]


# ============================================================
# Singleton
# ============================================================
_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """الحصول على مدير النماذج (Singleton)."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager