"""
التنبؤ بمشاعر نص عربي جديد
"""
import os
import sys
import joblib
import traceback

# ✅ إصلاح UTF-8 على Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

# ✅ الاستيراد الجديد
try:
    from preprocessing import preprocess
except ImportError as e:
    print(f"[predict.py] ImportError: {e}")
    print("[predict.py] تأكد من أن preprocessing.py يحتوي على دالة preprocess()")
    raise


MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_model.pkl")

LABEL_MAP = {
    0: "سلبي 😞",
    1: "محايد 😐",
    2: "إيجابي 😊",
    "negative": "سلبي 😞",
    "positive": "إيجابي 😊",
    "neutral": "محايد 😐"
}


class SentimentPredictor:
    """فئة للتنبؤ بمشاعر النصوص العربية."""

    def __init__(self, model_path: str = MODEL_PATH, mode: str = "classic"):
        """
        Args:
            model_path: مسار النموذج المدرب
            mode: وضع المعالجة ("classic" أو "light")
        """
        self.mode = mode
        self.model_path = model_path

        # التحقق من وجود الملف
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ لم يتم العثور على النموذج في: {model_path}\n"
                f"💡 شغّل: python src/train.py"
            )

        # تحميل النموذج
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"[predict.py] فشل تحميل النموذج: {e}")
            traceback.print_exc()
            raise

    def _clean(self, text: str) -> str:
        """تنظيف النص باستخدام الوضع المحدد."""
        return preprocess(text, mode=self.mode)

    def predict(self, text: str) -> dict:
        """التنبؤ بمشاعر نص واحد."""
        if not isinstance(text, str) or not text.strip():
            return {
                "text": text or "",
                "cleaned": "",
                "label": None,
                "label_ar": "نص فارغ ⚠️",
                "confidence": 0.0,
                "probabilities": [],
            }

        cleaned = self._clean(text)

        # التنبؤ
        pred = self.model.predict([cleaned])[0]

        result = {
            "text": text,
            "cleaned": cleaned,
            "label": pred,
            "label_ar": LABEL_MAP.get(pred, str(pred))
        }

        # الاحتمالات (إن كانت متاحة)
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba([cleaned])[0]
                result["confidence"] = float(max(probs))
                result["probabilities"] = probs.tolist()
            except Exception:
                pass
        elif hasattr(self.model, "decision_function"):
            try:
                import numpy as np
                decision = self.model.decision_function([cleaned])[0]
                # Softmax
                exp = np.exp(decision - decision.max())
                probs = exp / exp.sum()
                result["confidence"] = float(probs.max())
                result["probabilities"] = probs.tolist()
            except Exception:
                pass

        return result

    def predict_batch(self, texts: list) -> list:
        """التنبؤ بمجموعة نصوص."""
        return [self.predict(t) for t in texts]


def main():
    """اختبار سريع."""
    print("=" * 60)
    print("Arabic Sentiment Predictor - Test")
    print("=" * 60)

    try:
        predictor = SentimentPredictor()
        print(f"[OK] Model loaded: {type(predictor.model).__name__}\n")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    samples = [
        "هذا المنتج رائع جداً وأنا سعيد بشرائه",
        "تجربة سيئة للغاية، لن أشتري مرة أخرى",
        "المنتج عادي، لا شيء مميز فيه",
        "خدمة العملاء ممتازة وسريعة في الرد"
    ]

    for s in samples:
        result = predictor.predict(s)
        print(f"\n📝 النص: {result['text']}")
        print(f"🧹 بعد التنظيف: {result['cleaned']}")
        print(f"🎯 التصنيف: {result['label_ar']}")
        if "confidence" in result:
            print(f"📊 الثقة: {result['confidence']:.2%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()