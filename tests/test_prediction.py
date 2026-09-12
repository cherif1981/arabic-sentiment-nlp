"""
اختبارات predict.py
====================
يغطي:
- تحميل النموذج
- التنبؤ بنصوص مختلفة
- صيغة النتيجة
- الحالات الحدّية
"""
import pytest
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from predict import SentimentPredictor, LABEL_MAP


# ============================================================
# 1. تحميل النموذج
# ============================================================
class TestModelLoading:
    """اختبارات تحميل النموذج."""

    @pytest.mark.requires_model
    def test_model_loads(self, predictor):
        """النموذج يُحمّل بدون أخطاء."""
        assert predictor is not None
        assert predictor.model is not None

    def test_missing_model_raises(self, project_root):
        """ملف مفقود يرفع FileNotFoundError."""
        fake_path = os.path.join(project_root, "models", "nonexistent.pkl")
        with pytest.raises(FileNotFoundError):
            SentimentPredictor(model_path=fake_path)

    @pytest.mark.requires_model
    def test_model_has_classes(self, predictor):
        """النموذج يعرف الفئات."""
        assert hasattr(predictor.model, "classes_")
        classes = list(predictor.model.classes_)
        assert len(classes) >= 2


# ============================================================
# 2. التنبؤ الأساسي
# ============================================================
class TestBasicPrediction:
    """اختبارات التنبؤ الأساسية."""

    @pytest.mark.requires_model
    def test_predict_returns_dict(self, predictor, sample_texts):
        """التنبؤ يُعيد قاموساً."""
        result = predictor.predict(sample_texts["positive"])
        assert isinstance(result, dict)

    @pytest.mark.requires_model
    def test_result_has_required_keys(self, predictor, sample_texts):
        """النتيجة تحتوي المفاتيح المطلوبة."""
        result = predictor.predict(sample_texts["positive"])
        required_keys = {"text", "cleaned", "label", "label_ar"}
        assert required_keys.issubset(result.keys())

    @pytest.mark.requires_model
    def test_result_label_in_classes(self, predictor, sample_texts):
        """التصنيف ضمن الفئات المعروفة."""
        result = predictor.predict(sample_texts["positive"])
        assert result["label"] in predictor.model.classes_

    @pytest.mark.requires_model
    def test_confidence_in_range(self, predictor, sample_texts):
        """الثقة بين 0 و 1."""
        result = predictor.predict(sample_texts["positive"])
        if "confidence" in result:
            assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.requires_model
    def test_probabilities_sum_to_one(self, predictor, sample_texts):
        """مجموع الاحتمالات ≈ 1."""
        result = predictor.predict(sample_texts["positive"])
        if "probabilities" in result:
            total = sum(result["probabilities"])
            assert abs(total - 1.0) < 0.01

    @pytest.mark.requires_model
    def test_text_preserved(self, predictor, sample_texts):
        """النص الأصلي محفوظ."""
        result = predictor.predict(sample_texts["positive"])
        assert result["text"] == sample_texts["positive"]

    @pytest.mark.requires_model
    def test_cleaned_text_different(self, predictor):
        """النص المنظّف مختلف عن الأصلي."""
        text = "رااااائع 😍🎉!!!"
        result = predictor.predict(text)
        assert result["cleaned"] != text


# ============================================================
# 3. حالات الحدّية
# ============================================================
class TestPredictionEdgeCases:
    """اختبارات الحالات الحدّية."""

    @pytest.mark.requires_model
    def test_empty_text(self, predictor):
        """النص الفارغ لا يُسبب crash."""
        result = predictor.predict("")
        assert isinstance(result, dict)
        assert result["cleaned"] == ""

    @pytest.mark.requires_model
    def test_whitespace_only(self, predictor):
        """المسافات فقط."""
        result = predictor.predict("     ")
        assert isinstance(result, dict)

    @pytest.mark.requires_model
    def test_none_input(self, predictor):
        """None لا يُسبب crash."""
        result = predictor.predict(None)
        assert isinstance(result, dict)

    @pytest.mark.requires_model
    def test_number_input(self, predictor):
        """رقم لا يُسبب crash."""
        result = predictor.predict(123)
        assert isinstance(result, dict)

    @pytest.mark.requires_model
    def test_very_long_text(self, predictor):
        """نص طويل جداً."""
        long_text = "هذا المنتج ممتاز جداً " * 500
        result = predictor.predict(long_text)
        assert isinstance(result, dict)
        assert result["label"] in predictor.model.classes_

    @pytest.mark.requires_model
    def test_punctuation_only(self, predictor):
        """نص من رموز فقط."""
        result = predictor.predict("!@#$%^&*()")
        assert isinstance(result, dict)

    @pytest.mark.requires_model
    def test_emoji_only(self, predictor):
        """نص من إيموجي فقط."""
        result = predictor.predict("😍😍😍")
        assert isinstance(result, dict)

    @pytest.mark.requires_model
    def test_urls_only(self, predictor):
        """نص من روابط فقط."""
        result = predictor.predict("https://test.com https://example.org")
        assert isinstance(result, dict)

    @pytest.mark.requires_model
    def test_english_only(self, predictor):
        """نص إنجليزي فقط."""
        result = predictor.predict("This is a test")
        assert isinstance(result, dict)

    @pytest.mark.requires_model
    def test_mixed_languages(self, predictor):
        """نص مختلط."""
        result = predictor.predict("منتج product جيد good جداً")
        assert isinstance(result, dict)


# ============================================================
# 4. Batch Prediction
# ============================================================
class TestBatchPrediction:
    """اختبارات التنبؤ بالدفعات."""

    @pytest.mark.requires_model
    def test_batch_returns_list(self, predictor):
        """Batch يُعيد قائمة."""
        texts = ["رائع", "سيء", "عادي"]
        results = predictor.predict_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 3

    @pytest.mark.requires_model
    def test_batch_preserves_order(self, predictor):
        """الترتيب محفوظ."""
        texts = ["نص أول", "نص ثاني", "نص ثالث"]
        results = predictor.predict_batch(texts)
        for i, result in enumerate(results):
            assert result["text"] == texts[i]

    @pytest.mark.requires_model
    def test_batch_empty_list(self, predictor):
        """قائمة فارغة."""
        results = predictor.predict_batch([])
        assert results == []

    @pytest.mark.requires_model
    def test_batch_with_empty_items(self, predictor):
        """قائمة تحتوي نصوصاً فارغة."""
        texts = ["رائع", "", "سيء", "   "]
        results = predictor.predict_batch(texts)
        assert len(results) == 4


# ============================================================
# 5. صحة التصنيف
# ============================================================
class TestPredictionAccuracy:
    """اختبارات دقة التنبؤ."""

    @pytest.mark.requires_model
    @pytest.mark.skip(reason="يتطلب بيانات كافية - قد يفشل مع بيانات صغيرة")
    def test_obvious_positive(self, predictor):
        """نص إيجابي واضح يُصنّف إيجابياً."""
        result = predictor.predict("هذا المنتج ممتاز ورائع وأنا سعيد جداً")
        assert result["label"] == "positive"

    @pytest.mark.requires_model
    @pytest.mark.skip(reason="يتطلب بيانات كافية")
    def test_obvious_negative(self, predictor):
        """نص سلبي واضح يُصنّف سلبياً."""
        result = predictor.predict("خدمة سيئة للغاية ولن أشتري مرة أخرى")
        assert result["label"] == "negative"


# ============================================================
# 6. Label Map
# ============================================================
class TestLabelMap:
    """اختبارات خريطة التصنيفات."""

    def test_label_map_complete(self):
        """كل الفئات موجودة."""
        assert "positive" in LABEL_MAP
        assert "negative" in LABEL_MAP
        assert "neutral" in LABEL_MAP

    def test_label_map_arabic(self):
        """الترجمة العربية صحيحة."""
        assert "إيجابي" in LABEL_MAP["positive"]
        assert "سلبي" in LABEL_MAP["negative"]
        assert "محايد" in LABEL_MAP["neutral"]