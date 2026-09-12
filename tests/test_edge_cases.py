"""
اختبارات الحالات الحدّية والمتقدمة
====================================
يغطي:
- Unicodes غريبة
- بيانات ضخمة
- مشاكل الترميز
"""
import pytest
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocessing import preprocess


class TestUnicodeEdgeCases:
    """حالات Unicode غريبة."""

    def test_zero_width_space(self):
        text = "مر\u200bحبا"
        result = preprocess(text, mode="light")
        assert "\u200b" not in result

    def test_rtl_mark(self):
        text = "مرحبا\u200f"
        result = preprocess(text, mode="light")
        assert "\u200f" not in result

    def test_arabic_indic_digits(self):
        """الأرقام العربية-الهندية."""
        text = "١٢٣٤٥"
        result = preprocess(text, mode="classic")
        assert result.strip() == ""

    def test_extended_arabic(self):
        """حروف عربية موسّعة."""
        text = "ڪتاب"
        result = preprocess(text, mode="light")
        assert len(result) > 0

    def test_diacritics_variety(self):
        """تشكيل متنوع."""
        text = "مُحَمَّدٌ"
        result = preprocess(text, mode="classic")
        assert "ُ" not in result
        assert "َ" not in result
        assert "ّ" not in result
        assert "ٌ" not in result


class TestPerformance:
    """اختبارات الأداء."""

    def test_large_batch_performance(self):
        """معالجة 1000 نص خلال وقت معقول."""
        import time
        texts = ["هذا المنتج ممتاز جداً"] * 1000
        start = time.time()
        results = [preprocess(t, mode="classic") for t in texts]
        elapsed = time.time() - start

        assert len(results) == 1000
        # يجب أن تكون أقل من 5 ثوانٍ
        assert elapsed < 5.0, f"بطيء جداً: {elapsed:.2f}s"

    def test_single_very_long_text(self):
        """نص طويل جداً."""
        text = "المنتج ممتاز " * 10000  # 140K حرف
        result = preprocess(text, mode="classic")
        assert isinstance(result, str)


class TestDataTypes:
    """اختبارات أنواع البيانات."""

    @pytest.mark.parametrize("input_val", [
        None, 123, 45.67, [], {}, True, False,
    ])
    def test_non_string_inputs(self, input_val):
        """مدخلات غير نصية لا تُسبب crash."""
        result = preprocess(input_val, mode="classic")
        assert isinstance(result, str)

    def test_bytes_input(self):
        """مدخل بايت."""
        result = preprocess(b"test", mode="classic")
        assert result == ""

    def test_tuple_input(self):
        """مدخل tuple."""
        result = preprocess(("test",), mode="classic")
        assert result == ""