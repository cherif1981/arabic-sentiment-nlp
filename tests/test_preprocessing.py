"""
اختبارات preprocessing.py
==========================
يغطي:
- الدوال الأساسية
- الوضعين (classic, light)
- الحالات الحدّية
"""
import pytest
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(CURRENT_DIR), "src"))

from preprocessing import (
    preprocess,
    remove_urls, remove_emails, remove_mentions, remove_hashtags,
    remove_emojis, remove_tashkeel, remove_tatweel,
    normalize_repeated_chars, normalize_arabic, normalize_unicode,
    remove_punctuation, remove_digits, remove_english,
    remove_stopwords, stem_arabic, clean_whitespace,
)


# ============================================================
# 1. الدوال الأساسية
# ============================================================
class TestBasicCleaners:
    """اختبارات دوال التنظيف الفردية."""

    def test_remove_urls(self):
        assert "http" not in remove_urls("زيارة https://example.com")
        assert "www" not in remove_urls("اذهب إلى www.test.com")
        assert remove_urls("نص بدون رابط") == "نص بدون رابط"

    def test_remove_emails(self):
        assert "@" not in remove_emails("تواصل: test@mail.com")

    def test_remove_mentions(self):
        assert "@" not in remove_mentions("@ahmed مرحبا")
        assert "ahmed" not in remove_mentions("@ahmed مرحبا")

    def test_remove_hashtags_keeps_word(self):
        result = remove_hashtags("#رائع منتج", keep_word=True)
        assert "رائع" in result
        assert "#" not in result

    def test_remove_hashtags_removes_word(self):
        result = remove_hashtags("#رائع منتج", keep_word=False)
        assert "رائع" not in result

    def test_remove_emojis(self):
        result = remove_emojis("رائع 😍 جداً 🎉")
        assert "😍" not in result
        assert "🎉" not in result

    def test_remove_tashkeel(self):
        assert remove_tashkeel("مَرْحَبًا") == "مرحبا"
        assert remove_tashkeel("كَتَبَ") == "كتب"

    def test_remove_tatweel(self):
        assert remove_tatweel("مـــرحبا") == "مرحبا"

    def test_normalize_repeated_chars(self):
        assert normalize_repeated_chars("راااائع") == "رائع"
        assert normalize_repeated_chars("جمييييل") == "جميل"
        assert normalize_repeated_chars("كتااااااااب") == "كتاب"
        # كلمتان بحرفين مكررين فقط لا تُعدّلان
        assert normalize_repeated_chars("رد") == "رد"

    def test_normalize_arabic(self):
        assert normalize_arabic("إسلام") == "اسلام"
        assert normalize_arabic("أحمد") == "احمد"
        assert normalize_arabic("آمن") == "امن"
        assert normalize_arabic("مصطفى") == "مصطفي"
        assert normalize_arabic("مدرسة") == "مدرسه"

    def test_remove_punctuation(self):
        assert "!" not in remove_punctuation("مرحبا!")
        assert "؟" not in remove_punctuation("كيف حالك؟")

    def test_remove_digits(self):
        assert "123" not in remove_digits("مرحبا123")
        assert "٢٠٢٤" not in remove_digits("سنة ٢٠٢٤")

    def test_remove_english(self):
        result = remove_english("مرحبا hello")
        assert "hello" not in result
        assert "مرحبا" in result

    def test_remove_stopwords(self):
        result = remove_stopwords("هذا المنتج رائع")
        assert "هذا" not in result
        assert "المنتج" in result

    def test_stem_arabic(self):
        result = stem_arabic("الكتاب جميل")
        # ISRI stemmer يُرجع جذر الكلمة
        assert len(result.split()) == 2

    def test_clean_whitespace(self):
        assert clean_whitespace("مرحبا    بالعالم") == "مرحبا بالعالم"
        assert clean_whitespace("  نص  ") == "نص"


# ============================================================
# 2. الوضعان
# ============================================================
class TestPreprocessModes:
    """اختبارات وضعي المعالجة."""

    def test_light_keeps_tashkeel(self):
        result = preprocess("مَرْحَبًا", mode="light")
        assert "َ" in result  # الفتحة باقية

    def test_classic_removes_tashkeel(self):
        result = preprocess("مَرْحَبًا", mode="classic")
        assert "َ" not in result

    def test_light_keeps_stopwords(self):
        result = preprocess("هذا المنتج رائع", mode="light")
        assert "هذا" in result

    def test_classic_removes_stopwords(self):
        result = preprocess("هذا المنتج رائع", mode="classic")
        assert "هذا" not in result

    def test_light_keeps_digits(self):
        result = preprocess("المنتج 10/10", mode="light")
        assert "10" in result

    def test_classic_removes_digits(self):
        result = preprocess("المنتج 10/10", mode="classic")
        assert "10" not in result

    def test_classic_stem(self):
        result = preprocess("الكتاب جميل", mode="classic", do_stem=True)
        assert len(result.split()) == 2

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="غير مدعوم"):
            preprocess("نص", mode="invalid")


# ============================================================
# 3. الحالات الحدّية
# ============================================================
class TestEdgeCases:
    """اختبارات الحالات الحدّية."""

    def test_empty_string(self):
        assert preprocess("") == ""
        assert preprocess("", mode="light") == ""
        assert preprocess("", mode="classic") == ""

    def test_whitespace_only(self):
        assert preprocess("     ") == ""
        assert preprocess("\n\n\t") == ""

    def test_none_input(self):
        assert preprocess(None) == ""
        assert preprocess(None, mode="light") == ""

    def test_numbers_only(self):
        result = preprocess("123 456", mode="classic")
        assert result.strip() == ""

    def test_punctuation_only(self):
        result = preprocess("!@#$%^&*()", mode="classic")
        assert result.strip() == ""

    def test_emoji_only(self):
        result = preprocess("😍🎉😊", mode="classic")
        assert result.strip() == ""

    def test_very_long_text(self):
        long_text = "هذا المنتج ممتاز " * 1000
        result = preprocess(long_text, mode="classic")
        assert len(result) > 0
        # يجب أن يكون أقصر من الأصل (إزالة التكرار)
        assert len(result) < len(long_text)

    def test_arabic_with_english(self):
        result = preprocess("منتج product رائع", mode="classic")
        assert "product" not in result
        assert "منتج" in result

    def test_mixed_everything(self):
        text = "😍 مرحباً!!! @ahmed https://test.com #رائع 123"
        result = preprocess(text, mode="classic")
        assert "http" not in result
        assert "@" not in result
        assert "😍" not in result
        assert "123" not in result


# ============================================================
# 4. Integration
# ============================================================
class TestFullPipeline:
    """اختبارات خط الأنابيب الكامل."""

    def test_real_tweet_classic(self):
        tweet = "رااااائع 😍 @user https://example.com #رائع!!! ١٠/١٠"
        result = preprocess(tweet, mode="classic")

        # التحقق من إزالة العناصر
        assert "http" not in result
        assert "@" not in result
        assert "😍" not in result
        assert "راااائع" not in result
        assert "١٠" not in result

        # التحقق من وجود كلمة بعد التطبيع
        # normalize_arabic يحوّل ئ → ي
        # لذا نتوقع "رايع" (أو "رائع" لو لم يُطبّع)
        words = result.split()
        assert len(words) >= 1, f"النتيجة فارغة: {result!r}"

        # نتأكد أن النتيجة تحتوي على "راي" (بداية كلمة "رايع" بعد التطبيع)
        # أو "رائ" (بداية "رائع" بدون تطبيع)
        assert any(w.startswith(("راي", "رائ")) for w in words), \
            f"لا توجد كلمة تبدأ بـ 'راي' أو 'رائ' في: {words}"

    def test_real_tweet_light(self):
        tweet = "رااااائع 😍 @user https://example.com #رائع!!!"
        result = preprocess(tweet, mode="light")
        assert "http" not in result
        assert "@" not in result
        assert "😍" not in result
        assert "رائع" in result  # التكرار تم توحيده (في الوضعين)

    def test_arabic_normalization_classic(self):
        result = preprocess("إلى المدرسة", mode="classic")
        assert "إ" not in result
        assert "ة" not in result

    def test_unicode_normalization(self):
        # Zero-width space
        text = "مر\u200bحبا"
        result = preprocess(text, mode="light")
        assert "\u200b" not in result


# ============================================================
# 5. Parametrized Tests
# ============================================================
@pytest.mark.parametrize("text,expected_contains", [
    ("راااااااااائع", "رائع"),
    ("جميييييييييل", "جميل"),
    ("كتاااااااااااااب", "كتاب"),
])
def test_repeated_chars_parametrized(text, expected_contains):
    """اختبار حدود تكرار الحروف."""
    result = normalize_repeated_chars(text)
    assert expected_contains in result


@pytest.mark.parametrize("text", [
    "",
    "   ",
    None,
    "!@#$",
    "😍😍😍",
    "123",
])
def test_empty_after_classic(text):
    """نصوص تُصبح فارغة بعد الوضع الكلاسيكي."""
    result = preprocess(text, mode="classic")
    assert result.strip() == ""