"""
اختبارات وحدة لـ preprocessing.py
====================================
تركيز خاص على:
- اللهجة الجزائرية (بصح، ماشي، واش، هايلة...)
- الكلمات البذيئة (يجب ألا تُمسح)
- الكلمات الفرنسية (الدارجة مختلطة)
- النصوص المختلطة (مدح + ذم)
- تجنب مشاكل ترميز Windows (لا طباعة عربية في رسائل الفشل)

تشغيل:
    pytest tests/test_preprocessing.py -v
    pytest tests/test_preprocessing.py -v --basetemp=tmp_test
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocessing import (
    clean_text,
    remove_diacritics,
    remove_elongation,
    normalize_letters,
    normalize_digits,
    remove_urls_mentions,
    normalize_emoji,
    normalize_punctuation,
)


# ============================================================
# أدوات مساعدة للاختبار (ASCII-safe)
# ============================================================
def write_debug(tmp_path, name, **kwargs):
    """يكتب تفاصيل الاختبار في ملف UTF-8 (لتجنب مشاكل الطرفية)."""
    path = tmp_path / f"{name}.txt"
    lines = []
    for k, v in kwargs.items():
        if isinstance(v, str):
            lines.append(f"{k}: {v}")
            lines.append(f"{k}_hex: {v.encode('utf-8').hex()}")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def ascii_safe(s):
    """يحوّل النص إلى تمثيل ASCII آمن (لرسائل الفشل)."""
    return s.encode("unicode_escape").decode("ascii")


# ============================================================
# 1) التطويل (Elongation)
# ============================================================
class TestElongation:
    def test_three_repeats_collapsed(self):
        assert remove_elongation("هاييييلة") == "هايلة"

    def test_two_repeats_kept(self):
        assert remove_elongation("مرحباا") == "مرحباا"

    def test_waw_repeats(self):
        assert remove_elongation("جمييييل") == "جميل"

    def test_many_repeats(self):
        assert remove_elongation("راااااااائع") == "رائع"

    def test_no_repeats(self):
        assert remove_elongation("مرحبا") == "مرحبا"

    def test_mixed_repeats(self):
        result = remove_elongation("هااااايل بزاااااف")
        assert result == "هايل بزاف"


# ============================================================
# 2) التشكيل (Diacritics)
# ============================================================
class TestDiacritics:
    def test_fatha_damma_kasra(self):
        assert remove_diacritics("مَرْحَبًا") == "مرحبا"

    def test_shadda(self):
        assert remove_diacritics("مُحَمَّد") == "محمد"

    def test_tatweel(self):
        assert remove_diacritics("مـــن") == "من"

    def test_sukun(self):
        assert remove_diacritics("مَكْتَبْ") == "مكتب"

    def test_no_diacritics(self):
        assert remove_diacritics("مرحبا") == "مرحبا"

    def test_full_sentence(self):
        assert remove_diacritics("مَرْحَبًا بِكُمْ فِي مُنْتَدَانَا") == "مرحبا بكم في منتدانا"


# ============================================================
# 3) توحيد الأحرف (حساس للدارجة)
# ============================================================
class TestLettersNormalization:
    def test_alef_variants(self):
        assert normalize_letters("أحمد") == "احمد"
        assert normalize_letters("إبراهيم") == "ابراهيم"
        assert normalize_letters("آمنة") == "امنه"

    def test_yaa(self):
        assert normalize_letters("على") == "علي"
        assert normalize_letters("مصطفى") == "مصطفي"

    def test_taa_marbuta(self):
        assert normalize_letters("مدرسة") == "مدرسه"

    def test_algerian_letters_guaf(self):
        # ڨ (قاف جزائرية) → ق
        assert normalize_letters("ڨاع") == "قاع"

    def test_algerian_letters_veh(self):
        # ڤ → ف
        assert normalize_letters("ڤرن") == "فرن"

    def test_algerian_letters_gaf(self):
        # گ → ك
        assert normalize_letters("گاع") == "كاع"

    def test_no_change(self):
        assert normalize_letters("مرحبا") == "مرحبا"


# ============================================================
# 4) الأرقام
# ============================================================
class TestDigits:
    def test_arabic_digits(self):
        assert normalize_digits("١٢٣") == "123"

    def test_mixed_digits(self):
        assert normalize_digits("١2٣") == "123"

    def test_no_digits(self):
        assert normalize_digits("مرحبا") == "مرحبا"

    def test_full_number(self):
        assert normalize_digits("٠١٢٣٤٥٦٧٨٩") == "0123456789"


# ============================================================
# 5) الروابط والإشارات
# ============================================================
class TestUrlsMentions:
    def test_url_removed(self):
        result = remove_urls_mentions("شوف هذا https://example.com رائع")
        assert "https" not in result
        assert "example.com" not in result
        assert "رائع" in result

    def test_www_removed(self):
        result = remove_urls_mentions("زيارة www.example.com للتفاصيل")
        assert "www" not in result
        assert "للتفاصيل" in result

    def test_mention_removed(self):
        result = remove_urls_mentions("@user مرحبا")
        assert "@user" not in result
        assert "مرحبا" in result

    def test_hashtag_kept_as_word(self):
        result = remove_urls_mentions("هذا #رائع جدا")
        assert "#" not in result
        assert "رائع" in result

    def test_multiple(self):
        result = remove_urls_mentions("@a @b https://x.com #test كلمة")
        assert "@a" not in result
        assert "@b" not in result
        assert "https" not in result
        assert "test" in result
        assert "كلمة" in result


# ============================================================
# 6) الإيموجي
# ============================================================
class TestEmoji:
    def test_positive_emoji_to_word(self):
        result = normalize_emoji("منتج ممتاز 😍")
        assert "حب" in result
        assert "😍" not in result

    def test_negative_emoji_to_word(self):
        result = normalize_emoji("خدمة سيئة 😡")
        assert "غاضب" in result

    def test_emoji_removed_mode(self):
        result = normalize_emoji("منتج ممتاز 😍", keep_as_words=False)
        assert "😍" not in result
        assert "ممتاز" in result

    def test_multiple_emoji(self):
        result = normalize_emoji("رائع 😍🎉")
        assert "😍" not in result
        assert "🎉" not in result


# ============================================================
# 7) الترقيم
# ============================================================
class TestPunctuation:
    def test_repeated_exclamation(self):
        assert normalize_punctuation("رائع!!!") == "رائع!"

    def test_repeated_question(self):
        assert normalize_punctuation("لماذا؟؟؟") == "لماذا؟"

    def test_multiple_spaces(self):
        assert normalize_punctuation("كلمة    كلمة") == "كلمة كلمة"

    def test_leading_trailing_spaces(self):
        assert normalize_punctuation("  مرحبا  ") == "مرحبا"

    def test_tabs_newlines(self):
        assert normalize_punctuation("كلمة\t\nكلمة") == "كلمة كلمة"


# ============================================================
# 8) الدالة الرئيسية (clean_text) — الحالات العامة
# ============================================================
class TestCleanTextGeneral:
    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_only_spaces(self):
        assert clean_text("     ") == ""

    def test_simple_arabic(self):
        assert clean_text("مرحبا") == "مرحبا"

    def test_with_tashkeel(self, tmp_path):
        result = clean_text("مَرْحَبًا")
        log = write_debug(tmp_path, "tashkeel", input="مَرْحَبًا", output=result)
        assert "مرحبا" in result or result == "مرحبا", f"See {log}"

    def test_with_url(self):
        result = clean_text("زيارة https://example.com للتفاصيل")
        assert "https" not in result
        assert "example.com" not in result
        assert "للتفاصيل" in result

    def test_with_mention(self):
        result = clean_text("@ahmed شكراً لك")
        assert "@ahmed" not in result
        assert "شكر" in result or "شكرا" in result

    def test_with_hashtag(self):
        result = clean_text("#رائع هذا المنتج")
        assert "#" not in result

    def test_with_numbers(self):
        result = clean_text("المنتج 10/10 ممتاز")
        assert "ممتاز" in result

    def test_mixed_arabic_english(self):
        result = clean_text("منتج جيد product quality 100/100")
        assert "منتج" in result or "منتج" in result
        assert "quality" in result or "product" in result

    def test_long_text(self):
        text = "هذا المنتج " * 200
        result = clean_text(text)
        assert len(result) > 0


# ============================================================
# 9) الدالة الرئيسية — الدارجة الجزائرية (الأهم)
# ============================================================
class TestAlgerianDialect:
    """كلمات جزائرية حاسمة يجب ألا تُمسّ."""

    @pytest.mark.parametrize("word", [
        "بصح", "ماشي", "واش", "راه", "راك", "كاش", "شوية",
        "بزاف", "هايل", "هايلة", "زوالي", "طاكسي", "صباط",
        "كوزينة", "بيتزا", "طوموبيل", "قهوة", "خبز",
    ])
    def test_dialect_words_survive(self, word, tmp_path):
        """كل كلمة دارجة يجب أن تنجو من التنظيف."""
        result = clean_text(word)
        log = write_debug(tmp_path, f"word_{len(word)}",
                          input=word, output=result)
        assert len(result) >= 2, (
            f"FAIL: word disappeared. "
            f"in_len={len(word)}, out_len={len(result)}. See {log}"
        )

    def test_no_stemming_hayla(self, tmp_path):
        """'هايلة' يجب ألا تُجذّر إلى 'هال' أو ما شابه."""
        result = clean_text("هايلة")
        log = write_debug(tmp_path, "hayla", input="هايلة", output=result)
        assert result == "هايله", (
            f"FAIL: unexpected output. "
            f"out_len={len(result)}. See {log}"
        )

    def test_besh_survives(self, tmp_path):
        result = clean_text("راه مليح بصح غالي")
        log = write_debug(tmp_path, "besh",
                          input="راه مليح بصح غالي", output=result)
        assert "بصح" in result, (
            f"FAIL: 'besh' missing. out_len={len(result)}. See {log}"
        )

    def test_negation_ma(self, tmp_path):
        result = clean_text("ما عجبنيش")
        log = write_debug(tmp_path, "negation_ma",
                          input="ما عجبنيش", output=result)
        assert "ما" in result, (
            f"FAIL: 'ma' missing. out_len={len(result)}. See {log}"
        )

    def test_negation_machi(self, tmp_path):
        result = clean_text("هذا ماشي مليح")
        log = write_debug(tmp_path, "negation_machi",
                          input="هذا ماشي مليح", output=result)
        assert "ماشي" in result, (
            f"FAIL: 'machi' missing. out_len={len(result)}. See {log}"
        )


# ============================================================
# 10) الدالة الرئيسية — حالات حساسة
# ============================================================
class TestCleanTextSensitive:
    def test_algerian_hayla_with_elongation(self, tmp_path):
        """'هاييييلة' → 'هايلة' (بلا تجذير)."""
        text = "هاييييلة بزاف هذا المنتج"
        result = clean_text(text)
        log = write_debug(tmp_path, "hayla_elong",
                          input=text, output=result)

        has_hayla = "هايلة" in result
        has_elongated = "هاييييلة" in result

        assert has_hayla, (
            f"FAIL: 'hayla' not found. "
            f"in_len={len(text)}, out_len={len(result)}. See {log}"
        )
        assert not has_elongated, (
            f"FAIL: elongated form still present. See {log}"
        )

    def test_mixed_sentiment(self, tmp_path):
        """نص مختلط: مدح + ذم — يجب أن تبقى كل الكلمات."""
        text = "المنتج ممتاز بصح الخدمة سيئة"
        result = clean_text(text)
        log = write_debug(tmp_path, "mixed",
                          input=text, output=result)

        has_mumtaz = "ممتاز" in result
        has_besh = "بصح" in result
        has_sayia = ("سيئة" in result) or ("سيئه" in result)

        assert has_mumtaz, f"FAIL: 'mumtaz' missing. See {log}"
        assert has_besh, f"FAIL: 'besh' missing. See {log}"
        assert has_sayia, f"FAIL: 'sayia' missing. See {log}"

    def test_insult_not_removed(self, tmp_path):
        """الكلمات البذيئة يجب أن تبقى (للتصنيف السلبي)."""
        text = "قحبة"
        result = clean_text(text)
        log = write_debug(tmp_path, "insult", input=text, output=result)

        assert len(result) >= 2, (
            f"FAIL: insult removed. out_len={len(result)}. See {log}"
        )
        assert "قح" in result, (
            f"FAIL: expected 'qh' root. See {log}"
        )

    def test_french_kept(self, tmp_path):
        """الدارجة الجزائرية مختلطة بالفرنسية."""
        text = "c'est magnifique wallah"
        result = clean_text(text)
        log = write_debug(tmp_path, "french", input=text, output=result)

        assert "magnifique" in result or "magnifique" in result.replace("'", ""), (
            f"FAIL: french word removed. See {log}"
        )

    def test_religious_positive(self, tmp_path):
        """'الحمد لله' نص ديني إيجابي."""
        text = "الحمد لله"
        result = clean_text(text)
        log = write_debug(tmp_path, "religious", input=text, output=result)

        assert "الحمد" in result, (
            f"FAIL: 'alhamd' missing. See {log}"
        )

    def test_all_together(self, tmp_path):
        """نص معقد: تطويل + تشكيل + إيموجي + رابط + فرنسي."""
        text = "هاااااايل 😍 شوف https://x.com @user c'est super!!!"
        result = clean_text(text)
        log = write_debug(tmp_path, "complex", input=text, output=result)

        assert "هايل" in result, f"FAIL: 'hayel' missing. See {log}"
        assert "https" not in result, f"FAIL: url not removed. See {log}"
        assert "@user" not in result, f"FAIL: mention not removed. See {log}"
        assert "!!!" not in result, f"FAIL: punctuation not collapsed. See {log}"

    def test_emoji_to_word(self, tmp_path):
        text = "الخدمة سيئة جداً 😡"
        result = clean_text(text)
        log = write_debug(tmp_path, "emoji_neg", input=text, output=result)

        assert ("غاضب" in result) or ("😡" not in result), (
            f"FAIL: emoji not processed. See {log}"
        )


# ============================================================
# 11) الأداء والثبات
# ============================================================
class TestPerformance:
    def test_long_text(self):
        text = "هايلة بزاف " * 1000
        result = clean_text(text)
        assert len(result) > 0

    def test_unicode_mixed(self):
        text = "عربي English 123 ١٢٣ 😀🎉"
        result = clean_text(text)
        assert len(result) > 0

    def test_idempotent(self):
        """تشغيل clean_text مرتين يعطي نفس النتيجة."""
        text = "هاييييلة بزاف هذا المنتج 😍"
        once = clean_text(text)
        twice = clean_text(once)
        assert once == twice, (
            f"FAIL: not idempotent. "
            f"once_len={len(once)}, twice_len={len(twice)}"
        )


# ============================================================
# 12) اختبارات المعالجة الكاملة (End-to-End)
# ============================================================
class TestEndToEnd:
    def test_full_pipeline_positive(self, tmp_path):
        text = "هذا المنتج ممتاز جداً 😍 https://x.com"
        result = clean_text(text)
        log = write_debug(tmp_path, "e2e_pos", input=text, output=result)

        assert "ممتاز" in result, f"See {log}"
        assert "https" not in result, f"See {log}"
        assert "😍" not in result, f"See {log}"

    def test_full_pipeline_negative(self, tmp_path):
        text = "الخدمة سيئة للغاية 😡 @support"
        result = clean_text(text)
        log = write_debug(tmp_path, "e2e_neg", input=text, output=result)

        assert "سيئ" in result, f"See {log}"
        assert "@support" not in result, f"See {log}"

    def test_full_pipeline_dialect(self, tmp_path):
        text = "هاييييلة بزاف! بصح غالية شوية 😅"
        result = clean_text(text)
        log = write_debug(tmp_path, "e2e_dz", input=text, output=result)

        assert "هايلة" in result, f"See {log}"
        assert "بزاف" in result, f"See {log}"
        assert "بصح" in result, f"See {log}" 
