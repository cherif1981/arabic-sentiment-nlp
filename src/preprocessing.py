"""
تنظيف النصوص العربية واللهجة الجزائرية
========================================
مبادئ التصميم:
- لا تجذير (No Stemming): يحفظ "هايلة"، "ماشي"، "واش"
- لا حذف كلمات وظيفية (No Stopwords): يحفظ "بصح"، "ما"، "راه"
- تنظيف أدنى فقط: تشكيل + تطويل + توحيد ألف
- يحتفظ بالكلمات الفرنسية (الدارجة الجزائرية مختلطة)
- يحتفظ بالإيموجي كإشارة مشاعر (اختياري)
"""
import re
import unicodedata

# ============================================================
# 1) التشكيل والتطويل
# ============================================================
DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")

def remove_diacritics(text: str) -> str:
    """إزالة التشكيل والتطويل."""
    return DIACRITICS.sub("", text)

# ============================================================
# 2) التكرار اللفظي التطويلي
# ============================================================
# "هايييييلة" → "هايلة"
ELONGATION = re.compile(r"(.)\1{2,}")

def remove_elongation(text: str) -> str:
    """تقليص الحرف المكرر 3+ مرات إلى واحد."""
    return ELONGATION.sub(r"\1", text)

# ============================================================
# 3) توحيد الأحرف العربية
# ============================================================
ALEF_NORMALIZATION = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
    "گ": "ك", "ک": "ك",
    "ڤ": "ف", "ڨ": "ق",  # حروف مستخدمة في الدارجة
    "ﭺ": "ش", "ﭖ": "ب",
}

def normalize_letters(text: str) -> str:
    """توحيد الأحرف العربية مع الحفاظ على الحروف اللهجية."""
    for src, dst in ALEF_NORMALIZATION.items():
        text = text.replace(src, dst)
    return text

# ============================================================
# 4) الأرقام العربية
# ============================================================
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_digits(text: str) -> str:
    return text.translate(ARABIC_DIGITS)

# ============================================================
# 5) الإيموجي (اختياري: تحويلها إلى كلمات)
# ============================================================
EMOJI_MAP = {
    "😀": " سعيد ", "😃": " سعيد ", "😄": " سعيد ", "😁": " سعيد ",
    "😊": " سعيد ", "🙂": " سعيد ", "😍": " حب ", "❤️": " حب ",
    "😢": " حزين ", "😭": " حزين ", "😞": " حزين ", "😔": " حزين ",
    "😡": " غاضب ", "😠": " غاضب ", "🤬": " غاضب ",
    "👍": " جيد ", "👎": " سيء ", "🔥": " رائع ", "💯": " رائع ",
}

EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U0001FA00-\U0001FAFF]+",
    flags=re.UNICODE,
)

def normalize_emoji(text: str, keep_as_words: bool = True) -> str:
    """تحويل الإيموجي إلى كلمات دالة على المشاعر."""
    if not keep_as_words:
        return EMOJI_RE.sub(" ", text)
    for emo, word in EMOJI_MAP.items():
        text = text.replace(emo, word)
    return EMOJI_RE.sub(" ", text)

# ============================================================
# 6) الروابط والإشارات
# ============================================================
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")

def remove_urls_mentions(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)   # نحتفظ بكلمة الهاشتاغ
    return text

# ============================================================
# 7) الترقيم المتكرر والمسافات
# ============================================================
PUNCT_REPEAT = re.compile(r"([!؟?.,])\1+")
MULTISPACE = re.compile(r"\s+")

def normalize_punctuation(text: str) -> str:
    text = PUNCT_REPEAT.sub(r"\1", text)
    return MULTISPACE.sub(" ", text).strip()

# ============================================================
# 8) الدالة الرئيسية
# ============================================================
def clean_text(
    text: str,
    keep_emoji: bool = True,
    keep_foreign: bool = True,
    min_len: int = 1,
) -> str:
    """
    تنظيف نص عربي/جزائري مع الحفاظ على الكلمات اللهجية.

    Args:
        text: النص الخام
        keep_emoji: تحويل الإيموجي إلى كلمات (True) أو حذفها (False)
        keep_foreign: الاحتفاظ بالكلمات الفرنسية/الإنجليزية (مهمة للدارجة)
        min_len: الحد الأدنى لطول النص الناتج

    Returns:
        النص المنظف
    """
    if not isinstance(text, str):
        return ""

    # 1. تطبيع Unicode (يحول الأحرف المركبة)
    text = unicodedata.normalize("NFKC", text)

    # 2. إزالة الروابط والإشارات
    text = remove_urls_mentions(text)

    # 3. الإيموجي
    text = normalize_emoji(text, keep_as_words=keep_emoji)

    # 4. إزالة التشكيل
    text = remove_diacritics(text)

    # 5. تقليص التطويل
    text = remove_elongation(text)

    # 6. توحيد الأحرف
    text = normalize_letters(text)

    # 7. الأرقام
    text = normalize_digits(text)

    # 8. إزالة الرموز غير المفيدة (مع الحفاظ على الترقيم الأساسي)
    if not keep_foreign:
        # احذف كل ما ليس عربيًا ولا رقمًا ولا ترقيمًا
        text = re.sub(r"[^\u0600-\u06FF0-9\s!؟?.,]", " ", text)
    else:
        # احذف الرموز الغريبة فقط، واحتفظ بالحروف اللاتينية
        text = re.sub(r"[^\u0600-\u06FF\u0600-\u06FFA-Za-z0-9\s!؟?.,]", " ", text)

    # 9. الترقيم والمسافات
    text = normalize_punctuation(text)

    # 10. التحقق من الطول
    if len(text) < min_len:
        return ""

    return text


# ============================================================
# 9) اختبار سريع
# ============================================================
if __name__ == "__main__":
    tests = [
        "هاييييلة بزاف هذا المنتج",
        "ماشي مشكل، بصح غالي شوية",
        "واش راك؟ الحمد لله",
        "قحبة",  # يجب أن تبقى كما هي
        "c'est magnifique wallah 😍",
        "الخدمة سيئة جداً 😡",
        "المنتج عادي!",
    ]
    print("=" * 70)
    for t in tests:
        print(f"قبل: {t}")
        print(f"بعد: {clean_text(t)}")
        print("-" * 70)