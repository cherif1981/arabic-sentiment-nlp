"""
Feature Engineering — ميزات إضافية للنصوص العربية
====================================================
يستخرج ميزات لغوية وإحصائية لتقوية النموذج.
"""
import re
import numpy as np
import pandas as pd


# ============================================================
# قواميس المشاعر (كلمات مفتاحية قوية)
# ============================================================
POSITIVE_WORDS = {
    "ممتاز", "رائع", "جميل", "جيد", "مذهل", "fantastic",
    "أحببت", "احبه", "سعيد", "سعيدة", "فرحان", "فرح",
    "أنصح", "انصح", "بالتأكيد", "مضمون", "ممتازة",
    "جودة", "سريع", "سريعة", "احترافي", "راقي",
    "شكرا", "شكراً", "أفضل", "افضل", "ينصح",
    "مبدع", "متقن", "لذيذ", "رائعة", "مثالي",
    "افضل", "أحسن", "احسن", "الأفضل", "الافضل",
}

NEGATIVE_WORDS = {
    "سيء", "سيئة", "سيئه", "رديء", "رديئة",
    "سيئ", "فظيع", "مروع", "مزعج", "مزعجة",
    "أكره", "اكره", "كرهت", "غاضب", "زعلان",
    "لا أنصح", "لا انصح", "احذر", "تحذير",
    "تالف", "تالفة", "مكسور", "مخرب",
    "خدعة", "احتيال", "نصب", "غش",
    "بطيء", "بطيئة", "متأخر", "متأخرة",
    "سيء جدا", "سيئه جدا", "فاشل", "فاشلة",
    "مضيعة", "خسارة", "ندمت", "نادم",
}

INTENSIFIERS = {
    "جدا", "جداً", "كثيرا", "كثيراً", "للغاية", "لدرجه",
    "أوي", "اوي", "خالص", "تماما", "تماماً",
    "بشدة", "بشده", "لحد", "لحد", "بمره",
}

NEGATIONS = {
    "لا", "ما", "ليس", "ليست", "مش", "مو",
    "مافي", "مافيه", "بدون", "غير", "لا يوجد",
}


# ============================================================
# 1. ميزات النص الأساسية
# ============================================================
def extract_basic_features(text: str) -> dict:
    """ميزات أساسية (طول، كلمات، إلخ)."""
    if not isinstance(text, str):
        text = ""

    words = text.split()
    chars = len(text)
    word_count = len(words)

    return {
        "char_count": chars,
        "word_count": word_count,
        "avg_word_length": chars / max(word_count, 1),
        "unique_word_ratio": len(set(words)) / max(word_count, 1),
    }


# ============================================================
# 2. ميزات المشاعر
# ============================================================
def extract_sentiment_features(text: str) -> dict:
    """ميزات مبنية على قواميس المشاعر."""
    words = set(text.split()) if isinstance(text, str) else set()

    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    intensifier_count = len(words & INTENSIFIERS)
    negation_count = len(words & NEGATIONS)

    total_sentiment = pos_count + neg_count

    return {
        "positive_word_count": pos_count,
        "negative_word_count": neg_count,
        "intensifier_count": intensifier_count,
        "negation_count": negation_count,
        "sentiment_balance": (pos_count - neg_count) / max(total_sentiment, 1),
        "has_positive": int(pos_count > 0),
        "has_negative": int(neg_count > 0),
        "has_intensifier": int(intensifier_count > 0),
        "has_negation": int(negation_count > 0),
    }


# ============================================================
# 3. ميزات علامات الترقيم والرموز
# ============================================================
def extract_punctuation_features(text: str) -> dict:
    """ميزات علامات الترقيم والرموز."""
    if not isinstance(text, str):
        text = ""

    # علامات التعجب والاستفهام
    exclamations = text.count("!") + text.count("!")
    questions = text.count("?") + text.count("؟")
    ellipsis = text.count("...") + text.count("…")

    # الإيموجي في النص الأصلي (قبل المعالجة)
    emoji_count = len(re.findall(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]',
        text
    ))

    # الأحرف المتكررة (مثل: راااائع)
    repeated = len(re.findall(r'(.)\1{2,}', text))

    return {
        "exclamation_count": exclamations,
        "question_count": questions,
        "ellipsis_count": ellipsis,
        "emoji_count": emoji_count,
        "repeated_chars_count": repeated,
        "is_all_caps_arabic": 0,  # لا ينطبق على العربية
    }


# ============================================================
# 4. ميزات إجمالية
# ============================================================
def extract_all_features(text: str) -> dict:
    """استخراج جميع الميزات."""
    features = {}
    features.update(extract_basic_features(text))
    features.update(extract_sentiment_features(text))
    features.update(extract_punctuation_features(text))
    return features


def extract_features_dataframe(df: pd.DataFrame,
                                text_col: str = "text") -> pd.DataFrame:
    """
    استخراج الميزات لكل صف في DataFrame.
    
    Returns:
        DataFrame مع الميزات الجديدة
    """
    features_list = df[text_col].apply(extract_all_features).tolist()
    features_df = pd.DataFrame(features_list)
    return pd.concat([df.reset_index(drop=True), features_df], axis=1)


# ============================================================
# 5. اختبار سريع
# ============================================================
if __name__ == "__main__":
    samples = [
        "هذا المنتج ممتاز جداً وأنا سعيد بشرائه!!! 😍",
        "الخدمة سيئة للغاية، لن أشتري مرة أخرى 😡😡",
        "المنتج عادي، لا شيء مميز",
        "رااااائع جدا جدا جدا، أنصح الجميع!!!",
        "لا أنصح به أبداً، خسارة للمال والوقت",
    ]

    for text in samples:
        print(f"\n📝 النص: {text}")
        features = extract_all_features(text)
        for key, value in features.items():
            print(f"   {key:25} = {value}")