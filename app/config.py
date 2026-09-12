"""
إعدادات التطبيق
"""
import os

# ---------------- المسارات ----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARABERT_DIR = os.path.join(MODELS_DIR, "arabert")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# ---------------- إعدادات الصفحة ----------------
PAGE_TITLE = "Arabic Sentiment Analyzer 🇩🇿"
PAGE_ICON = "🎭"
LAYOUT = "wide"

# ---------------- خريطة التصنيفات ----------------
LABEL_MAP = {
    "negative": "سلبي 😞",
    "positive": "إيجابي 😊",
    "neutral": "محايد 😐",
    0: "سلبي 😞",
    1: "محايد 😐",
    2: "إيجابي 😊",
}

LABEL_COLORS = {
    "negative": "#e74c3c",
    "positive": "#2ecc71",
    "neutral": "#95a5a6",
    "سلبي 😞": "#e74c3c",
    "إيجابي 😊": "#2ecc71",
    "محايد 😐": "#95a5a6",
}

# ---------------- أمثلة جاهزة ----------------
EXAMPLES = {
    "إيجابي": [
        "هذا المنتج ممتاز جداً وأنا سعيد بشرائه",
        "خدمة راقية وسريعة، أنصح الجميع بالتعامل معهم",
        "جودة عالية وسعر منافس، تجربة رائعة",
    ],
    "سلبي": [
        "الخدمة سيئة جداً ولا أنصح بها أبداً",
        "المنتج تالف ولم يستجب البائع لشكواي",
        "تجربة سيئة، لن أكرر الشراء مرة أخرى",
    ],
    "محايد": [
        "المنتج عادي، لا شيء مميز فيه",
        "لا بأس به لكن ليس الأفضل في فئته",
        "جيد نسبياً، لكن السعر مرتفع قليلاً",
    ],
}

# ---------------- النماذج المتاحة ----------------
MODELS_INFO = {
    "AraBERT": {
        "path": ARABERT_DIR,
        "desc": "🏆 نموذج Transformer حديث — الأدق (85%+)",
        "type": "arabert",
        "color": "#e74c3c",
    },
    "Linear SVM": {
        "path": os.path.join(MODELS_DIR, "model_linear_svm.pkl"),
        "desc": "⚡ سريع ودقيق نسبياً (~58%)",
        "type": "classic",
        "color": "#3498db",
    },
    "Logistic Regression": {
        "path": os.path.join(MODELS_DIR, "model_logistic_regression.pkl"),
        "desc": "📊 متوازن ويعطي احتمالات (~50%)",
        "type": "classic",
        "color": "#9b59b6",
    },
    "Naive Bayes": {
        "path": os.path.join(MODELS_DIR, "model_naive_bayes.pkl"),
        "desc": "🚀 الأسرع في التدريب (~42%)",
        "type": "classic",
        "color": "#f39c12",
    },
}