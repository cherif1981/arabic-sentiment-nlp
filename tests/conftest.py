"""
Pytest Fixtures — إعدادات مشتركة لكل الاختبارات
"""
import os
import sys
import pytest

# إضافة المسارات
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Marker مخصص
# ============================================================
def pytest_configure(config):
    """تسجيل markers مخصصة."""
    config.addinivalue_line("markers", "slow: اختبارات بطيئة")
    config.addinivalue_line("markers", "requires_model: تحتاج نموذجاً مدرباً")
    config.addinivalue_line("markers", "requires_api: تحتاج API يعمل")


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="session")
def project_root():
    """مسار جذر المشروع."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def src_dir():
    """مسار مجلد src."""
    return SRC_DIR


@pytest.fixture(scope="session")
def model_path(project_root):
    """مسار النموذج المدرب."""
    return os.path.join(project_root, "models", "sentiment_model.pkl")


@pytest.fixture(scope="session")
def model_exists(model_path):
    """هل النموذج موجود؟"""
    return os.path.exists(model_path)


@pytest.fixture(scope="session")
def predictor(model_path, model_exists):
    """SentimentPredictor جاهز للاختبار."""
    if not model_exists:
        pytest.skip(f"النموذج غير موجود في {model_path}")

    from predict import SentimentPredictor
    return SentimentPredictor(model_path=model_path)


@pytest.fixture
def sample_texts():
    """نصوص عربية للاختبار."""
    return {
        "positive": "هذا المنتج ممتاز جداً وأنا سعيد بشرائه",
        "negative": "الخدمة سيئة للغاية ولن أكرر التجربة",
        "neutral":  "المنتج عادي، لا شيء مميز فيه",
        "short":    "جيد",
        "empty":    "",
        "spaces":   "     ",
        "emoji":    "رائع 😍🎉",
        "url":      "زيارة https://example.com للتفاصيل",
        "mention":  "@ahmed شكراً لك",
        "hashtag":  "#رائع هذا المنتج",
        "mixed":    "منتج جيد product quality 100/100",
        "numbers":  "المنتج 10/10 ممتاز",
        "long":     "هذا المنتج " * 200,
        "special":  "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        "tashkeel": "مَرْحَبًا بِكُمْ فِي مُنْتَدَانَا",
    }


# ============================================================
# Hooks
# ============================================================
def pytest_report_header(config):
    """رأس التقرير."""
    return [
        "=" * 60,
        "Arabic Sentiment NLP - Test Suite",
        f"Python: {sys.version.split()[0]}",
        f"Project: {PROJECT_ROOT}",
        "=" * 60,
    ]