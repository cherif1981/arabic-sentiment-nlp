"""
اختبارات FastAPI
=================
يستخدم TestClient من fastapi.testclient
"""
import pytest
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# تخطي إذا لم يكن fastapi مثبتاً
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient."""
    from api.main import app
    return TestClient(app)


# ============================================================
# 1. Root & Health
# ============================================================
class TestRootEndpoints:
    """اختبارات النقاط الأساسية."""

    def test_root(self, client):
        """الصفحة الرئيسية."""
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health(self, client):
        """فحص الحالة."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ["ok", "degraded"]
        assert "version" in data
        assert "model_loaded" in data

    def test_docs_available(self, client):
        """Swagger UI متاح."""
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_schema(self, client):
        """OpenAPI schema متاح."""
        r = client.get("/openapi.json")
        assert r.status_code == 200
        data = r.json()
        assert "paths" in data


# ============================================================
# 2. /predict
# ============================================================
class TestPredictEndpoint:
    """اختبارات نقطة التنبؤ."""

    @pytest.mark.requires_model
    def test_predict_valid(self, client):
        """طلب صحيح."""
        r = client.post("/predict", json={"text": "هذا المنتج ممتاز"})
        assert r.status_code == 200
        data = r.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert "sentiment_ar" in data

    @pytest.mark.requires_model
    def test_predict_empty_text(self, client):
        """نص فارغ."""
        r = client.post("/predict", json={"text": ""})
        # إما 400 أو 422
        assert r.status_code in [400, 422]

    @pytest.mark.requires_model
    def test_predict_missing_text(self, client):
        """حقل text مفقود."""
        r = client.post("/predict", json={})
        assert r.status_code == 422  # Validation error

    @pytest.mark.requires_model
    def test_predict_whitespace(self, client):
        """مسافات فقط."""
        r = client.post("/predict", json={"text": "     "})
        assert r.status_code == 400

    @pytest.mark.requires_model
    def test_predict_too_long(self, client):
        """نص يتجاوز الحد."""
        long_text = "أ" * 6000
        r = client.post("/predict", json={"text": long_text})
        assert r.status_code == 422  # Validation error

    @pytest.mark.requires_model
    def test_predict_response_structure(self, client):
        """صيغة الرد صحيحة."""
        r = client.post("/predict", json={"text": "منتج رائع"})
        assert r.status_code == 200
        data = r.json()

        # الحقول المطلوبة
        required = ["text", "cleaned_text", "sentiment",
                    "sentiment_ar", "confidence", "model_name"]
        for field in required:
            assert field in data, f"حقل مفقود: {field}"

        # صحة القيم
        assert data["sentiment"] in ["negative", "neutral", "positive"]
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["probabilities"], list)

    @pytest.mark.requires_model
    def test_predict_with_mode(self, client):
        """الوضع light و classic."""
        for mode in ["classic", "light"]:
            r = client.post("/predict", json={
                "text": "منتج جيد",
                "mode": mode
            })
            assert r.status_code == 200


# ============================================================
# 3. /predict/batch
# ============================================================
class TestBatchEndpoint:
    """اختبارات الدفعات."""

    @pytest.mark.requires_model
    def test_batch_valid(self, client):
        """دفعة صحيحة."""
        r = client.post("/predict/batch", json={
            "texts": ["رائع", "سيء", "عادي"]
        })
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 3
        assert len(data["results"]) == 3

    @pytest.mark.requires_model
    def test_batch_empty_list(self, client):
        """قائمة فارغة."""
        r = client.post("/predict/batch", json={"texts": []})
        assert r.status_code == 422  # Validation error

    @pytest.mark.requires_model
    def test_batch_too_many(self, client):
        """قائمة كبيرة جداً."""
        texts = ["نص"] * 200
        r = client.post("/predict/batch", json={"texts": texts})
        assert r.status_code == 422

    @pytest.mark.requires_model
    def test_batch_response_structure(self, client):
        """صيغة رد الدفعة."""
        r = client.post("/predict/batch", json={"texts": ["رائع", "سيء"]})
        data = r.json()
        assert "results" in data
        assert "count" in data
        assert "processing_time_ms" in data
        assert data["count"] == 2


# ============================================================
# 4. Error Handling
# ============================================================
class TestErrorHandling:
    """اختبارات معالجة الأخطاء."""

    def test_invalid_json(self, client):
        """JSON غير صحيح."""
        r = client.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 422

    def test_wrong_content_type(self, client):
        """Content-Type خاطئ."""
        r = client.post(
            "/predict",
            data="text=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        # FastAPI يرفضه بـ 422
        assert r.status_code == 422

    def test_nonexistent_endpoint(self, client):
        """نقطة غير موجودة."""
        r = client.get("/nonexistent")
        assert r.status_code == 404