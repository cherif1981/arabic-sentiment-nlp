"""
FastAPI Application — Arabic Sentiment Analyzer
================================================
التشغيل:
    uvicorn api.main:app --reload
"""
import os
import sys

# ✅ إصلاح UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ✅ الاستيرادات الصحيحة (لاحظ Request)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from api import __version__
from api.routers import predict, health
from api.dependencies import get_predictor


# ============================================================
# Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """تحميل النموذج عند بدء الخادم."""
    print("=" * 60)
    print(f"  Arabic Sentiment Analyzer API v{__version__}")
    print("=" * 60)
    print("\n[Startup] Loading model...")

    try:
        predictor = get_predictor()
        print(f"[Startup] [OK] Model loaded: {type(predictor.model).__name__}")
    except Exception as e:
        print(f"[Startup] [WARN] Model loading failed: {e}")
        print("[Startup] API will start but /predict will fail")

    print("[Startup] Ready to serve requests\n")
    yield
    print("\n[Shutdown] Closing API...")


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Arabic Sentiment Analyzer API",
    description=(
        "## تحليل مشاعر النصوص العربية\n\n"
        "API احترافي باستخدام **TF-IDF + Machine Learning**.\n\n"
        "### الفئات:\n"
        "- `negative` — سلبي\n"
        "- `neutral` — محايد\n"
        "- `positive` — إيجابي\n"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Exception Handlers — لاحظ Request كأول معامل
# ============================================================
@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """معالج خطأ النموذج المفقود."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "Model not available",
            "detail": str(exc),
            "status_code": 503
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """معالج خطأ القيم."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "detail": str(exc),
            "status_code": 400
        }
    )


# ============================================================
# Routers
# ============================================================
app.include_router(health.router)
app.include_router(predict.router)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )