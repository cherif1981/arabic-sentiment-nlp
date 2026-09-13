"""
FastAPI Application - Arabic Sentiment Analyzer
"""
import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__
from api.routers import health, predict, models
from api.model_manager import get_model_manager, AVAILABLE_MODELS, DEFAULT_MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    """بدء وإيقاف التطبيق."""
    print("=" * 60)
    print("Arabic Sentiment Analyzer API v" + __version__)
    print("=" * 60)

    manager = get_model_manager()
    print("[Startup] Loading default model: " + DEFAULT_MODEL)

    try:
        loaded = manager.load_model(DEFAULT_MODEL)
        print("[Startup] OK: " + loaded['full_name'])
    except Exception as e:
        print("[Startup] WARN: " + str(e))

    print("[Startup] Docs: http://localhost:8000/docs")
    print("[Startup] Ready")
    yield
    print("[Shutdown] Stopping...")


app = FastAPI(
    title="Arabic Sentiment Analyzer API",
    description="نظام تحليل مشاعر النصوص العربية - Multi-Model API",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=503,
        content={"error": "Model not available", "detail": str(exc)},
    )


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)


@app.get("/", tags=["Root"])
async def root():
    manager = get_model_manager()
    return {
        "name": "Arabic Sentiment Analyzer API",
        "version": __version__,
        "default_model": DEFAULT_MODEL,
        "loaded_models": manager.get_loaded_models(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "models": "/models",
            "predict": "POST /predict",
            "predict_model": "POST /predict/{model_name}",
            "batch": "POST /predict/batch",
        },
        "labels": ["negative", "neutral", "positive"],
        "available_models": list(AVAILABLE_MODELS.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)