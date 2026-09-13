"""
تدريب TF-IDF + Logistic Regression مُحسَّن
============================================
يجمع:
1. Feature Engineering
2. Grid Search
3. Random Search
4. Cross-Validation
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score, StratifiedKFold
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from scipy.stats import uniform, randint

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from preprocessing import preprocess_dataframe


# ============================================================
# الإعدادات
# ============================================================
RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed", "clean_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OPTIMIZED_MODEL = os.path.join(MODELS_DIR, "tfidf_logistic_optimized.pkl")
SEARCH_RESULTS = os.path.join(MODELS_DIR, "search_results.json")

TEXT_COL = "text"
LABEL_COL = "label"
RANDOM_STATE = 42


# ============================================================
# 1. تحميل البيانات
# ============================================================
def load_data(path: str = RAW_DATA) -> pd.DataFrame:
    for enc in ['utf-8-sig', 'utf-8', 'cp1256', 'latin-1']:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✅ تم تحميل {len(df)} صف")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("فشل قراءة الملف")


# ============================================================
# 2. بناء النموذج الأساسي
# ============================================================
def build_baseline_pipeline() -> Pipeline:
    """النموذج الأساسي (بدون تحسين)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 1),
            sublinear_tf=True,
            lowercase=False,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced',
        )),
    ])


# ============================================================
# 3. Grid Search
# ============================================================
def run_grid_search(df: pd.DataFrame,
                    cv: int = 3,
                    quick: bool = False) -> dict:
    """
    Grid Search على Hyperparameters.
    
    Args:
        df: البيانات
        cv: عدد الطيات
        quick: وضع سريع (شبكة صغيرة)
    """
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    # التحقق من إمكانية CV
    min_class = pd.Series(y).value_counts().min()
    cv = min(cv, min_class)
    if cv < 2:
        print("⚠️ بيانات قليلة — تخطي Grid Search")
        return {"best_params": {}, "best_score": 0, "best_model": None}

    pipeline = build_baseline_pipeline()

    if quick:
        param_grid = {
            "tfidf__max_features": [3000, 5000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__C": [0.5, 1.0],
        }
    else:
        param_grid = {
            "tfidf__max_features": [3000, 5000, 10000, 20000],
            "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "tfidf__min_df": [1, 2],
            "tfidf__max_df": [0.85, 0.95, 1.0],
            "clf__C": [0.1, 0.5, 1.0, 5.0, 10.0],
            "clf__solver": ["lbfgs", "liblinear"],
        }

    # عدد التركيبات
    total = 1
    for v in param_grid.values():
        total *= len(v)

    print(f"\n🔍 Grid Search: {total} تركيبة × {cv} طية = {total * cv} تدريب")
    print(f"   (قد يستغرق {total * cv * 0.05:.1f} ثانية تقريباً)")

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=1,       # ✅ لتفادي مشاكل Windows
        verbose=0,
        return_train_score=True,
    )

    start = time.time()
    grid.fit(X, y)
    elapsed = time.time() - start

    print(f"\n⏱️  استغرق: {elapsed:.2f} ثانية")
    print(f"🏆 أفضل F1: {grid.best_score_:.4f}")
    print(f"📌 أفضل Parameters:")
    for key, value in grid.best_params_.items():
        print(f"   {key}: {value}")

    return {
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "best_model": grid.best_estimator_,
        "cv_results": {
            "mean_test_score": grid.cv_results_["mean_test_score"].tolist(),
            "std_test_score": grid.cv_results_["std_test_score"].tolist(),
            "params": [str(p) for p in grid.cv_results_["params"]],
        },
        "elapsed_sec": elapsed,
    }


# ============================================================
# 4. Random Search (أسرع)
# ============================================================
def run_random_search(df: pd.DataFrame,
                     n_iter: int = 30,
                     cv: int = 3) -> dict:
    """
    Random Search — أسرع من Grid Search للفضاءات الكبيرة.
    """
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    min_class = pd.Series(y).value_counts().min()
    cv = min(cv, min_class)
    if cv < 2:
        print("⚠️ بيانات قليلة — تخطي Random Search")
        return {"best_params": {}, "best_score": 0, "best_model": None}

    pipeline = build_baseline_pipeline()

    param_dist = {
        "tfidf__max_features": randint(1000, 20000),
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": randint(1, 4),
        "tfidf__max_df": uniform(0.7, 0.3),
        "clf__C": uniform(0.01, 10),
        "clf__solver": ["lbfgs", "liblinear"],
    }

    print(f"\n🎲 Random Search: {n_iter} محاولة × {cv} طية")

    random_search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=1,
        verbose=0,
        random_state=RANDOM_STATE,
    )

    start = time.time()
    random_search.fit(X, y)
    elapsed = time.time() - start

    print(f"⏱️  استغرق: {elapsed:.2f} ثانية")
    print(f"🏆 أفضل F1: {random_search.best_score_:.4f}")
    print(f"📌 أفضل Parameters:")
    for key, value in random_search.best_params_.items():
        print(f"   {key}: {value}")

    return {
        "best_params": random_search.best_params_,
        "best_score": float(random_search.best_score_),
        "best_model": random_search.best_estimator_,
        "elapsed_sec": elapsed,
    }


# ============================================================
# 5. مقارنة النماذج
# ============================================================
def compare_models(df: pd.DataFrame,
                   test_size: float = 0.2) -> pd.DataFrame:
    """مقارنة عدة إعدادات."""
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    configs = [
        {
            "name": "Baseline (1-gram, 5K)",
            "params": {
                "max_features": 5000,
                "ngram_range": (1, 1),
                "min_df": 1,
                "max_df": 0.95,
                "C": 1.0,
            }
        },
        {
            "name": "Bi-grams (10K)",
            "params": {
                "max_features": 10000,
                "ngram_range": (1, 2),
                "min_df": 1,
                "max_df": 0.95,
                "C": 1.0,
            }
        },
        {
            "name": "Tri-grams (20K)",
            "params": {
                "max_features": 20000,
                "ngram_range": (1, 3),
                "min_df": 2,
                "max_df": 0.85,
                "C": 0.5,
            }
        },
        {
            "name": "Strong Reg (C=0.1)",
            "params": {
                "max_features": 5000,
                "ngram_range": (1, 2),
                "min_df": 1,
                "max_df": 0.95,
                "C": 0.1,
            }
        },
        {
            "name": "Weak Reg (C=10)",
            "params": {
                "max_features": 5000,
                "ngram_range": (1, 2),
                "min_df": 1,
                "max_df": 0.95,
                "C": 10.0,
            }
        },
    ]

    results = []
    for config in configs:
        p = config["params"]
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=p["max_features"],
                ngram_range=p["ngram_range"],
                min_df=p["min_df"],
                max_df=p["max_df"],
                sublinear_tf=True,
                lowercase=False,
            )),
            ("clf", LogisticRegression(
                C=p["C"],
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
            )),
        ])

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)

        results.append({
            "name": config["name"],
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "train_time": train_time,
            "n_features": len(model.named_steps['tfidf'].vocabulary_),
        })

    # DataFrame
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    return results_df


# ============================================================
# 6. عرض جدول المقارنة
# ============================================================
def print_comparison_table(df: pd.DataFrame):
    """طباعة جدول مقارنة احترافي."""
    print("\n" + "=" * 90)
    print("📊 جدول مقارنة الإعدادات")
    print("=" * 90)
    print(f"{'الإعداد':<25} {'Accuracy':>10} {'F1':>10} {'ميزات':>10} {'وقت':>10}")
    print("-" * 90)

    for _, row in df.iterrows():
        print(f"{row['name']:<25} {row['accuracy']:>10.4f} "
              f"{row['f1']:>10.4f} {row['n_features']:>10,} "
              f"{row['train_time']:>9.3f}s")

    print("=" * 90)

    best = df.iloc[0]
    print(f"\n🏆 أفضل إعداد: {best['name']}")
    print(f"   F1-Score: {best['f1']:.4f}")


# ============================================================
# 7. الرئيسية
# ============================================================
def main():
    print("=" * 90)
    print("🚀 تحسين TF-IDF + Logistic Regression")
    print("=" * 90)

    # 1. تحميل البيانات
    df = load_data()

    # 2. معالجة
    print("\n🧹 معالجة النصوص...")
    df_clean = preprocess_dataframe(
        df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        mode="classic",
        output_path=PROCESSED_DATA,
    )

    if len(df_clean) < 20:
        print(f"\n⚠️ بيانات قليلة ({len(df_clean)} صف). "
              "التحسين سيكون محدوداً.")
        print("💡 حمّل بيانات أكثر للحصول على نتائج أفضل.")

    # 3. مقارنة الإعدادات
    print("\n📊 مقارنة الإعدادات...")
    comparison = compare_models(df_clean)
    print_comparison_table(comparison)

    # 4. Grid Search (فقط إذا البيانات كافية)
    if len(df_clean) >= 30:
        print("\n" + "=" * 90)
        print("🔍 Grid Search")
        print("=" * 90)
        quick = len(df_clean) < 100
        grid_results = run_grid_search(df_clean, cv=3, quick=quick)

        # 5. Random Search
        print("\n" + "=" * 90)
        print("🎲 Random Search")
        print("=" * 90)
        random_results = run_random_search(df_clean, n_iter=20, cv=3)

        # 6. اختيار الأفضل
        candidates = [
            ("Grid Search", grid_results),
            ("Random Search", random_results),
        ]
        candidates = [(n, r) for n, r in candidates if r.get("best_model")]

        if candidates:
            best_name, best_result = max(candidates, key=lambda x: x[1]["best_score"])
            print(f"\n🏆 أفضل بحث: {best_name} (F1={best_result['best_score']:.4f})")

            # حفظ النموذج الأفضل
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(best_result["best_model"], OPTIMIZED_MODEL)
            print(f"💾 تم حفظ النموذج: {OPTIMIZED_MODEL}")

            # حفظ النتائج
            report = {
                "best_search": best_name,
                "best_score": best_result["best_score"],
                "best_params": {
                    k: str(v) if not isinstance(v, (int, float, str, tuple))
                    else v
                    for k, v in best_result["best_params"].items()
                },
                "comparison": comparison.to_dict(orient="records"),
            }
            with open(SEARCH_RESULTS, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"📄 تم حفظ التقرير: {SEARCH_RESULTS}")

    print("\n✅ اكتمل التحسين!")


if __name__ == "__main__":
    main()