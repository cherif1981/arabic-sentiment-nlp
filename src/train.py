"""
تدريب ومقارنة نماذج Baseline لتحليل المشاعر العربية
=====================================================
يدرّب 3 نماذج ويقارنها تلقائياً:
    1. TF-IDF + Multinomial Naive Bayes
    2. TF-IDF + Logistic Regression
    3. TF-IDF + Linear SVM
ثم يحفظ أفضل نموذج في models/best_model.pkl
"""
import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, confusion_matrix)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from preprocessing import preprocess_dataframe


# ---------------- الإعدادات ----------------
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clean_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_PATH = os.path.join(MODELS_DIR, "model_comparison.json")

TEXT_COL = "text"
LABEL_COL = "label"

# ⚠️ يجب أن يطابق predict.py و app.py
PREPROCESS_MODE = "classic"


# ============================================================
# 1. تحميل البيانات
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    """تحميل البيانات من CSV مع دعم ترميزات متعددة."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ لم يتم العثور على الملف: {path}\n"
            f"ضع ملف dataset.csv في مجلد data/raw/"
        )

    for enc in ['utf-8-sig', 'utf-8', 'cp1256', 'latin-1']:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✅ تم تحميل {len(df)} صف بترميز {enc}")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError("❌ فشل في قراءة الملف بأي ترميز")


# ============================================================
# 2. بناء النماذج الثلاثة
# ============================================================
def build_models() -> dict:
    """
    بناء النماذج الثلاثة.
    
    Returns:
        dict: {اسم_النموذج: Pipeline}
    """
    # نفس المُميِّز لكل النماذج لضمان مقارنة عادلة
    def make_vectorizer():
        return TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents=None
        )

    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", MultinomialNB(alpha=1.0))
        ]),

        "Logistic Regression": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                n_jobs=None
            ))
        ]),

        "Linear SVM": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42
            ))
        ]),
    }

    return models


# ============================================================
# 3. تدريب وتقييم نموذج واحد
# ============================================================
def evaluate_single_model(name: str,
                          pipeline: Pipeline,
                          X_train, X_test,
                          y_train, y_test) -> dict:
    """تدريب وتقييم نموذج واحد وإرجاع النتائج."""
    print(f"\n{'─' * 60}")
    print(f"🎓 تدريب: {name}")
    print(f"{'─' * 60}")

    # التدريب
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    # التنبؤ
    y_pred = pipeline.predict(X_test)

    # المقاييس
    metrics = {
        "model": name,
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        "f1_macro":    float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        "train_time_sec": round(train_time, 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # عرض النتائج
    print(f"   ⏱️  زمن التدريب:  {train_time:.3f} ثانية")
    print(f"   🎯 الدقة:        {metrics['accuracy']:.4f}")
    print(f"   📊 F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"   📊 F1 (macro):    {metrics['f1_macro']:.4f}")

    # حفظ مصفوفة الالتباس في السجل
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classes"] = [str(c) for c in pipeline.classes_]

    return {
        "metrics": metrics,
        "pipeline": pipeline,
        "y_pred": y_pred,
    }


# ============================================================
# 4. مقارنة جميع النماذج
# ============================================================
def train_all_models(df: pd.DataFrame,
                     test_size: float = 0.2) -> dict:
    """تدريب جميع النماذج ومقارنتها."""
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    # stratify شرطي
    use_stratify = None
    class_counts = pd.Series(y).value_counts()
    if len(df) >= 30 and class_counts.min() >= 2:
        use_stratify = y
        print("📌 استخدام stratify")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=use_stratify
    )

    print(f"\n📊 حجم التدريب: {len(X_train)} | حجم الاختبار: {len(X_test)}")
    print(f"📌 توزيع فئات التدريب: {dict(pd.Series(y_train).value_counts())}")

    # تدريب كل النماذج
    models = build_models()
    results = {}

    for name, pipeline in models.items():
        result = evaluate_single_model(
            name, pipeline,
            X_train, X_test, y_train, y_test
        )
        results[name] = result

    return {
        "results": results,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


# ============================================================
# 5. عرض جدول المقارنة
# ============================================================
def print_comparison_table(results: dict):
    """طباعة جدول مقارنة احترافي."""
    print(f"\n\n{'═' * 78}")
    print("📊 جدول مقارنة النماذج")
    print(f"{'═' * 78}")
    print(f"{'النموذج':<22} {'Accuracy':>10} {'F1-weighted':>13} "
          f"{'F1-macro':>11} {'الزمن (ث)':>11}")
    print(f"{'─' * 78}")

    # ترتيب حسب F1-weighted تنازلياً
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"]["f1_weighted"],
        reverse=True
    )

    for name, res in sorted_results:
        m = res["metrics"]
        print(f"{name:<22} {m['accuracy']:>10.4f} "
              f"{m['f1_weighted']:>13.4f} "
              f"{m['f1_macro']:>11.4f} "
              f"{m['train_time_sec']:>11.3f}")

    print(f"{'═' * 78}\n")

    # أفضل نموذج
    best_name, best_res = sorted_results[0]
    print(f"🏆 أفضل نموذج: {best_name}")
    print(f"   الدقة: {best_res['metrics']['accuracy']:.4f}")
    print(f"   F1-weighted: {best_res['metrics']['f1_weighted']:.4f}\n")

    return best_name, best_res


# ============================================================
# 6. حفظ النتائج
# ============================================================
def save_results(results: dict, best_name: str):
    """حفظ النموذج الأفضل + تقرير JSON."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. حفظ أفضل نموذج
    best_pipeline = results[best_name]["pipeline"]
    best_model_path = os.path.join(MODELS_DIR, "sentiment_model.pkl")
    joblib.dump(best_pipeline, best_model_path)
    print(f"💾 حفظ أفضل نموذج ({best_name}) → {best_model_path}")

    # 2. حفظ جميع النماذج منفصلة (اختياري لكن مفيد)
    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(MODELS_DIR, f"model_{safe_name}.pkl")
        joblib.dump(res["pipeline"], path)
    print(f"💾 حفظ جميع النماذج في {MODELS_DIR}/")

    # 3. حفظ تقرير JSON
    report = {
        "preprocessing_mode": PREPROCESS_MODE,
        "best_model": best_name,
        "results": {
            name: res["metrics"]
            for name, res in results.items()
        }
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"📄 حفظ تقرير المقارنة → {RESULTS_PATH}")


# ============================================================
# 7. الدالة الرئيسية
# ============================================================
def main():
    print("=" * 78)
    print("🚀 تدريب ومقارنة نماذج Baseline — تحليل المشاعر العربية")
    print("=" * 78)

    # 1. تحميل البيانات
    df = load_data(RAW_DATA_PATH)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"❌ يجب أن يحتوي الملف على أعمدة: '{TEXT_COL}', '{LABEL_COL}'\n"
            f"الأعمدة الموجودة: {list(df.columns)}"
        )

    print(f"\n📌 توزيع الفئات:")
    print(df[LABEL_COL].value_counts().to_string())

    # 2. تنظيف البيانات
    print(f"\n🧹 تنظيف النصوص (mode={PREPROCESS_MODE})...")
    df_clean = preprocess_dataframe(
        df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        mode=PREPROCESS_MODE,
        output_path=PROCESSED_DATA_PATH
    )
    print(f"📊 بعد التنظيف: {len(df_clean)} صف")

    # 3. تدريب جميع النماذج
    training_output = train_all_models(df_clean)
    results = training_output["results"]

    # 4. عرض جدول المقارنة
    best_name, best_res = print_comparison_table(results)

    # 5. عرض التقرير المفصل لأفضل نموذج
    print(f"{'─' * 60}")
    print(f"📋 تقرير التصنيف المفصل لأفضل نموذج ({best_name})")
    print(f"{'─' * 60}")
    y_test = training_output["y_test"]
    y_pred = best_res["y_pred"]
    print(classification_report(y_test, y_pred, zero_division=0))

    print("📊 مصفوفة الالتباس:")
    print(confusion_matrix(y_test, y_pred))
    print(f"📌 الفئات: {list(best_res['pipeline'].classes_)}")

    # 6. حفظ النتائج
    save_results(results, best_name)

    # 7. تحذير عند دقة منخفضة
    if best_res["metrics"]["accuracy"] < 0.6:
        print(f"\n{'⚠️ ' * 25}")
        print("⚠️  الدقة منخفضة! الأسباب المحتملة:")
        print("   1. البيانات قليلة جداً (< 500 مثال)")
        print("   2. الفئات غير متوازنة")
        print("   3. النصوص قصيرة جداً")
        print("   💡 الحل: حمّل مجموعة بيانات أكبر من Kaggle/GitHub")
        print(f"{'⚠️ ' * 25}")

    print("\n" + "=" * 78)
    print("✅ اكتمل التدريب والمقارنة بنجاح!")
    print("=" * 78)


if __name__ == "__main__":
    main()