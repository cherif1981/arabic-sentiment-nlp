"""
TF-IDF + Logistic Regression — النموذج الكلاسيكي
====================================================
هذا الملف مخصص لتدريب وتقييم هذا النموذج تحديداً.
"""
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from preprocessing import preprocess_dataframe


# ============================================================
# الإعدادات
# ============================================================
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_logistic.pkl")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clean_data.csv")

TEXT_COL = "text"
LABEL_COL = "label"
RANDOM_STATE = 42


# ============================================================
# 1. بناء النموذج
# ============================================================
def build_tfidf_logistic(
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    min_df: int = 1,
    max_df: float = 0.95,
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
) -> Pipeline:
    """
    بناء نموذج TF-IDF + Logistic Regression.
    
    Args:
        max_features: أقصى عدد من الميزات (الكلمات)
        ngram_range: نطاق n-grams ((1,1) = unigrams, (1,2) = uni+bi)
        min_df: أقل تكرار للوثيقة (لتصفية الكلمات النادرة)
        max_df: أقصى تكرار للوثيقة (لتصفية الكلمات الشائعة جداً)
        C: معامل التنظيم (أصغر = تنظيم أقوى)
        solver: الخوارزمية ('lbfgs', 'liblinear', 'saga')
        max_iter: أقصى عدد تكرارات
    
    Returns:
        Pipeline جاهز للتدريب
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,       # log(TF) بدلاً من TF
        strip_accents=None,      # النص العربي منظّف مسبقاً
        lowercase=False,         # العربية لا تحتاج lowercase
        norm='l2',               # L2 normalization
    )

    classifier = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        n_jobs=None,             # ✅ لا نستخدم -1 لتجنب مشاكل Windows
        class_weight='balanced', # ✅ يتعامل مع الفئات غير المتوازنة
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


# ============================================================
# 2. تحميل البيانات
# ============================================================
def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """تحميل البيانات."""
    for enc in ['utf-8-sig', 'utf-8', 'cp1256', 'latin-1']:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✅ تم تحميل {len(df)} صف")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("فشل قراءة الملف")


# ============================================================
# 3. التدريب
# ============================================================
def train(df: pd.DataFrame, test_size: float = 0.2, **kwargs) -> dict:
    """تدريب النموذج وتقييمه."""
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    # stratify شرطي
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    print(f"\n📊 التدريب: {len(X_train)} | الاختبار: {len(X_test)}")

    # بناء وتدريب
    model = build_tfidf_logistic(**kwargs)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # التنبؤ
    y_pred = model.predict(X_test)

    # المقاييس
    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        "train_time_sec": round(train_time, 4),
        "n_features": len(model.named_steps['tfidf'].vocabulary_),
    }

    return {
        "model": model,
        "metrics": metrics,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
    }


# ============================================================
# 4. عرض النتائج
# ============================================================
def print_results(results: dict):
    """عرض النتائج بشكل جميل."""
    m = results["metrics"]

    print("\n" + "=" * 60)
    print("📊 نتائج TF-IDF + Logistic Regression")
    print("=" * 60)
    print(f"  🎯 Accuracy:   {m['accuracy']:.4f}")
    print(f"  🎯 Precision:  {m['precision']:.4f}")
    print(f"  🎯 Recall:     {m['recall']:.4f}")
    print(f"  🎯 F1-Score:   {m['f1']:.4f}")
    print(f"  ⏱️  زمن التدريب: {m['train_time_sec']:.4f} ثانية")
    print(f"  📚 عدد الميزات: {m['n_features']:,}")
    print("=" * 60)

    print("\n📋 تقرير التصنيف:")
    print(classification_report(
        results["y_test"], results["y_pred"],
        zero_division=0
    ))

    print("📊 مصفوفة الالتباس:")
    print(confusion_matrix(results["y_test"], results["y_pred"]))


# ============================================================
# 5. أهم الكلمات لكل فئة (التفسيرية)
# ============================================================
def show_top_features(results: dict, top_n: int = 15):
    """
    عرض أهم الكلمات لكل فئة.
    
    هذه ميزة قوية في Logistic Regression —
    يمكن معرفة الكلمات التي تُرجّح كل فئة.
    """
    model = results["model"]
    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['clf']

    feature_names = vectorizer.get_feature_names_out()
    classes = classifier.classes_

    print("\n" + "=" * 60)
    print("🔍 أهم الكلمات لكل فئة")
    print("=" * 60)

    for i, cls in enumerate(classes):
        coefs = classifier.coef_[i]

        # أعلى 15 كلمة إيجابية (ترجّح الفئة)
        top_positive_idx = np.argsort(coefs)[-top_n:][::-1]
        # أعلى 15 كلمة سلبية (تضعف الفئة)
        top_negative_idx = np.argsort(coefs)[:top_n]

        print(f"\n📌 الفئة: {cls}")
        print(f"{'─' * 60}")
        print("  ✅ كلمات ترجّح هذه الفئة:")
        for idx in top_positive_idx:
            print(f"     {feature_names[idx]:20} {coefs[idx]:+.4f}")

        print(f"\n  ❌ كلمات تُضعف هذه الفئة:")
        for idx in top_negative_idx:
            print(f"     {feature_names[idx]:20} {coefs[idx]:+.4f}")


# ============================================================
# 6. Cross-Validation
# ============================================================
def cross_validate(df: pd.DataFrame, cv: int = 5, **kwargs):
    """تحقق متقاطع بـ K-Fold."""
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    model = build_tfidf_logistic(**kwargs)

    # تحقق من أن كل فئة لها على الأقل cv عناصر
    min_class_count = pd.Series(y).value_counts().min()
    cv = min(cv, min_class_count)

    if cv < 2:
        print("⚠️ لا يمكن إجراء Cross-Validation (بيانات قليلة)")
        return

    print(f"\n🔄 Cross-Validation ({cv}-Fold)...")
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=1
    )

    print(f"  📊 F1 scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  🎯 المتوسط:   {scores.mean():.4f}")
    print(f"  📉 الانحراف:  {scores.std():.4f}")


# ============================================================
# 7. Grid Search (تحسين Hyperparameters)
# ============================================================
def grid_search(df: pd.DataFrame):
    """بحث في أفضل Hyperparameters."""
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    if len(df) < 30:
        print("⚠️ بيانات قليلة جداً لـ Grid Search")
        return

    print("\n🔍 Grid Search...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, lowercase=False)),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced',
        )),
    ])

    param_grid = {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=1,
        verbose=1,
    )

    grid.fit(X, y)

    print(f"\n🏆 أفضل Parameters: {grid.best_params_}")
    print(f"🎯 أفضل F1: {grid.best_score_:.4f}")

    return grid.best_estimator_


# ============================================================
# 8. الرئيسية
# ============================================================
def main():
    print("=" * 60)
    print("🎯 TF-IDF + Logistic Regression")
    print("=" * 60)

    # 1. تحميل البيانات
    df = load_data()

    # 2. معالجة
    print("\n🧹 معالجة النصوص...")
    df_clean = preprocess_dataframe(
        df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        mode="classic",
        output_path=PROCESSED_PATH,
    )

    # 3. تدريب
    print("\n🎓 التدريب...")
    results = train(df_clean)

    # 4. عرض النتائج
    print_results(results)

    # 5. أهم الكلمات
    show_top_features(results, top_n=10)

    # 6. Cross-Validation
    cross_validate(df_clean, cv=5)

    # 7. حفظ النموذج
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(results["model"], MODEL_PATH)
    print(f"\n💾 تم حفظ النموذج: {MODEL_PATH}")

    print("\n✅ اكتمل!")


if __name__ == "__main__":
    main()