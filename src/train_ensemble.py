"""
Ensemble — دمج عدة نماذج لتحسين الدقة
========================================
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from preprocessing import preprocess_dataframe


RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")
ENSEMBLE_MODEL = os.path.join(PROJECT_ROOT, "models", "ensemble_model.pkl")

TEXT_COL = "text"
LABEL_COL = "label"
RANDOM_STATE = 42


def build_base_models():
    """بناء النماذج الأساسية."""
    def make_vectorizer():
        return TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=False,
        )

    return [
        ("lr", Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
            )),
        ])),
        ("svm", Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LinearSVC(
                C=1.0, max_iter=2000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
            )),
        ])),
        ("nb", Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", MultinomialNB(alpha=1.0)),
        ])),
    ]


def build_voting_ensemble(voting: str = "soft") -> VotingClassifier:
    """
    بناء Voting Ensemble.
    
    Args:
        voting: "soft" (يستخدم الاحتمالات) أو "hard" (تصويت)
    """
    return VotingClassifier(
        estimators=build_base_models(),
        voting=voting,
        n_jobs=1,
    )


def build_stacking_ensemble() -> StackingClassifier:
    """Stacking Ensemble — يستخدم meta-model."""
    return StackingClassifier(
        estimators=build_base_models(),
        final_estimator=LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        cv=3,
        n_jobs=1,
    )


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """تقييم نموذج."""
    y_pred = model.predict(X_test)

    return {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "y_pred": y_pred,
    }


def main():
    print("=" * 70)
    print("🎯 Ensemble Methods")
    print("=" * 70)

    # 1. تحميل البيانات
    df = pd.read_csv(RAW_DATA, encoding='utf-8-sig')

    print("\n🧹 معالجة...")
    df_clean = preprocess_dataframe(
        df, text_col=TEXT_COL, label_col=LABEL_COL, mode="classic"
    )

    X = df_clean[TEXT_COL].values
    y = df_clean[LABEL_COL].values

    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )

    print(f"📊 التدريب: {len(X_train)} | الاختبار: {len(X_test)}")

    # 2. النماذج الفردية
    print("\n" + "=" * 70)
    print("📊 النماذج الفردية")
    print("=" * 70)

    results = []

    for name, model in build_base_models():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")

    # 3. Voting (soft)
    print("\n" + "=" * 70)
    print("🗳️  Voting Ensemble (soft)")
    print("=" * 70)

    try:
        voting = build_voting_ensemble("soft")
        voting.fit(X_train, y_train)
        metrics = evaluate_model(voting, X_test, y_test, "Voting (soft)")
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
    except Exception as e:
        print(f"  ⚠️ فشل: {e}")
        voting = None

    # 4. Voting (hard)
    print("\n" + "=" * 70)
    print("🗳️  Voting Ensemble (hard)")
    print("=" * 70)

    try:
        voting_hard = build_voting_ensemble("hard")
        voting_hard.fit(X_train, y_train)
        metrics = evaluate_model(voting_hard, X_test, y_test, "Voting (hard)")
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
    except Exception as e:
        print(f"  ⚠️ فشل: {e}")
        voting_hard = None

    # 5. Stacking
    print("\n" + "=" * 70)
    print("📚 Stacking Ensemble")
    print("=" * 70)

    try:
        stacking = build_stacking_ensemble()
        stacking.fit(X_train, y_train)
        metrics = evaluate_model(stacking, X_test, y_test, "Stacking")
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
    except Exception as e:
        print(f"  ⚠️ فشل: {e}")
        stacking = None

    # 6. جدول المقارنة
    print("\n" + "=" * 70)
    print("📊 جدول المقارنة النهائي")
    print("=" * 70)
    print(f"{'النموذج':<25} {'Accuracy':>12} {'F1':>12}")
    print("-" * 70)

    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)
    for r in results_sorted:
        marker = " 🏆" if r == results_sorted[0] else ""
        print(f"{r['name']:<25} {r['accuracy']:>12.4f} {r['f1']:>12.4f}{marker}")

    # 7. حفظ أفضل نموذج
    best_name = results_sorted[0]["name"]
    best_model = None

    if best_name.startswith("Voting (soft)") and voting:
        best_model = voting
    elif best_name.startswith("Voting (hard)") and voting_hard:
        best_model = voting_hard
    elif best_name == "Stacking" and stacking:
        best_model = stacking
    elif best_name == "lr":
        best_model = build_base_models()[0][1]
    elif best_name == "svm":
        best_model = build_base_models()[1][1]
    elif best_name == "nb":
        best_model = build_base_models()[2][1]

    if best_model:
        os.makedirs(os.path.dirname(ENSEMBLE_MODEL), exist_ok=True)
        joblib.dump(best_model, ENSEMBLE_MODEL)
        print(f"\n💾 تم حفظ أفضل نموذج ({best_name}): {ENSEMBLE_MODEL}")


if __name__ == "__main__":
    main()