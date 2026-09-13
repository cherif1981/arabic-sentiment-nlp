"""
نظام تقييم احترافي موحّد
============================
- يدعم Baseline Models و AraBERT
- ينتج تقارير نصية + JSON + رسوم بيانية
- يقارن بين جميع النماذج
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # للعمل بدون واجهة رسومية
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from preprocessing import preprocess_dataframe


# ---------------- الإعدادات ----------------
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed", "clean_data.csv")
RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARABERT_DIR = os.path.join(MODELS_DIR, "arabert")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

TEXT_COL = "text"
LABEL_COL = "label"

# ترتيب الفئات الثابت
LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_AR = {"negative": "سلبي", "neutral": "محايد", "positive": "إيجابي"}

# إعدادات الرسم
plt.rcParams['font.family'] = 'DejaVu Sans'  # يدعم العربية جزئياً
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


# ============================================================
# 1. إنشاء مجلدات التقارير
# ============================================================
def setup_directories():
    """إنشاء مجلدات التقارير والرسوم."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# 2. حساب جميع المقاييس
# ============================================================
def compute_all_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    حساب جميع مقاييس التقييم.
    
    Args:
        y_true: التصنيفات الحقيقية
        y_pred: التصنيفات المتوقعة
        y_proba: الاحتمالات (اختياري، لحساب AUC)
    
    Returns:
        dict: جميع المقاييس
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }

    # مقاييس لكل فئة
    per_class = {}
    for label in LABEL_ORDER:
        if label not in np.unique(y_true) and label not in np.unique(y_pred):
            continue
        per_class[label] = {
            "precision": float(precision_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]),
            "recall": float(recall_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]),
            "f1": float(f1_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]),
            "support": int((y_true == label).sum()),
        }
    metrics["per_class"] = per_class

    return metrics


# ============================================================
# 3. رسم Confusion Matrix
# ============================================================
def plot_confusion_matrix(y_true, y_pred, title: str, output_path: str,
                          normalize: bool = False, labels=None):
    """رسم مصفوفة الالتباس."""
    if labels is None:
        labels = [l for l in LABEL_ORDER if l in np.unique(y_true) or l in np.unique(y_pred)]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
        cmap = 'Blues'
    else:
        cm_display = cm
        fmt = 'd'
        cmap = 'Blues'

    fig, ax = plt.subplots(figsize=(8, 6))

    # التسميات المزدوجة (عربي + إنجليزي)
    display_labels = [f"{LABEL_AR.get(l, l)}\n({l})" for l in labels]

    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=display_labels, yticklabels=display_labels,
        cbar_kws={'label': 'النسبة' if normalize else 'العدد'},
        ax=ax, annot_kws={"size": 13}
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('التصنيف المتوقع', fontsize=12)
    ax.set_ylabel('التصنيف الحقيقي', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 حفظ: {output_path}")


# ============================================================
# 4. رسم مقاييس لكل فئة
# ============================================================
def plot_per_class_metrics(metrics: dict, title: str, output_path: str):
    """رسم Precision/Recall/F1 لكل فئة."""
    per_class = metrics.get("per_class", {})
    if not per_class:
        return

    labels = list(per_class.keys())
    display_labels = [f"{LABEL_AR.get(l, l)}\n({l})" for l in labels]

    precision = [per_class[l]["precision"] for l in labels]
    recall = [per_class[l]["recall"] for l in labels]
    f1 = [per_class[l]["f1"] for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    # إضافة القيم فوق الأعمدة
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('الفئة', fontsize=12)
    ax.set_ylabel('القيمة', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 حفظ: {output_path}")


# ============================================================
# 5. مقارنة بين النماذج
# ============================================================
def plot_models_comparison(all_metrics: dict, output_path: str):
    """رسم مقارنة بين جميع النماذج."""
    if not all_metrics:
        return

    models = list(all_metrics.keys())
    accuracy = [all_metrics[m]["accuracy"] for m in models]
    f1_weighted = [all_metrics[m]["f1_weighted"] for m in models]
    f1_macro = [all_metrics[m]["f1_macro"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='#3498db')
    bars2 = ax.bar(x, f1_weighted, width, label='F1 (weighted)', color='#2ecc71')
    bars3 = ax.bar(x + width, f1_macro, width, label='F1 (macro)', color='#e74c3c')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('النموذج', fontsize=12)
    ax.set_ylabel('القيمة', fontsize=12)
    ax.set_title('مقارنة النماذج — Accuracy vs F1', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 حفظ: {output_path}")


# ============================================================
# 6. توزيع الثقة (Confidence Distribution)
# ============================================================
def plot_confidence_distribution(confidences: list, correct: list,
                                  title: str, output_path: str):
    """رسم توزيع الثقة للنموذج."""
    confidences = np.array(confidences)
    correct = np.array(correct)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(confidences[correct], bins=20, alpha=0.7,
            label='تصنيف صحيح', color='#2ecc71')
    ax.hist(confidences[~correct], bins=20, alpha=0.7,
            label='تصنيف خطأ', color='#e74c3c')

    ax.set_xlabel('نسبة الثقة', fontsize=12)
    ax.set_ylabel('عدد الأمثلة', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 حفظ: {output_path}")


# ============================================================
# 7. تقرير نصي مفصل
# ============================================================
def generate_text_report(metrics: dict, y_true, y_pred, model_name: str) -> str:
    """توليد تقرير نصي مفصل."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"📊 تقرير التقييم — {model_name}")
    lines.append("=" * 70)
    lines.append("")

    # المقاييس الإجمالية
    lines.append("📈 المقاييس الإجمالية:")
    lines.append("-" * 70)
    lines.append(f"  Accuracy:              {metrics['accuracy']:.4f}")
    lines.append(f"  Precision (weighted):  {metrics['precision_weighted']:.4f}")
    lines.append(f"  Recall (weighted):     {metrics['recall_weighted']:.4f}")
    lines.append(f"  F1-Score (weighted):   {metrics['f1_weighted']:.4f}")
    lines.append("")
    lines.append(f"  Precision (macro):     {metrics['precision_macro']:.4f}")
    lines.append(f"  Recall (macro):        {metrics['recall_macro']:.4f}")
    lines.append(f"  F1-Score (macro):      {metrics['f1_macro']:.4f}")
    lines.append("")

    # المقاييس لكل فئة
    lines.append("📋 المقاييس لكل فئة:")
    lines.append("-" * 70)
    lines.append(f"  {'الفئة':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("  " + "-" * 60)

    for label in LABEL_ORDER:
        if label not in metrics.get("per_class", {}):
            continue
        m = metrics["per_class"][label]
        display = f"{LABEL_AR.get(label, label)} ({label})"
        lines.append(f"  {display:<15} {m['precision']:>10.4f} "
                     f"{m['recall']:>10.4f} {m['f1']:>10.4f} "
                     f"{m['support']:>10d}")
    lines.append("")

    # تقرير sklearn
    lines.append("📋 Classification Report (sklearn):")
    lines.append("-" * 70)
    labels_present = [l for l in LABEL_ORDER if l in np.unique(y_true) or l in np.unique(y_pred)]
    lines.append(classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=[f"{LABEL_AR.get(l, l)} ({l})" for l in labels_present],
        zero_division=0
    ))
    lines.append("")

    # مصفوفة الالتباس
    lines.append("📊 Confusion Matrix:")
    lines.append("-" * 70)
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    lines.append(f"  الفئات: {labels_present}")
    lines.append("")
    lines.append(f"  {'':<20} " + " ".join(f"{l:>10}" for l in labels_present))
    for i, label in enumerate(labels_present):
        lines.append(f"  {label:<20} " + " ".join(f"{v:>10d}" for v in cm[i]))
    lines.append("")

    # مصفوفة الالتباس الطبيعية
    lines.append("📊 Confusion Matrix (Normalized):")
    lines.append("-" * 70)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    lines.append(f"  {'':<20} " + " ".join(f"{l:>10}" for l in labels_present))
    for i, label in enumerate(labels_present):
        lines.append(f"  {label:<20} " + " ".join(f"{v:>10.2%}" for v in cm_norm[i]))
    lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# 8. تقييم نموذج Baseline (sklearn)
# ============================================================
def evaluate_baseline(model_path: str, model_name: str,
                      X_test, y_test) -> dict:
    """تقييم نموذج sklearn وحفظ التقارير."""
    print(f"\n{'=' * 70}")
    print(f"📊 تقييم: {model_name}")
    print(f"{'=' * 70}")

    pipeline = joblib.load(model_path)
    y_pred = pipeline.predict(X_test)

    # الاحتمالات (إن كانت متاحة)
    y_proba = None
    confidences = []
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)
        confidences = y_proba.max(axis=1).tolist()
    elif hasattr(pipeline, "decision_function"):
        # LinearSVC: نستخدم softmax على decision_function
        decision = pipeline.decision_function(X_test)
        exp = np.exp(decision - decision.max(axis=1, keepdims=True))
        y_proba = exp / exp.sum(axis=1, keepdims=True)
        confidences = y_proba.max(axis=1).tolist()

    # حساب المقاييس
    metrics = compute_all_metrics(y_test, y_pred, y_proba)
    metrics["model"] = model_name
    metrics["n_test"] = len(y_test)

    # عرض في الطرفية
    print(f"\n🎯 Accuracy:        {metrics['accuracy']:.4f}")
    print(f"🎯 F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"🎯 F1 (macro):      {metrics['f1_macro']:.4f}")

    # التقرير النصي
    report_text = generate_text_report(metrics, y_test, y_pred, model_name)
    print("\n" + report_text)

    # حفظ التقرير النصي
    safe_name = model_name.lower().replace(" ", "_")
    report_path = os.path.join(REPORTS_DIR, f"{safe_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"📄 حفظ التقرير: {report_path}")

    # حفظ JSON
    json_path = os.path.join(REPORTS_DIR, f"{safe_name}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"📄 حفظ JSON: {json_path}")

    # الرسوم البيانية
    labels_present = [l for l in LABEL_ORDER if l in np.unique(y_test) or l in np.unique(y_pred)]

    # CM عادية
    plot_confusion_matrix(
        y_test, y_pred,
        f"Confusion Matrix — {model_name}",
        os.path.join(FIGURES_DIR, f"{safe_name}_cm.png"),
        normalize=False, labels=labels_present
    )

    # CM مطبّعة
    plot_confusion_matrix(
        y_test, y_pred,
        f"Confusion Matrix (Normalized) — {model_name}",
        os.path.join(FIGURES_DIR, f"{safe_name}_cm_normalized.png"),
        normalize=True, labels=labels_present
    )

    # مقاييس لكل فئة
    plot_per_class_metrics(
        metrics,
        f"Precision / Recall / F1 per Class — {model_name}",
        os.path.join(FIGURES_DIR, f"{safe_name}_per_class.png")
    )

    # توزيع الثقة
    if confidences:
        correct = [p == t for p, t in zip(y_pred, y_test)]
        plot_confidence_distribution(
            confidences, correct,
            f"Confidence Distribution — {model_name}",
            os.path.join(FIGURES_DIR, f"{safe_name}_confidence.png")
        )

    return metrics


# ============================================================
# 9. تقييم AraBERT
# ============================================================
def evaluate_arabert(X_test, y_test) -> dict:
    """تقييم AraBERT."""
    print(f"\n{'=' * 70}")
    print(f"📊 تقييم: AraBERT")
    print(f"{'=' * 70}")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("⚠️ transformers غير مثبتة — تخطي AraBERT")
        return None

    if not os.path.exists(ARABERT_DIR):
        print(f"⚠️ لم يتم العثور على AraBERT في: {ARABERT_DIR}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  الجهاز: {device}")

    tokenizer = AutoTokenizer.from_pretrained(ARABERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(ARABERT_DIR)
    model.to(device)
    model.eval()

    # التنبؤ بالدفعات
    batch_size = 16
    all_preds = []
    all_probs = []

    print(f"🔄 التنبؤ على {len(X_test)} مثال...")
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size].tolist()
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=128, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_preds.extend(probs.argmax(axis=1).tolist())

    # تحويل الأرقام إلى تصنيفات نصية
    id2label = model.config.id2label
    y_pred = [id2label[int(p)] for p in all_preds]

    # حساب المقاييس
    metrics = compute_all_metrics(y_test, y_pred, np.array(all_probs))
    metrics["model"] = "AraBERT"
    metrics["n_test"] = len(y_test)

    print(f"\n🎯 Accuracy:        {metrics['accuracy']:.4f}")
    print(f"🎯 F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"🎯 F1 (macro):      {metrics['f1_macro']:.4f}")

    # التقرير النصي
    report_text = generate_text_report(metrics, y_test, y_pred, "AraBERT")
    print("\n" + report_text)

    # حفظ
    report_path = os.path.join(REPORTS_DIR, "arabert_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"📄 حفظ التقرير: {report_path}")

    json_path = os.path.join(REPORTS_DIR, "arabert_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"📄 حفظ JSON: {json_path}")

    # الرسوم
    labels_present = [l for l in LABEL_ORDER if l in np.unique(y_test) or l in np.unique(y_pred)]

    plot_confusion_matrix(
        y_test, y_pred,
        "Confusion Matrix — AraBERT",
        os.path.join(FIGURES_DIR, "arabert_cm.png"),
        normalize=False, labels=labels_present
    )

    plot_confusion_matrix(
        y_test, y_pred,
        "Confusion Matrix (Normalized) — AraBERT",
        os.path.join(FIGURES_DIR, "arabert_cm_normalized.png"),
        normalize=True, labels=labels_present
    )

    plot_per_class_metrics(
        metrics,
        "Precision / Recall / F1 per Class — AraBERT",
        os.path.join(FIGURES_DIR, "arabert_per_class.png")
    )

    # توزيع الثقة
    confidences = [max(p) for p in all_probs]
    correct = [p == t for p, t in zip(y_pred, y_test)]
    plot_confidence_distribution(
        confidences, correct,
        "Confidence Distribution — AraBERT",
        os.path.join(FIGURES_DIR, "arabert_confidence.png")
    )

    return metrics


# ============================================================
# 10. تقرير المقارنة النهائي
# ============================================================
def generate_comparison_report(all_metrics: dict) -> str:
    """توليد تقرير مقارنة بين جميع النماذج."""
    lines = []
    lines.append("=" * 90)
    lines.append("📊 تقرير المقارنة النهائي — جميع النماذج")
    lines.append("=" * 90)
    lines.append("")

    # جدول المقارنة
    lines.append(f"{'النموذج':<25} {'Accuracy':>10} {'F1(w)':>10} "
                 f"{'F1(m)':>10} {'Precision':>11} {'Recall':>10}")
    lines.append("-" * 90)

    # ترتيب حسب F1-weighted
    sorted_models = sorted(
        all_metrics.items(),
        key=lambda x: x[1]["f1_weighted"],
        reverse=True
    )

    for name, m in sorted_models:
        lines.append(
            f"{name:<25} {m['accuracy']:>10.4f} {m['f1_weighted']:>10.4f} "
            f"{m['f1_macro']:>10.4f} {m['precision_weighted']:>11.4f} "
            f"{m['recall_weighted']:>10.4f}"
        )

    lines.append("=" * 90)
    lines.append("")

    # أفضل نموذج
    best_name, best_m = sorted_models[0]
    lines.append(f"🏆 أفضل نموذج: {best_name}")
    lines.append(f"   Accuracy:     {best_m['accuracy']:.4f}")
    lines.append(f"   F1-weighted:  {best_m['f1_weighted']:.4f}")
    lines.append(f"   F1-macro:     {best_m['f1_macro']:.4f}")
    lines.append("")

    # جدول Markdown للـ README
    lines.append("📋 Markdown Table (للنسخ في README):")
    lines.append("")
    lines.append("| Model | Accuracy | F1 (weighted) | F1 (macro) | Precision | Recall |")
    lines.append("|-------|----------|---------------|------------|-----------|--------|")
    for name, m in sorted_models:
        marker = " 🏆" if name == best_name else ""
        lines.append(
            f"| {name}{marker} | {m['accuracy']:.4f} | {m['f1_weighted']:.4f} | "
            f"{m['f1_macro']:.4f} | {m['precision_weighted']:.4f} | "
            f"{m['recall_weighted']:.4f} |"
        )
    lines.append("")
    lines.append("=" * 90)

    return "\n".join(lines)


# ============================================================
# 11. الدالة الرئيسية
# ============================================================
def main():
    print("=" * 90)
    print("📊 نظام التقييم الاحترافي — تحليل المشاعر العربية")
    print("=" * 90)

    setup_directories()

    # 1. تحميل البيانات
    if not os.path.exists(PROCESSED_DATA):
        print(f"❌ لم يتم العثور على: {PROCESSED_DATA}")
        print("💡 شغّل train.py أولاً.")
        return

    df = pd.read_csv(PROCESSED_DATA, encoding='utf-8-sig')
    print(f"📂 تم تحميل {len(df)} صف")

    # 2. تقسيم train/test موحد لجميع النماذج
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    use_stratify = None
    class_counts = pd.Series(y).value_counts()
    if len(df) >= 30 and class_counts.min() >= 2:
        use_stratify = y

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=use_stratify
    )
    print(f"📊 مجموعة الاختبار: {len(X_test)} مثال")

    # 3. تقييم جميع نماذج Baseline
    all_metrics = {}

    baseline_models = [
        ("model_naive_bayes.pkl", "Naive Bayes"),
        ("model_logistic_regression.pkl", "Logistic Regression"),
        ("model_linear_svm.pkl", "Linear SVM"),
    ]

    for filename, model_name in baseline_models:
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            metrics = evaluate_baseline(model_path, model_name, X_test, y_test)
            all_metrics[model_name] = metrics
        else:
            print(f"⚠️ تخطي {model_name} — الملف غير موجود")

    # 4. تقييم AraBERT (إن كان متاحاً)
    arabert_metrics = evaluate_arabert(X_test, y_test)
    if arabert_metrics:
        all_metrics["AraBERT"] = arabert_metrics

    # 5. تقرير المقارنة النهائي
    if len(all_metrics) > 1:
        comparison = generate_comparison_report(all_metrics)
        print("\n" + comparison)

        # حفظ التقرير
        comparison_path = os.path.join(REPORTS_DIR, "comparison_report.txt")
        with open(comparison_path, "w", encoding="utf-8") as f:
            f.write(comparison)
        print(f"📄 حفظ التقرير النهائي: {comparison_path}")

        # رسم المقارنة
        plot_models_comparison(
            all_metrics,
            os.path.join(FIGURES_DIR, "models_comparison.png")
        )

        # حفظ JSON
        json_path = os.path.join(REPORTS_DIR, "comparison_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 90)
    print("✅ اكتمل التقييم بنجاح!")
    print(f"📁 جميع التقارير في: {REPORTS_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()