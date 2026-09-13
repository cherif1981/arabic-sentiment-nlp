"""
تدريب نموذج AraBERT لتحليل المشاعر العربية
=============================================
يستخدم AraBERTv0.2-base مع Fine-tuning كامل.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from preprocessing import preprocess


# ---------------- الإعدادات ----------------
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")
ARABERT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "arabert")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "models", "arabert_results.json")

TEXT_COL = "text"
LABEL_COL = "label"

# خريطة التصنيفات: النص → رقم
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Hyperparameters (مبنية على توصيات AraBERT)
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42


# ============================================================
# 1. تحميل البيانات
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    """تحميل البيانات من CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ لم يتم العثور على: {path}")

    for enc in ['utf-8-sig', 'utf-8', 'cp1256']:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✅ تم تحميل {len(df)} صف")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("❌ فشل قراءة الملف")


# ============================================================
# 2. Dataset مخصص لـ PyTorch
# ============================================================
class ArabicSentimentDataset(Dataset):
    """Dataset مخصص لـ AraBERT."""

    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # ✅ AraBERT يحتاج preprocessing خاص
        # لا نستخدم clean_text القوي، بل AraBERT preprocessing الخفيف
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# 3. مقاييس التقييم
# ============================================================
def compute_metrics(eval_pred):
    """حساب الدقة و F1."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
    }


# ============================================================
# 4. التدريب الرئيسي
# ============================================================
def main():
    print("=" * 70)
    print("🤖 تدريب AraBERT لتحليل المشاعر العربية")
    print("=" * 70)

    # ✅ التحقق من GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  الجهاز: {device.upper()}")
    if device == "cpu":
        print("⚠️  لا يوجد GPU — التدريب سيكون بطيئاً جداً!")
        print("   💡 فكّر في استخدام Google Colab (GPU مجاني)")

    # 1. تحميل البيانات
    df = load_data(RAW_DATA_PATH)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"❌ الملف يحتاج عمودين: {TEXT_COL}, {LABEL_COL}")

    # تحويل التصنيفات إلى أرقام
    df = df[df[LABEL_COL].isin(LABEL2ID.keys())].reset_index(drop=True)
    df['label_id'] = df[LABEL_COL].map(LABEL2ID)

    print(f"\n📌 توزيع الفئات:")
    print(df[LABEL_COL].value_counts())

    # 2. تقسيم البيانات
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[TEXT_COL].values,
        df['label_id'].values,
        test_size=0.2,
        random_state=SEED,
        stratify=df['label_id'].values if df['label_id'].value_counts().min() >= 2 else None
    )

    print(f"\n📊 التدريب: {len(train_texts)} | التحقق: {len(val_texts)}")

    # 3. تحميل الـ Tokenizer
    print(f"\n📥 تحميل {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. تحميل النموذج
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    # 5. بناء Datasets
    train_dataset = ArabicSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = ArabicSentimentDataset(val_texts, val_labels, tokenizer)

    # 6. إعدادات التدريب
    os.makedirs(ARABERT_MODEL_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=ARABERT_MODEL_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        fp16=(device == "cuda"),  # ✅ تسريع على GPU
        report_to="none"
    )

    # 7. إنشاء Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 8. التدريب
    print("\n🎓 بدء التدريب...")
    trainer.train()

    # 9. التقييم النهائي
    print("\n📊 التقييم النهائي...")
    results = trainer.evaluate()

    print(f"\n{'=' * 60}")
    print(f"🎯 Accuracy:     {results['eval_accuracy']:.4f}")
    print(f"🎯 F1 (weighted): {results['eval_f1_weighted']:.4f}")
    print(f"🎯 F1 (macro):    {results['eval_f1_macro']:.4f}")
    print(f"{'=' * 60}")

    # 10. حفظ النموذج
    trainer.save_model(ARABERT_MODEL_DIR)
    tokenizer.save_pretrained(ARABERT_MODEL_DIR)
    print(f"\n💾 تم حفظ النموذج في: {ARABERT_MODEL_DIR}")

    # 11. تقرير مفصل
    print("\n📋 تقرير التصنيف المفصل:")
    preds_output = trainer.predict(val_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    print(classification_report(val_labels, y_pred,
                                target_names=list(LABEL2ID.keys()),
                                zero_division=0))

    # 12. حفظ النتائج في JSON
    report = {
        "model": MODEL_NAME,
        "accuracy": float(results['eval_accuracy']),
        "f1_weighted": float(results['eval_f1_weighted']),
        "f1_macro": float(results['eval_f1_macro']),
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_len": MAX_LEN,
    }
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"📄 تم حفظ التقرير في: {RESULTS_PATH}")

    print("\n✅ اكتمل التدريب بنجاح!")


if __name__ == "__main__":
    main()