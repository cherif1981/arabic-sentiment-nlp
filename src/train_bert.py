# src/train_bert.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import clean_text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "arabert")
os.makedirs(MODEL_DIR, exist_ok=True)

LABELS = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABELS.items()}

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["text_clean"] = df["text"].astype(str).apply(clean_text)
df["label_id"] = df["label"].map(LABELS)
df = df.dropna(subset=["label_id"])
df = df[df["text_clean"].str.len() >= 3].reset_index(drop=True)

train_df, test_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["label_id"]
)

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ArabicDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=128, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels))

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}

train_ds = ArabicDataset(train_df["text_clean"], train_df["label_id"])
test_ds = ArabicDataset(test_df["text_clean"], test_df["label_id"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABELS
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"\n✅ تم الحفظ في: {MODEL_DIR}")