# 📊 تقارير التقييم

هذا المجلد يحتوي على تقارير تقييم النماذج.

## 📁 الملفات

### تقارير نصية
- `baseline_report.txt` — تقرير النماذج الكلاسيكية
- `arabert_report.txt` — تقرير AraBERT
- `comparison_report.txt` — مقارنة جميع النماذج

### تقارير JSON
- `baseline_report.json`, `arabert_report.json`, `comparison_report.json`
- مناسبة للاستخدام الآلي أو التكامل مع لوحات المعلومات

### الرسوم البيانية (`figures/`)
- `*_cm.png` — Confusion Matrix
- `*_cm_normalized.png` — Confusion Matrix مطبّعة
- `*_per_class.png` — Precision/Recall/F1 لكل فئة
- `*_confidence.png` — توزيع الثقة
- `models_comparison.png` — مقارنة جميع النماذج

## 🎯 كيف تقرأ التقارير؟

### المقاييس الأساسية
- **Accuracy**: النسبة الإجمالية للتصنيفات الصحيحة
- **Precision**: من بين ما تنبأ به النموذج كإيجابي، كم كان صحيحاً؟
- **Recall**: من بين الإيجابيات الحقيقية، كم اكتشفها النموذج؟
- **F1**: المتوسط التوافقي بين Precision و Recall

### متى تختار أي مقياس؟
- **فئات متوازنة**: Accuracy كافٍ
- **فئات غير متوازنة**: F1-macro أو F1-weighted
- **مهم تجنّب الإنذارات الكاذبة**: Precision
- **مهم اكتشاف كل الحالات**: Recall

## 🔄 إعادة التوليد

```bash
python src/evaluate.py