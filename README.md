# 🇩🇿 Arabic Sentiment Analysis — تحليل مشاعر النصوص العربية (اللهجة الجزائرية)

نظام لتحليل مشاعر النصوص المكتوبة بالعربية الفصحى أو اللهجة الجزائرية (تعليقات فيسبوك/يوتيوب)، ويصنّفها إلى: **إيجابي / سلبي / محايد**.

## 🎯 الهدف
بناء نموذج NLP قادر على فهم التعقيدات الخاصة باللهجة الجزائرية (خلط عربي-فرنسي، غياب التشكيل، اختلاف الإملاء) وتصنيف المشاعر بدقة.

## 📊 مصادر البيانات المقترحة
- [Algerian Arabic Sentiment Analysis Dataset (Kaggle)](https://www.kaggle.com/datasets/attiabendjedou/algerian-arabic-sentiment-analysis-dataset)
- DZDialect (117K تعليق مصنّف)
- Algerian Dialect Dataset (Mendeley, 45K تعليق يوتيوب)

## 🏗️ هيكل المشروع
```
arabic-sentiment-nlp/
├── data/
│   ├── raw/            # البيانات الخام (ضع ملف CSV هنا)
│   └── processed/      # البيانات بعد التنظيف
├── notebooks/          # دفاتر Jupyter للاستكشاف والتجارب
├── src/
│   ├── preprocessing.py  # تنظيف النصوص العربية
│   ├── train.py          # تدريب النموذج
│   └── predict.py        # التنبؤ بنص جديد
├── models/              # النماذج المدرَّبة المحفوظة
├── app/
│   └── streamlit_app.py  # واجهة تفاعلية
├── requirements.txt
└── README.md
```

## 🚀 التشغيل السريع
```bash
# 1. تثبيت المتطلبات
pip install -r requirements.txt

# 2. ضع بيانات CSV في data/raw/ (عمود text وعمود label)

# 3. المعالجة المسبقة
python src/preprocessing.py

# 4. التدريب
python src/train.py

# 5. تشغيل الواجهة
streamlit run app/streamlit_app.py
```

## 🧠 خطة التطوير (Roadmap)
- [x] هيكل المشروع الأساسي
- [x] معالجة مسبقة للنصوص العربية (تطبيع، إزالة تشكيل)
- [x] Baseline: TF-IDF + Logistic Regression
- [ ] تجربة AraBERT / MARBERT (Fine-tuning)
- [ ] واجهة Streamlit تفاعلية
- [ ] نشر النموذج (Deployment)

## 📈 النتائج
| النموذج | الدقة (Accuracy) | F1-Score |
|---|---|---|
| TF-IDF + LogReg | - | - |
| AraBERT | - | - |

## 🛠️ التقنيات المستخدمة
Python · scikit-learn · PyTorch · Transformers (Hugging Face) · Streamlit · PyArabic

## 📄 الترخيص
MIT License
