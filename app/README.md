# 🎭 Arabic Sentiment Analyzer

واجهة Streamlit تفاعلية لتحليل مشاعر النصوص العربية.

## 🚀 التشغيل

```bash
# من جذر المشروع
pip install -r requirements.txt

# تأكد من تدريب النماذج أولاً
python src/train.py
python src/arabert_train.py  # اختياري

# تشغيل الواجهة
cd app
streamlit run app.py