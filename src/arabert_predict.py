"""
واجهة Streamlit — تدعم TF-IDF و AraBERT
"""
import os
import sys
import streamlit as st
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


st.set_page_config(page_title="تحليل المشاعر العربية", page_icon="😊", layout="wide")

st.markdown("""
    <style>
    .main { direction: rtl; text-align: right; }
    .stTextArea textarea { direction: rtl; text-align: right;
                          font-size: 16px; font-family: 'Arial'; }
    h1, h2, h3, p, label { direction: rtl; text-align: right; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classic_predictor():
    try:
        from predict import SentimentPredictor
        return SentimentPredictor(), "TF-IDF"
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_arabert_predictor():
    try:
        from arabert_predict import AraBERTPredictor
        return AraBERTPredictor(), "AraBERT"
    except Exception as e:
        return None, str(e)


def main():
    st.title("🎭 تحليل المشاعر للنصوص العربية")
    st.markdown("قارن بين النموذجين: **TF-IDF** الكلاسيكي و **AraBERT** الحديث")

    # ---------------- اختيار النموذج ----------------
    model_choice = st.radio(
        "🔬 اختر النموذج:",
        ["TF-IDF + Linear SVM", "AraBERT (الأدق)"],
        horizontal=True
    )

    # ---------------- تحميل النموذج ----------------
    if "AraBERT" in model_choice:
        with st.spinner("جاري تحميل AraBERT (قد يستغرق دقيقة)..."):
            predictor, status = load_arabert_predictor()
    else:
        with st.spinner("جاري تحميل النموذج الكلاسيكي..."):
            predictor, status = load_classic_predictor()

    if predictor is None:
        st.error(f"⚠️ فشل تحميل النموذج: {status}")
        st.stop()

    st.success(f"✅ تم تحميل: {status}")

    # ---------------- الإدخال ----------------
    user_text = st.text_area(
        "📝 أدخل النص هنا:",
        height=150,
        placeholder="مثال: هذا المنتج رائع جداً، أنصح بمشاهدته"
    )

    if st.button("🔍 تحليل المشاعر", use_container_width=True, type="primary"):
        if not user_text.strip():
            st.warning("⚠️ يرجى إدخال نص")
        else:
            with st.spinner("جاري التحليل..."):
                result = predictor.predict(user_text)

            st.markdown("---")
            st.markdown("### 📊 النتيجة")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("التصنيف", result["label_ar"])
            with col_b:
                if "confidence" in result:
                    st.metric("الثقة", f"{result['confidence']:.2%}")

            # شريط الاحتمالات
            if "probabilities" in result:
                st.markdown("### 📈 توزيع الاحتمالات")
                probs = result["probabilities"]
                labels = ["سلبي", "محايد", "إيجابي"]
                chart_data = pd.DataFrame({
                    "الفئة": labels,
                    "الاحتمال": probs
                }).set_index("الفئة")
                st.bar_chart(chart_data)


if __name__ == "__main__":
    main()