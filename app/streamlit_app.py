"""
واجهة Streamlit التفاعلية لتحليل المشاعر العربية
"""
import os
import sys
import streamlit as st

# إضافة src إلى المسار
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import SentimentPredictor


# ---------------- إعداد الصفحة ----------------
st.set_page_config(
    page_title="تحليل المشاعر العربية",
    page_icon="😊",
    layout="wide"
)

# CSS مخصص للعربية
st.markdown("""
    <style>
    .main { direction: rtl; text-align: right; }
    .stTextArea textarea { direction: rtl; text-align: right;
                          font-size: 16px; font-family: 'Arial'; }
    .result-box { padding: 20px; border-radius: 10px;
                 background-color: #f0f2f6; margin-top: 20px; }
    h1, h2, h3, p { direction: rtl; text-align: right; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """تحميل النموذج مرة واحدة فقط."""
    try:
        return SentimentPredictor()
    except FileNotFoundError as e:
        return None


def main():
    st.title("🎭 تحليل المشاعر للنصوص العربية")
    st.markdown("أدخل نصاً عربياً لتحليل مشاعره (إيجابي / سلبي / محايد)")

    predictor = load_predictor()

    if predictor is None:
        st.error("⚠️ لم يتم العثور على النموذج المدرب. "
                 "يرجى تشغيل `src/train.py` أولاً.")
        st.stop()

    # ---------------- الإدخال ----------------
    col1, col2 = st.columns([3, 1])

    with col1:
        user_text = st.text_area(
            "📝 أدخل النص هنا:",
            height=150,
            placeholder="مثال: هذا الفيلم رائع جداً، أنصح بمشاهدته"
        )

    with col2:
        st.markdown("### ⚙️ خيارات")
        show_clean = st.checkbox("عرض النص المُنظَّف", value=True)
        show_conf = st.checkbox("عرض نسبة الثقة", value=True)

    # ---------------- التنبؤ ----------------
    if st.button("🔍 تحليل المشاعر", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ يرجى إدخال نص أولاً")
        else:
            with st.spinner("جاري التحليل..."):
                result = predictor.predict(user_text)

            st.markdown("---")
            st.markdown("### 📊 النتيجة")

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("التصنيف المتوقع", result["label_ar"])

            with col_b:
                if show_conf and "confidence" in result:
                    st.metric("نسبة الثقة",
                              f"{result['confidence']:.2%}")

            if show_clean:
                with st.expander("🧹 النص بعد التنظيف"):
                    st.write(result["cleaned"])

            # شريط الاحتمالات
            if "probabilities" in result:
                st.markdown("### 📈 توزيع الاحتمالات")
                import pandas as pd
                probs = result["probabilities"]
                labels = [str(c) for c in predictor.model.classes_]
                chart_data = pd.DataFrame({
                    "الفئة": labels,
                    "الاحتمال": probs
                }).set_index("الفئة")
                st.bar_chart(chart_data)

    # ---------------- أمثلة ----------------
    st.markdown("---")
    st.markdown("### 💡 أمثلة للتجربة")

    examples = [
        "المنتج ممتاز وجودة عالية جداً",
        "خدمة سيئة ولن أتعامل معهم مجدداً",
        "الأسعار معقولة والجودة متوسطة"
    ]

    for i, ex in enumerate(examples):
        if st.button(f"مثال {i+1}: {ex[:40]}...", key=f"ex_{i}"):
            st.session_state["example"] = ex
            st.rerun()


if __name__ == "__main__":
    main()