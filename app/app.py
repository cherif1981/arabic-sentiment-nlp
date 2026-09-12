"""
واجهة Streamlit لتحليل المشاعر العربية
=======================================
واجهة احترافية تدعم عدة نماذج ومقارنة فورية.
"""
import os
import sys
import streamlit as st
import pandas as pd

# إضافة المجلدات إلى المسار
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    EXAMPLES, MODELS_INFO, LABEL_MAP
)
from components import (
    render_result_card, render_probability_chart,
    render_confidence_gauge, render_cleaned_text,
    render_model_info, render_sidebar_info
)


# ============================================================
# إعداد الصفحة
# ============================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS مخصص
# ============================================================
st.markdown("""
<style>
    /* العربية والاتجاه */
    .main, .stMarkdown, h1, h2, h3, h4, h5, p, label {
        direction: rtl;
        text-align: right;
    }

    /* TextArea */
    .stTextArea textarea {
        direction: rtl !important;
        text-align: right !important;
        font-size: 17px !important;
        font-family: 'Arial', 'Segoe UI', sans-serif !important;
        padding: 15px !important;
        border-radius: 10px !important;
    }

    /* الأزرار */
    .stButton > button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }

    /* العناوين */
    h1 {
        background: linear-gradient(135deg, #3498db, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px !important;
        font-weight: 900 !important;
        text-align: center !important;
        padding: 10px 0;
    }

    /* Radio buttons */
    .stRadio > div {
        direction: rtl;
        text-align: right;
    }

    /* Expander */
    .streamlit-expanderHeader {
        direction: rtl;
        text-align: right;
        font-size: 16px;
    }

    /* المترية */
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }

    /* إخفاء شعار Streamlit */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* الخلفية */
    .stApp {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# تحميل النماذج (Cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_classic_predictor(model_path: str, model_name: str):
    """تحميل نموذج sklearn."""
    try:
        from predict import SentimentPredictor
        return SentimentPredictor(model_path=model_path), None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_arabert_predictor(model_path: str):
    """تحميل AraBERT."""
    try:
        from arabert_predict import AraBERTPredictor
        return AraBERTPredictor(model_dir=model_path), None
    except Exception as e:
        return None, str(e)


def load_model(model_name: str):
    """تحميل النموذج المحدد."""
    info = MODELS_INFO[model_name]

    if not os.path.exists(info["path"]):
        return None, f"❌ الملف غير موجود: {info['path']}"

    if info["type"] == "arabert":
        return load_arabert_predictor(info["path"])
    else:
        return load_classic_predictor(info["path"], model_name)


# ============================================================
# الواجهة الرئيسية
# ============================================================
def main():
    # Sidebar
    render_sidebar_info()

    # العنوان
    st.markdown("""
    <h1>🎭 Arabic Sentiment Analyzer 🇩🇿</h1>
    <p style="text-align:center; color:#666; font-size:18px; margin-top:-20px;">
        نظام تحليل مشاعر النصوص العربية بالذكاء الاصطناعي
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- اختيار النموذج ----------------
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🔬 اختر النموذج")
        model_choice = st.radio(
            "النموذج:",
            list(MODELS_INFO.keys()),
            horizontal=False,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("### 📊 المقارنة السريعة")

        # جدول مقارنة مصغّر
        comparison_data = pd.DataFrame({
            "النموذج": ["AraBERT", "Linear SVM", "Logistic Regression", "Naive Bayes"],
            "الدقة المتوقعة": ["85%+ 🏆", "~58%", "~50%", "~42%"],
        })
        st.dataframe(comparison_data, hide_index=True, use_container_width=True)

    # عرض معلومات النموذج
    render_model_info(model_choice, MODELS_INFO[model_choice])

    # ---------------- تحميل النموذج ----------------
    with st.spinner(f"جاري تحميل {model_choice}..."):
        predictor, error = load_model(model_choice)

    if predictor is None:
        st.error(f"⚠️ فشل تحميل النموذج: {error}")
        st.info("💡 تأكد من تشغيل `python src/train.py` لتدريب النماذج.")
        st.stop()

    st.success(f"✅ تم تحميل النموذج بنجاح")

    # ---------------- الإدخال ----------------
    st.markdown("---")
    st.markdown("### 📝 أدخل النص العربي للتحليل")

    # أمثلة سريعة
    st.markdown("**💡 أمثلة سريعة — اضغط للتجربة:**")

    col_ex1, col_ex2, col_ex3 = st.columns(3)
    example_categories = list(EXAMPLES.keys())

    for i, category in enumerate(example_categories):
        with [col_ex1, col_ex2, col_ex3][i]:
            if st.button(f"📌 {category}",
                         key=f"cat_{category}",
                         use_container_width=True):
                st.session_state["user_text"] = EXAMPLES[category][0]

    # حقل النص
    user_text = st.text_area(
        "النص:",
        value=st.session_state.get("user_text", ""),
        height=140,
        placeholder="اكتب أو الصق النص العربي هنا...",
        key="text_input",
        label_visibility="collapsed"
    )

    # ---------------- التحليل ----------------
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_clicked = st.button("🔍 تحليل المشاعر",
                                     use_container_width=True,
                                     type="primary")

    # زر مسح
    with col_btn1:
        if st.button("🗑️ مسح", use_container_width=True):
            st.session_state["user_text"] = ""
            st.rerun()

    # ---------------- النتيجة ----------------
    if analyze_clicked:
        if not user_text.strip():
            st.warning("⚠️ يرجى إدخال نص أولاً")
        else:
            with st.spinner("🔍 جاري التحليل..."):
                try:
                    result = predictor.predict(user_text)
                except Exception as e:
                    st.error(f"❌ خطأ في التحليل: {e}")
                    return

            st.markdown("---")
            st.markdown("## 📊 النتائج")

            # بطاقة النتيجة
            render_result_card(result, model_choice)

            # المقاييس في عمودين
            col_a, col_b = st.columns([1, 1])

            with col_a:
                if "confidence" in result:
                    render_confidence_gauge(result["confidence"])

            with col_b:
                st.markdown("#### 📈 ملخص")
                st.metric("التصنيف", result["label_ar"])
                if "confidence" in result:
                    st.metric("الثقة", f"{result['confidence']:.1%}")
                st.metric("النموذج", model_choice)

            # الرسم البياني
            if "probabilities" in result and hasattr(predictor, "model"):
                try:
                    # محاولة الحصول على الفئات
                    if hasattr(predictor.model, "classes_"):
                        classes = list(predictor.model.classes_)
                    elif hasattr(predictor.model, "config"):
                        classes = [predictor.model.config.id2label[i]
                                   for i in range(len(predictor.model.config.id2label))]
                    else:
                        classes = ["negative", "neutral", "positive"]

                    render_probability_chart(result["probabilities"], classes)
                except Exception as e:
                    pass

            # النص بعد التنظيف
            if "cleaned" in result:
                render_cleaned_text(result["cleaned"])

            # ---------------- التحليل المتعدد ----------------
            st.markdown("---")
            st.markdown("### 🔄 جرب نموذجاً آخر")

            other_models = [m for m in MODELS_INFO.keys() if m != model_choice]
            cols = st.columns(len(other_models))

            for i, other_model in enumerate(other_models):
                with cols[i]:
                    if st.button(f"🔄 {other_model}",
                                 key=f"switch_{other_model}",
                                 use_container_width=True):
                        st.session_state["switch_model"] = other_model
                        st.rerun()

    # ---------------- معلومات إضافية ----------------
    with st.expander("ℹ️ كيف يعمل النظام؟"):
        st.markdown("""
        ### 🔬 خط أنابيب المعالجة

        1. **تنظيف النص**: إزالة URLs، mentions، emojis، التكرار
        2. **التوحيد**: أ/إ/آ→ا، ة→ه، ى→ي
        3. **التصنيف**: النموذج يتوقع الفئة الأكثر احتمالاً

        ### 🎯 الفئات

        - **إيجابي 😊**: النص يعبر عن رضا أو إعجاب
        - **سلبي 😞**: النص يعبر عن استياء أو انتقاد
        - **محايد 😐**: النص بدون مشاعر واضحة

        ### 📊 المقاييس

        - **الثقة**: مدى تأكد النموذج من تصنيفه
        - **التوزيع**: احتمالات كل فئة
        """)

    # ---------------- Footer ----------------
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#999; padding:20px; direction:rtl;">
        <p>صُنع بـ ❤️ باستخدام Streamlit + scikit-learn + Transformers</p>
        <p style="font-size:13px;">
            Arabic Sentiment Analyzer — 
            <a href="https://github.com/username/arabic-sentiment-nlp" 
               style="color:#3498db; text-decoration:none;">
                GitHub
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()