"""
مكونات واجهة قابلة لإعادة الاستخدام
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import LABEL_MAP, LABEL_COLORS


def render_result_card(result: dict, model_name: str, show_cleaned: bool = True):
    """عرض بطاقة النتيجة بشكل أنيق."""
    label_ar = result["label_ar"]
    confidence = result.get("confidence", 0)
    color = LABEL_COLORS.get(label_ar, "#3498db")

    # بطاقة النتيجة الرئيسية
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border-right: 5px solid {color};
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        direction: rtl;
        text-align: right;
    ">
        <h2 style="color: {color}; margin: 0; font-size: 32px;">
            {label_ar}
        </h2>
        <p style="color: #666; margin: 10px 0 0 0; font-size: 16px;">
            نسبة الثقة: <b style="color: {color}; font-size: 20px;">{confidence:.1%}</b>
        </p>
        <p style="color: #888; margin: 5px 0 0 0; font-size: 13px;">
            النموذج: {model_name}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_probability_chart(probabilities: list, classes: list):
    """عرض توزيع الاحتمالات بشكل أنيق."""
    labels = [LABEL_MAP.get(c, str(c)) for c in classes]

    df = pd.DataFrame({
        "الفئة": labels,
        "الاحتمال": probabilities
    })

    colors = [LABEL_COLORS.get(l, "#3498db") for l in labels]

    fig = go.Figure(data=[
        go.Bar(
            x=df["الاحتمال"],
            y=df["الفئة"],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{p:.1%}" for p in df["الاحتمال"]],
            textposition='outside',
            textfont=dict(size=14, color='#333'),
        )
    ])

    fig.update_layout(
        title=dict(text="توزيع الاحتمالات", font=dict(size=18)),
        xaxis=dict(range=[0, 1.15], tickformat='.0%',
                   title="الاحتمال", showgrid=True),
        yaxis=dict(title=""),
        height=250,
        margin=dict(l=20, r=60, t=50, b=20),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, use_container_width=True)


def render_confidence_gauge(confidence: float):
    """عرض عدّاد دائري للثقة."""
    color = "#2ecc71" if confidence > 0.7 else "#f39c12" if confidence > 0.5 else "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': "%", 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color, 'thickness': 0.3},
            'steps': [
                {'range': [0, 50], 'color': "#ffe5e5"},
                {'range': [50, 70], 'color': "#fff5e5"},
                {'range': [70, 100], 'color': "#e5f5e5"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cleaned_text(cleaned: str):
    """عرض النص بعد التنظيف."""
    with st.expander("🧹 النص بعد المعالجة"):
        st.markdown(f"""
        <div style="
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-right: 4px solid #3498db;
            direction: rtl;
            text-align: right;
            font-family: 'Arial';
            color: #333;
        ">
            {cleaned if cleaned else '<i style="color:#999;">(نص فارغ بعد المعالجة)</i>'}
        </div>
        """, unsafe_allow_html=True)


def render_model_info(model_name: str, info: dict):
    """عرض معلومات النموذج المختار."""
    st.markdown(f"""
    <div style="
        background-color: {info['color']}15;
        padding: 12px 18px;
        border-radius: 8px;
        border-right: 4px solid {info['color']};
        direction: rtl;
        text-align: right;
        margin-bottom: 20px;
    ">
        <b style="color: {info['color']}; font-size: 15px;">
            {model_name}
        </b>
        <span style="color: #666; font-size: 14px;">
            — {info['desc']}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_info():
    """عرض معلومات جانبية."""
    st.sidebar.markdown("### ℹ️ عن المشروع")
    st.sidebar.markdown("""
    **Arabic Sentiment Analyzer**  
    نظام تحليل مشاعر النصوص العربية
    
    🎯 **الفئات:**
    - 😊 إيجابي (Positive)
    - 😞 سلبي (Negative)
    - 😐 محايد (Neutral)
    
    🔬 **النماذج المدعومة:**
    - AraBERT (Transformer)
    - Linear SVM
    - Logistic Regression
    - Naive Bayes
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 التقارير")
    st.sidebar.markdown("""
    يتم حفظ جميع التقارير في مجلد `reports/`
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 روابط")
    st.sidebar.markdown("""
    - [GitHub](https://github.com/username/arabic-sentiment-nlp)
    - [Documentation](README.md)
    """)