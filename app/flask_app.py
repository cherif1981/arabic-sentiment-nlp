"""
تطبيق Flask لتحليل المشاعر العربية
=====================================
تشغيل:
    python app/flask_app.py
ثم افتح: http://127.0.0.1:5000
"""
import os
import sys

# ✅ إصلاح UTF-8 على Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from flask import Flask, render_template_string, request, jsonify
from predict import SentimentPredictor


# ============================================================
# تحميل النموذج
# ============================================================
print("=" * 60)
print("  Arabic Sentiment Analyzer - Flask App")
print("=" * 60)
print("\n[1/2] Loading model...")

try:
    predictor = SentimentPredictor()
    print("[OK] Model loaded successfully\n")
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    print("\n[TIP] Run this first: python src/train.py")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)


# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)


# ============================================================
# HTML Template
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Arabic Sentiment Analyzer</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
        font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
        direction: rtl;
    }

    .container {
        background: white;
        border-radius: 24px;
        padding: 45px;
        max-width: 850px;
        width: 100%;
        box-shadow: 0 25px 70px rgba(0,0,0,0.3);
        animation: slideUp 0.5s ease-out;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .header { text-align: center; margin-bottom: 30px; }

    h1 {
        font-size: 34px;
        color: #333;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    .subtitle {
        color: #888;
        font-size: 15px;
    }

    .badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-top: 10px;
    }

    textarea {
        width: 100%;
        min-height: 140px;
        padding: 18px;
        font-size: 17px;
        font-family: inherit;
        border: 2px solid #e0e0e0;
        border-radius: 14px;
        resize: vertical;
        direction: rtl;
        text-align: right;
        transition: border-color 0.3s, box-shadow 0.3s;
        line-height: 1.6;
    }

    textarea:focus {
        outline: none;
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    }

    .btn-analyze {
        width: 100%;
        padding: 18px;
        font-size: 19px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 14px;
        cursor: pointer;
        margin-top: 15px;
        transition: transform 0.2s, box-shadow 0.2s;
        font-family: inherit;
    }

    .btn-analyze:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }

    .btn-analyze:disabled {
        opacity: 0.7;
        cursor: not-allowed;
    }

    .examples {
        margin-top: 20px;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: center;
    }

    .examples button {
        padding: 9px 18px;
        font-size: 14px;
        border: 2px solid #e0e0e0;
        background: white;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.2s;
        font-family: inherit;
    }

    .examples button:hover {
        border-color: #667eea;
        color: #667eea;
        transform: translateY(-1px);
    }

    .result {
        margin-top: 25px;
        padding: 30px;
        border-radius: 16px;
        display: none;
        animation: fadeIn 0.4s ease-out;
    }

    .result.show { display: block; }

    .result.positive {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-right: 5px solid #28a745;
    }

    .result.negative {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-right: 5px solid #dc3545;
    }

    .result.neutral {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-right: 5px solid #ffc107;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result .emoji {
        font-size: 70px;
        text-align: center;
        margin-bottom: 10px;
        animation: bounce 0.6s ease-out;
    }

    @keyframes bounce {
        0% { transform: scale(0); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }

    .result .label {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }

    .result.positive .label { color: #155724; }
    .result.negative .label { color: #721c24; }
    .result.neutral  .label { color: #856404; }

    .confidence {
        text-align: center;
        font-size: 19px;
        color: #444;
        margin-bottom: 20px;
    }

    .confidence b { font-size: 24px; }

    .progress-bg {
        background: rgba(255,255,255,0.6);
        height: 14px;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 25px;
    }

    .progress-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 1s ease-out;
    }

    .result.positive .progress-bar { background: linear-gradient(90deg, #28a745, #20c997); }
    .result.negative .progress-bar { background: linear-gradient(90deg, #dc3545, #e74c3c); }
    .result.neutral  .progress-bar { background: linear-gradient(90deg, #ffc107, #ffb300); }

    .section-title {
        font-size: 15px;
        font-weight: bold;
        color: #555;
        margin-bottom: 15px;
        text-align: right;
    }

    .prob-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 10px 0;
    }

    .prob-label {
        width: 85px;
        font-weight: bold;
        font-size: 14px;
        text-align: right;
    }

    .prob-bar-bg {
        flex: 1;
        background: rgba(255,255,255,0.7);
        height: 22px;
        border-radius: 11px;
        overflow: hidden;
        position: relative;
    }

    .prob-bar {
        height: 100%;
        border-radius: 11px;
        transition: width 0.8s ease-out;
    }

    .prob-value {
        width: 65px;
        text-align: left;
        font-size: 14px;
        font-weight: bold;
    }

    .cleaned {
        margin-top: 20px;
        padding: 14px 18px;
        background: rgba(255,255,255,0.6);
        border-radius: 10px;
        font-size: 13px;
        color: #555;
        direction: rtl;
        text-align: right;
    }

    .cleaned b { color: #333; }

    .loading {
        display: none;
        text-align: center;
        padding: 25px;
    }

    .loading.show { display: block; }

    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #e0e0e0;
        border-top-color: #667eea;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin: 0 auto 15px;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .loading-text {
        color: #667eea;
        font-size: 15px;
    }

    .error {
        margin-top: 20px;
        padding: 15px 20px;
        background: #f8d7da;
        border-right: 4px solid #dc3545;
        border-radius: 10px;
        color: #721c24;
        display: none;
    }

    .error.show { display: block; }

    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #eee;
        color: #999;
        font-size: 13px;
    }

    .footer a {
        color: #667eea;
        text-decoration: none;
    }

    .footer a:hover { text-decoration: underline; }
</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Arabic Sentiment Analyzer</h1>
            <p class="subtitle">نظام تحليل مشاعر النصوص العربية</p>
            <span class="badge" id="modelBadge">جاري التحميل...</span>
        </div>

        <textarea
            id="text"
            placeholder="اكتب أو الصق النص العربي هنا...

مثال: هذا المنتج ممتاز جداً وأنا سعيد بشرائه"
        ></textarea>

        <button class="btn-analyze" id="analyzeBtn" onclick="analyze()">
            تحليل المشاعر
        </button>

        <div class="examples">
            <button onclick="setExample('هذا المنتج ممتاز جداً وأنا سعيد بشرائه')">
                إيجابي
            </button>
            <button onclick="setExample('الخدمة سيئة جداً ولا أنصح بها')">
                سلبي
            </button>
            <button onclick="setExample('المنتج عادي، لا شيء مميز فيه')">
                محايد
            </button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">جاري التحليل...</div>
        </div>

        <div class="error" id="error"></div>

        <div class="result" id="result"></div>

        <div class="footer">
            Powered by scikit-learn & Flask |
            <a href="https://github.com" target="_blank">GitHub</a>
        </div>
    </div>

    <script>
        function setExample(text) {
            document.getElementById('text').value = text;
            document.getElementById('text').focus();
            document.getElementById('error').classList.remove('show');
        }

        async function analyze() {
            const text = document.getElementById('text').value.trim();
            const errorDiv = document.getElementById('error');

            if (!text) {
                showError('الرجاء إدخال نص أولاً');
                return;
            }

            const btn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const resultDiv = document.getElementById('result');

            // Reset
            errorDiv.classList.remove('show');
            resultDiv.classList.remove('show');
            btn.disabled = true;
            btn.textContent = 'جاري التحليل...';
            loading.classList.add('show');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.error || 'فشل التحليل');
                }

                const data = await response.json();
                showResult(data);
            } catch (error) {
                showError(error.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'تحليل المشاعر';
                loading.classList.remove('show');
            }
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = 'خطأ: ' + message;
            errorDiv.classList.add('show');
        }

        function showResult(data) {
            const resultDiv = document.getElementById('result');
            const labelAr = data.label_ar;
            const confidence = data.confidence || 0;

            let cls = 'neutral';
            let emoji = '😐';
            let color = '#ffc107';

            if (labelAr.includes('إيجابي') || labelAr.includes('ايجابي')) {
                cls = 'positive';
                emoji = '😊';
                color = '#28a745';
            } else if (labelAr.includes('سلبي')) {
                cls = 'negative';
                emoji = '😞';
                color = '#dc3545';
            }

            let html = `
                <div class="emoji">${emoji}</div>
                <div class="label">${labelAr}</div>
                <div class="confidence">نسبة الثقة: <b>${(confidence * 100).toFixed(1)}%</b></div>
                <div class="progress-bg">
                    <div class="progress-bar" style="width: ${confidence * 100}%; background: ${color};"></div>
                </div>
            `;

            if (data.probabilities && data.probabilities.length === 3) {
                const classes = [
                    { name: 'سلبي', color: '#dc3545' },
                    { name: 'محايد', color: '#ffc107' },
                    { name: 'إيجابي', color: '#28a745' }
                ];

                html += '<div class="section-title">توزيع الاحتمالات:</div>';
                data.probabilities.forEach((prob, i) => {
                    const c = classes[i];
                    html += `
                        <div class="prob-row">
                            <span class="prob-label" style="color: ${c.color};">${c.name}</span>
                            <div class="prob-bar-bg">
                                <div class="prob-bar" style="width: ${prob * 100}%; background: ${c.color};"></div>
                            </div>
                            <span class="prob-value" style="color: ${c.color};">${(prob * 100).toFixed(1)}%</span>
                        </div>
                    `;
                });
            }

            if (data.cleaned) {
                html += `<div class="cleaned"><b>النص بعد المعالجة:</b> ${data.cleaned}</div>`;
            }

            resultDiv.className = 'result show ' + cls;
            resultDiv.innerHTML = html;

            // Scroll إلى النتيجة
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        // Ctrl+Enter للتحليل
        document.getElementById('text').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                analyze();
            }
        });

        // فحص حالة الخادم عند التحميل
        fetch('/health')
            .then(r => r.json())
            .then(data => {
                document.getElementById('modelBadge').textContent = 'النموذج جاهز';
            })
            .catch(() => {
                document.getElementById('modelBadge').textContent = 'الخادم غير متصل';
            });
    </script>
</body>
</html>
"""


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    """الصفحة الرئيسية."""
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict_route():
    """نقطة التنبؤ."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "لا توجد بيانات"}), 400

        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "نص فارغ"}), 400

        if len(text) > 5000:
            return jsonify({"error": "النص طويل جداً (الحد 5000 حرف)"}), 400

        result = predictor.predict(text)
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"خطأ في التحليل: {str(e)}"}), 500


@app.route("/health")
def health():
    """فحص حالة الخادم."""
    return jsonify({
        "status": "ok",
        "model": type(predictor.model).__name__,
    })


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("[2/2] Starting Flask server...")
    print()
    print("=" * 60)
    print("  Open your browser at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )