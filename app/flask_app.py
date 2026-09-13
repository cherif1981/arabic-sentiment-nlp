"""
تطبيق Flask لتحليل المشاعر العربية
تشغيل:
    python app/flask_app.py
ثم افتح: http://127.0.0.1:5000
"""
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from flask import Flask, render_template, request, jsonify
from predict import SentimentPredictor


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


# نحدد مجلد templates و static داخل app/
app = Flask(
    __name__,
    template_folder=os.path.join(CURRENT_DIR, "templates"),
    static_folder=os.path.join(CURRENT_DIR, "static"),
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
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
    return jsonify({
        "status": "ok",
        "model": type(predictor.model).__name__,
    })


if __name__ == "__main__":
    print("[2/2] Starting Flask server...")
    print()
    print("=" * 60)
    print("  Open your browser at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)