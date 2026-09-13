import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import clean_text

cases = [
    "هاييييلة بزاف هذا المنتج",
    "المنتج ممتاز بصح الخدمة سيئة",
    "الخدمة سيئة للغاية @support",
    "هاييييلة بزاف! بصح غالية شوية 😅",
]

out_path = os.path.join(os.path.dirname(__file__), "debug_failures.txt")
with open(out_path, "w", encoding="utf-8") as f:
    for text in cases:
        result = clean_text(text)
        f.write(f"INPUT : {text}\n")
        f.write(f"OUTPUT: {result}\n")
        f.write(f"IN_LEN : {len(text)}\n")
        f.write(f"OUT_LEN: {len(result)}\n")
        # تحقق من كل كلمة متوقعة
        for word in ["هايلة", "بزاف", "ممتاز", "بصح", "سيئة", "سيئه", "سيي", "سيئ", "غالية", "شوية"]:
            mark = "✓" if word in result else "✗"
            f.write(f"  {mark} '{word}'\n")
        f.write("-" * 70 + "\n")

print(f"wrote {out_path}")