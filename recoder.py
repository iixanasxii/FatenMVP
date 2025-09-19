"""
app.py

Flask app with:
- The same design (gradient background, logo + title in header).
- A fake record button that toggles after 6 seconds to generate a .txt file.

Usage:
  python app.py
Then open http://localhost:5000
"""

import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder="static")

# Ensure the folder for meeting reports exists
REPORTS_FOLDER = "generated_meeting_reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True)

@app.route("/")
def index():
    """Render the 'Meetings Analyzer' page."""
    return render_template("meetings_analyzer.html")

@app.route("/generate_report", methods=["POST"])
def generate_report():
    """
    After a 6-second wait on the client side,
    create a .txt file with 'the meeting report' in 'generated_meeting_reports/'.
    """
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{now_str}.txt"
    file_path = os.path.join(REPORTS_FOLDER, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("""مقدمة الاجتماع:
تم عقد اجتماع لعرض مشروع فطن، وهو نظام إدارة معرفي ذكي يهدف إلى تحسين الكفاءة الإدارية في الشركات من خلال أتمتة المهام وتقديم حلول شخصية للموظفين بناءً على تحليل البيانات.

أبرز النقاط التي تم مناقشتها في العرض:
المشكلة المطروحة:

بطء إنجاز المهام الإدارية والتقنية.
الاعتماد الكبير على الموظفين في نقل المعرفة.
الوقت الضائع في إعداد المستندات والاجتماعات.
صعوبة تتبع المهام مما يؤثر سلبًا على الإنتاجية والتكاليف التشغيلية.
الحل المقترح (مشروع فطن):

نظام إدارة معرفي ذكي يساعد على أتمتة المهام.
يقدم فطن مساعد شخصي للموظفين يحلل البيانات ويقدم حلولاً مخصصة لتحسين الكفاءة.
القيمة المضافة:

تقليل الوقت والجهد المبذول في المهام اليومية.
تعزيز دقة البيانات المقدمة.
تحسين تجربة الموظفين في بيئة العمل.
العملاء المستهدفون:

الشركات التي تسعى لتحسين الكفاءة التشغيلية وتقليل التكاليف.
تميّز فطن عن المنافسين:

يعمل على سيرفر داخلي لضمان الخصوصية.
يعتمد على تحليل البيانات لتقديم حلول مخصصة.
سهولة الاستخدام وأتمتة المهام اليومية.
النموذج الربحي:
رسوم تركيب النظام: تبدأ من 6000 ريال.
رسوم اشتراك شهرية: 75 ريال لكل مستخدم.
مصادر الإيرادات: رسوم تركيب النظام والاشتراكات الشهرية.
الفريق المشارك في المشروع:
أنس الغامدي: مسؤول الذكاء الاصطناعي وتحليل البيانات.
لمير كردي: محللة أمن سيبراني متخصصة في إدارة الامتثال الأمني والتحقيقات الرقمية.""")

    return jsonify({"status": "ok", "message": "Report generated", "filename": filename})

if __name__ == "__main__":
    # You can change port or remove debug=True if desired
    app.run(host="0.0.0.0", port=5000, debug=True)
