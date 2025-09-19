# Fatan – Local AI Assistant (MVP)
> Updated: 2025-09-19

**Fatan** هو مساعد ذكي **محلي بالكامل** للموظفين. يوفر شات مدعوم بنموذج لغوي محلي (Ollama) مع **استرجاع معرفي RAG** من مستندات الشركة المخزّنة محليًا، بالإضافة إلى **لوحة مهام** لإدارة الأعمال اليومية. لا يحتاج لاتصال إنترنت ما لم تختَر تفعيل البحث على الويب.

---

## ✨ المزايا الرئيسية
- **تشغيل محلي 100%**: LLM عبر **Ollama** + **ChromaDB** للفهرسة، دون إرسال بيانات للخارج.
- **RAG من مستنداتك**: ضع ملفاتك في `docs/` ثم شغّل `analyzer.py` لبناء فهرس **Chroma**.
- **شات للموظفين**: واجهة بسيطة عبر Flask (مسار `/chat`) مع حفظ تاريخ المحادثات اختياريًا.
- **مهام وخطوات**: واجهة مهام (`tasks_app.py`) مع حالات (جديدة/قيد التنفيذ/مكتملة) وتحديث فوري.
- **تحليل اجتماعات (اختياري)**: توليد تقارير نصية في `generated_meeting_reports/`.
- **بحث ويب اختياري**: عبر Tavily عند عدم توفر معرفة محلية (يتطلب مفتاح API).

---

## 📦 هيكل المشروع
- app.py — واجهة الشات (Flask) على المنفذ 5000
- tasks_app.py — واجهة المهام (Flask) على المنفذ 8000
- analyzer.py — بناء فهرس محلي (ChromaDB) من مجلد docs/
- agent.py — استخراج مهام من محاضر الاجتماعات وحفظها في tasks.json (اختياري)
- recoder.py — صفحة تسجيل/توليد تقارير اجتماعات (اختياري)
- templates/ — ملفات HTML (index.html, tasks_index.html, meetings_analyzer.html)
- static/ — شعار وصور ثابتة
- docs/ — ملفات المستندات التي تُفهرس محليًا
- db_store/ — قاعدة بيانات Chroma (تُنشأ بعد تشغيل analyzer.py)
- generated_meeting_reports/ — تقارير الاجتماعات (txt)
- tasks.json — مصدر بيانات المهام (active/completed)
- conversation_history.json — سجل محادثات الشات (اختياري)

---

## 🧰 المتطلبات (Requirements)

### النظام والأدوات
- **Python 3.10+** (مُفضَّل 3.11 أو 3.12)
- **Ollama** مُثبّت ويعمل محليًا
  - تنزيل النموذج المستخدم في الشات:  
    ```bash
    ollama pull llama3.2:3b-instruct-fp16
    ```
- (اختياري) تمكين البحث على الويب عبر **Tavily** بوضع المتغير:
  - **Linux/macOS**:
    ```bash
    export TAVILY_API_KEY=YOUR_KEY
    ```
  - **Windows (PowerShell)**:
    ```powershell
    setx TAVILY_API_KEY "YOUR_KEY"
    ```

### بايثون (مكتبات)
انظر ملف `requirements.txt` المرفق، أو ثبّت مباشرة:
```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

**المكتبات الأساسية** (ملخّص):
- `flask` — الواجهات
- `langchain`, `langchain-community`, `langchain-core` — السلسلة المنطقية
- `langchain-ollama` — التكامل مع Ollama (LLM محلي)
- `langchain-nomic` + `nomic` + `onnxruntime` — **تضمينات Nomic** محليًا
- `chromadb` — مخزن المتجهات (محلي)
- `tiktoken` — ترميز النصوص

> **ملاحظة**: إذا واجهت مشكلة في **NomicEmbeddings**، تأكّد من تثبيت `onnxruntime` ومن أن خيار `inference_mode="local"` مدعوم في بيئتك. بديلًا، يمكنك تبديل التضمينات إلى `OllamaEmbeddings` من `langchain_ollama` إن رغبت.

---

## 🚀 التشغيل المحلي (بدون إنترنت)

1) **تثبيت المتطلبات** (أول مرة فقط):  
```bash
python -m venv .venv
source .venv/bin/activate  # أو: .venv\Scripts\Activate.ps1 على ويندوز
pip install -r requirements.txt
```

2) **تشغيل Ollama** وسحب النموذج:
```bash
ollama serve  # إن لم يعمل تلقائيًا
ollama pull llama3.2:3b-instruct-fp16
```

3) **إضافة المستندات** إلى `docs/` ثم بناء الفهرس:
```bash
python analyzer.py
# سيُنشئ/يحدّث مجلد db_store/ (ChromaDB)
```

4) **تشغيل واجهة الشات** (منفذ 5000):
```bash
python app.py
# افتح المتصفح على: http://localhost:5000
```

5) **تشغيل واجهة المهام** (منفذ 8000):
```bash
python tasks_app.py
# افتح: http://localhost:8000
```
> ملف `tasks.json` هو مصدر البيانات. يضم قوائم `active` و`completed`. كل مهمة تحتوي `title` و`steps` (لكل خطوة `description` و`done`).

6) (اختياري) **تحليل الاجتماعات/التسجيل**:
- `recoder.py` يولّد تقارير نصيّة في `generated_meeting_reports/`.
- `agent.py` يحاول استخراج مهام من نصوص الاجتماعات وإضافتها إلى `tasks.json`.

---

## 🔌 نقاط الـ API (مختصر)
- **POST** `/chat` — جسد الطلب: `{ "message": "..." }` → الرد: `{ "answer": "..." }`
- **GET** `/api/tasks` — يعيد بنية المهام (active/completed)
- **POST** `/api/complete_step` — جسم الطلب: `{ "task_index": 0, "step_index": 2 }`

> المسارات تُقدَّم من تطبيقات Flask: `app.py` (الشات) و`tasks_app.py` (المهام).

---

## 🧪 اختبار سريع
- ضع ملفًا تجريبيًا (txt/pdf/docx) داخل `docs/` (نماذج مرفقة بالفعل).
- شغّل `python analyzer.py` ثم افتح الشات واسأل عن محتوى المستند.
- جرّب واجهة المهام: وضع علامة «تم» على خطوة، وتأكد من انتقال المهمة لقائمة المكتملة عند إنهاء جميع خطواتها.

---

## 🛠️ مشاكل شائعة (Troubleshooting)
- **Model not found**: نفّذ `ollama pull llama3.2:3b-instruct-fp16` وتأكد أن `ollama serve` يعمل.
- **ImportError: langchain_*:** حدّث `pip` وثبّت الحزم المذكورة في `requirements.txt`.
- **NomicEmbeddings (local) fails**: ثبّت `onnxruntime` أو بدّل إلى تضمينات أخرى (مثال: `OllamaEmbeddings`).
- **ChromaDB lock**: إذا تعذّر الوصول لقاعدة `db_store/`، أغلق العمليات السابقة أو احذف المجلد وأعد بناءه بـ `analyzer.py`.
- **بدون إنترنت**: اترك `TAVILY_API_KEY` غير مخصص؛ سيعتمد الشات على الفهرس المحلي فقط.

---

## 🧩 المكوّنات الرئيسية (نظرة تقنية)
- **LLM**: ChatOllama (نموذج: `llama3.2:3b-instruct-fp16`)
- **Embeddings**: Nomic (`nomic-embed-text-v1.5`, وضع محلي)
- **Vector DB**: Chroma (مجلد `db_store/`)
- **Orchestration**: LangChain (Router → Retriever → Graders → Generator)
- **Frontend**: قوالب HTML بسيطة + Fetch API
- **Backend**: Flask (منفذ 5000 و8000)

---

## 📄 الرخصة والاستخدام
هذا **MVP للاختبارات الداخلية**. البيانات تبقى محليًا. لا ينصح للنشر العام بدون مراجعة أمنية.

---

## 🇬🇧 English (Brief)
**Fatan** is a fully local employee assistant: Flask chat UI backed by **Ollama** + **ChromaDB** (RAG), and a tasks dashboard. Optional web search via **Tavily**.

**Quick start:**
1. `pip install -r requirements.txt`
2. `ollama pull llama3.2:3b-instruct-fp16`
3. Put docs in `docs/` → `python analyzer.py`
4. `python app.py` (chat on http://localhost:5000)
5. `python tasks_app.py` (tasks on http://localhost:8000)

**Optional:** set `TAVILY_API_KEY` to enable web search.
