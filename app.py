import os
import uuid
import markdown
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.file_processing import extract_text_from_file
from utils.vector_store import store_to_chroma, query_chroma

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev")

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'heic'}
MAX_PAGES = 100

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('documents')
        total_pages = 0
        extracted_texts = []
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)

        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(session_folder, filename)
                file.save(path)
                print(f"📥 Saved: {filename}")

                try:
                    text, pages = extract_text_from_file(path)
                    print(f"📄 {filename} — {pages} pages, {len(text)} characters")

                    if pages == 0 or not text.strip():
                        print(f"⚠️ Skipping {filename}: No readable content.")
                        continue  # Skip this file

                    total_pages += pages
                    if total_pages > MAX_PAGES:
                        return render_template("index.html", error=f"⛔ Upload exceeds 100 pages limit.")
                    extracted_texts.append((filename, text))

                except Exception as e:
                    print(f"❌ Error reading {filename}: {e}")
                    return render_template("index.html", error=f"❌ Error reading {filename}: {e}")

        combined_text = "\n\n".join([f"{name}:\n{text}" for name, text in extracted_texts])
        store_to_chroma(session_id, combined_text)

        return render_template("index.html", success="✅ Files uploaded successfully!", chat=True, session_id=session_id)

    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    session_id = data.get("session_id")
    question = data.get("question")

    if not session_id or not question:
        return jsonify({"response": "Invalid session or question."})

    context = query_chroma(session_id, question)
    doc_count = context.count(":\n")  # crude but works if each file is prefixed like 'filename:\ntext'
    prompt = (
        f"You are a professional document assistant. Based only on the given context below from {doc_count} document(s), "
        f"answer the question clearly. Format your answer using markdown.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    try:
        response = llm.invoke(prompt)
        return jsonify({"response": markdown.markdown(response.content.strip())})
    except Exception as e:
        return jsonify({"response": f"Gemini API error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)