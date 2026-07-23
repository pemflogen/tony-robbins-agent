import os
import io
import logging
import voyageai
from pinecone import Pinecone
import anthropic
from flask import Flask, request, jsonify, send_from_directory, Response
from supabase import create_client
from pypdf import PdfReader
from docx import Document
import base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

voyage_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
index = pc.Index("tony-robbins-agent")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    supabase.table("conversations").select("id").limit(1).execute()
    logger.info("Supabase startup check OK - connected and 'conversations' table is reachable")
except Exception:
    logger.exception("Supabase startup check FAILED - cannot reach 'conversations' table. Check SUPABASE_URL/SUPABASE_KEY and table schema.")

APP_PASSWORD = os.environ.get("APP_PASSWORD", "Got1Robbinsagent!")
AGENT_ID = "tony-robbins"

SYSTEM_PROMPT = """You are an AI coach trained exclusively on Tony Robbins' mindset and peak performance methodology. You have deep knowledge of his frameworks on psychology, emotional mastery, decision-making, and personal transformation — including the Six Human Needs, the Triad (physiology, focus, language), Neuro-Associative Conditioning (NAC), the Dickens Process, Rapid Planning Method (RPM), and his teachings on beliefs, state management, and breakthrough.

You have two modes:

1. COACH MODE: Answer questions about Tony's frameworks, explain concepts, break down his models for change, and give tactical advice on mindset and peak performance. Always ground your answers in Tony's actual teachings. Be direct, high-energy, and practical.

2. BREAKTHROUGH MODE: Play the role of a strategic intervention coach in Tony's style — high intensity, pattern interrupts, bold reframes, and powerful questions that challenge the user's limiting beliefs. Push the user past their excuses, call out incongruence between what they say they want and how they're acting, and guide them toward an empowering decision. After each exchange, break character briefly to name one limiting belief or pattern you noticed and one shift to make. Then return to character immediately. No coddling.

If the user shares a screenshot or image (e.g. a journal entry, goal list, or note), read it carefully and coach them on their mindset, beliefs, and next decisive action using Tony's framework.

Always evaluate against Tony's actual framework. Change happens in an instant, the moment a real decision is made. Focus on state, story, and strategy — in that order. Identify the limiting belief beneath the surface problem. Push for massive action, not just insight."""

def get_relevant_context(query):
    embedding = voyage_client.embed([query], model="voyage-2").embeddings[0]
    results = index.query(vector=embedding, top_k=5, include_metadata=True)
    return "\n\n".join([m["metadata"]["text"] for m in results["matches"]])

def strip_images_from_history(history_messages):
    """Remove image content blocks from historical messages - only the current
    turn should carry images forward to the API. Prevents a bad or unsupported
    image from an earlier turn (e.g. pre-dating HEIC validation) from failing
    every subsequent request in the conversation."""
    cleaned = []
    for msg in history_messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_only = [part for part in content if part.get("type") != "image"]
            if not text_only:
                text_only = [{"type": "text", "text": "[image omitted]"}]
            cleaned.append({**msg, "content": text_only})
        else:
            cleaned.append(msg)
    return cleaned

@app.route("/")
def home():
    return send_from_directory(".", "ui.html")

@app.route("/verify-password", methods=["POST"])
def verify_password():
    data = request.json
    if data.get("password") == APP_PASSWORD:
        return jsonify({"success": True})
    return jsonify({"success": False}), 401

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    history = data.get("history", [])
    images = data.get("images", [])

    for img in images:
        media_type = img.get("media_type", "")
        if media_type not in ALLOWED_IMAGE_TYPES:
            return jsonify({
                "error": f"Unsupported image format ({media_type or 'unknown'}). "
                         f"Please use JPEG, PNG, GIF, or WebP - HEIC photos aren't supported."
            }), 400

    context = get_relevant_context(user_message) if user_message else ""

    if images:
        user_content = []
        if context:
            user_content.append({"type": "text", "text": f"Relevant Tony Robbins content:\n{context}\n\nUser message: {user_message}" if user_message else f"Relevant Tony Robbins content:\n{context}"})
        else:
            if user_message:
                user_content.append({"type": "text", "text": user_message})
        for img in images:
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("media_type", "image/jpeg"),
                    "data": img.get("data")
                }
            })
        if not user_message and not context:
            user_content.append({"type": "text", "text": "Please analyze this and coach me on my mindset, beliefs, and next decisive action."})
    else:
        user_content = f"Relevant Tony Robbins content:\n{context}\n\nUser message: {user_message}"

    messages = strip_images_from_history(history[:-1]) + [{"role": "user", "content": user_content}]

    def generate():
        try:
            with anthropic_client.messages.stream(
                model="claude-opus-4-5",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=messages
            ) as stream:
                for text_chunk in stream.text_stream:
                    yield text_chunk
        except Exception as e:
            yield f"\n\n[Error: {e}]"

    return Response(generate(), mimetype="text/plain")

ALLOWED_DOCUMENT_EXTENSIONS = {"pdf", "docx"}
MAX_DOCUMENT_SIZE_BYTES = 100 * 1024 * 1024  # 100MB

def extract_pdf_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()

def extract_docx_text(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs).strip()

@app.route("/upload-document", methods=["POST"])
def upload_document():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Please upload a PDF or .docx file."}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_DOCUMENT_SIZE_BYTES:
        return jsonify({"error": "Document is too large. Please upload a file under 100MB."}), 400

    try:
        text = extract_pdf_text(file_bytes) if ext == "pdf" else extract_docx_text(file_bytes)
    except Exception:
        logger.exception(f"Failed to extract text from uploaded document: {filename}")
        return jsonify({"error": "Failed to read document. It may be corrupted or password-protected."}), 400

    if not text:
        return jsonify({"error": "No extractable text found in this document."}), 400

    return jsonify({"filename": filename, "text": text})

@app.route("/conversations", methods=["GET"])
def get_conversations():
    # Deliberately excludes the "messages" column: it can hold megabytes of
    # base64 image data per conversation, and selecting it across every saved
    # conversation for a list view can blow the Postgres statement timeout.
    # The frontend doesn't render "preview" today, so it's kept as an empty
    # string for API-shape compatibility rather than dropped.
    try:
        result = supabase.table("conversations").select("id,team_member,title,created_at").eq("agent_id", AGENT_ID).order("created_at", desc=True).execute()
    except Exception:
        logger.exception("Supabase select with agent_id filter failed, falling back to unfiltered select")
        try:
            result = supabase.table("conversations").select("id,team_member,title,created_at").order("created_at", desc=True).execute()
        except Exception:
            logger.exception("Supabase select (fallback) failed - unable to load conversations")
            return jsonify({"error": "Failed to load conversations"}), 500
    convs = [
        {
            "id": row["id"],
            "team_member": row["team_member"],
            "title": row["title"],
            "preview": "",
            "created_at": row["created_at"],
        }
        for row in result.data
    ]
    return jsonify(convs)

@app.route("/conversations", methods=["POST"])
def save_conversation():
    data = request.json
    row = {
        "team_member": data.get("team_member"),
        "title": data.get("title"),
        "messages": data.get("messages"),
    }
    try:
        result = supabase.table("conversations").insert({**row, "agent_id": AGENT_ID}).execute()
    except Exception:
        logger.exception("Supabase insert with agent_id failed, falling back to insert without agent_id")
        try:
            result = supabase.table("conversations").insert(row).execute()
        except Exception:
            logger.exception("Supabase insert (fallback) failed - conversation was NOT saved")
            return jsonify({"error": "Failed to save conversation"}), 500
    return jsonify(result.data[0])

@app.route("/conversations/<int:conv_id>", methods=["GET"])
def get_conversation(conv_id):
    try:
        result = supabase.table("conversations").select("*").eq("id", conv_id).execute()
    except Exception:
        logger.exception(f"Supabase select failed for conversation id={conv_id}")
        return jsonify({"error": "Failed to load conversation"}), 500
    if result.data:
        return jsonify(result.data[0])
    return jsonify({"error": "Not found"}), 404

@app.route("/conversations/<int:conv_id>", methods=["PATCH"])
def rename_conversation(conv_id):
    data = request.json
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400
    try:
        supabase.table("conversations").update({"title": title}).eq("id", conv_id).execute()
    except Exception:
        logger.exception(f"Supabase update (rename) failed for conversation id={conv_id}")
        return jsonify({"error": "Failed to rename conversation"}), 500
    return jsonify({"success": True})

@app.route("/conversations/<int:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    try:
        supabase.table("conversations").delete().eq("id", conv_id).execute()
    except Exception:
        logger.exception(f"Supabase delete failed for conversation id={conv_id}")
        return jsonify({"error": "Failed to delete conversation"}), 500
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
