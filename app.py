import os
import voyageai
from pinecone import Pinecone
import anthropic
from flask import Flask, request, jsonify, send_from_directory, Response
from supabase import create_client
import base64

app = Flask(__name__)

voyage_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
index = pc.Index("tony-robbins-agent")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

@app.route("/")
def home():
    return send_from_directory(".", "ui.html")

@app.route("/verify-password", methods=["POST"])
def verify_password():
    data = request.json
    if data.get("password") == APP_PASSWORD:
        return jsonify({"success": True})
    return jsonify({"success": False}), 401

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    history = data.get("history", [])
    images = data.get("images", [])

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

    messages = history[:-1] + [{"role": "user", "content": user_content}]

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

@app.route("/conversations", methods=["GET"])
def get_conversations():
    try:
        result = supabase.table("conversations").select("id,team_member,title,messages,created_at").eq("agent_id", AGENT_ID).order("created_at", desc=True).execute()
    except Exception:
        result = supabase.table("conversations").select("id,team_member,title,messages,created_at").order("created_at", desc=True).execute()
    convs = []
    for row in result.data:
        preview = ""
        messages = row.get("messages") or []
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    preview = content[:60]
                elif isinstance(content, list):
                    text_part = next((p for p in content if p.get("type") == "text"), None)
                    if text_part:
                        preview = text_part.get("text", "")[:60]
                break
        convs.append({
            "id": row["id"],
            "team_member": row["team_member"],
            "title": row["title"],
            "preview": preview,
            "created_at": row["created_at"],
        })
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
        result = supabase.table("conversations").insert(row).execute()
    return jsonify(result.data[0])

@app.route("/conversations/<int:conv_id>", methods=["GET"])
def get_conversation(conv_id):
    result = supabase.table("conversations").select("*").eq("id", conv_id).execute()
    if result.data:
        return jsonify(result.data[0])
    return jsonify({"error": "Not found"}), 404

@app.route("/conversations/<int:conv_id>", methods=["PATCH"])
def rename_conversation(conv_id):
    data = request.json
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400
    supabase.table("conversations").update({"title": title}).eq("id", conv_id).execute()
    return jsonify({"success": True})

@app.route("/conversations/<int:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    supabase.table("conversations").delete().eq("id", conv_id).execute()
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
