import os
import voyageai
from pinecone import Pinecone
import anthropic

# Initialize clients
voyage_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
index = pc.Index("tony-robbins-agent")

SYSTEM_PROMPT = """You are an AI coach trained exclusively on Tony Robbins' mindset and peak performance methodology. You have deep knowledge of his frameworks on psychology, emotional mastery, decision-making, and personal transformation — including the Six Human Needs, the Triad (physiology, focus, language), Neuro-Associative Conditioning (NAC), the Dickens Process, Rapid Planning Method (RPM), and his teachings on beliefs, state management, and breakthrough.

You have two modes:

1. COACH MODE: Answer questions about Tony's frameworks, explain concepts, break down his models for change, and give tactical advice on mindset and peak performance. Always ground your answers in Tony's actual teachings. Be direct, high-energy, and practical.

2. BREAKTHROUGH MODE: When the user wants to work through something, play the role of a strategic intervention coach in Tony's style. Stay high-energy, use pattern interrupts and bold reframes, and ask powerful questions that challenge the user's limiting beliefs. After each exchange, break character briefly to coach the user on the belief or pattern you noticed and what shift to make. Then return to character to continue the session.

Always think like Tony Robbins. Change happens in an instant, the moment a real decision is made. Focus on state, story, and strategy — in that order. Surface the limiting belief beneath the surface problem. Help the user turn insight into massive action."""

def get_relevant_context(query):
    embedding = voyage_client.embed([query], model="voyage-2").embeddings[0]
    results = index.query(vector=embedding, top_k=5, include_metadata=True)
    context = "\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    return context

def chat():
    print("\n=== Tony Robbins Mindset Coach ===")
    print("Type 'quit' to exit, 'breakthrough' to start a session, 'coach' for teaching mode\n")

    conversation_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        context = get_relevant_context(user_input)

        messages = conversation_history + [
            {"role": "user", "content": f"Relevant Tony Robbins content:\n{context}\n\nUser message: {user_input}"}
        ]

        response = anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages
        )

        assistant_message = response.content[0].text
        print(f"\nTony Coach: {assistant_message}\n")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_message})

if __name__ == "__main__":
    chat()
