import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel

# Import 2-stage retrieval layer (BGE-M3 candidate search + BGE-Reranker + Redis hydration)
from retrieval import retrieve

load_dotenv()

app = FastAPI(title="Medical RAG Assistant API")

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable missing.")

groq_client = Groq(api_key=groq_api_key)

SYSTEM_PROMPT = (
    "You are a clinical health assistant. Answer the user's question accurately "
    "and concisely based strictly on the provided medical context. If context is provided, "
    "rely on it primarily. If the information is not present in the context, clearly "
    "state that you do not have enough verified information to answer."
)


class ChatMessage(BaseModel):
    message: str


@app.post("/chat")
def chat(chat_message: ChatMessage):
    user_query = chat_message.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # 1. Execute 2-stage retrieval (Fetch 10 vector candidates -> Re-rank to top 3 -> Hydrate via Redis)
    retrieved_chunks = retrieve(query=user_query, fetch_k=10, final_k=3)

    # 2. Combine top re-ranked chunks into a single clean context block
    if retrieved_chunks:
        context_str = "\n\n---\n\n".join([chunk.text for chunk in retrieved_chunks])
        augmented_user_message = (
            f"Use the following medical context to answer the question.\n\n"
            f"--- RELEVANT CONTEXT ---\n{context_str}\n\n"
            f"--- USER QUESTION ---\n{user_query}"
        )
    else:
        augmented_user_message = user_query

    # 3. Formulate message payload for LLM inference
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": augmented_user_message},
    ]

    # 4. Generate LLM response using Groq (llama3-8b-8192)
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation Error: {str(e)}")

    return {
        "reply": reply,
        "context_retrieved": len(retrieved_chunks) > 0,
        "retrieved_chunks": [
            {
                "id": c.id,
                "section": f"{c.section_title} > {c.subsection_title}",
                "rerank_score": round(c.similarity, 4),
            }
            for c in retrieved_chunks
        ],
    }


@app.get("/")
def health_check():
    return {"status": "Medical RAG API running smoothly"}