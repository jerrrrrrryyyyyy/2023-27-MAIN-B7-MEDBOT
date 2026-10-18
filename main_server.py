import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_history = [
    {"role": "system", "content": "You are a helpful health information assistant. Keep answers short and clear."}
]


class ChatMessage(BaseModel):
    message: str


@app.post("/chat")
def chat(chat_message: ChatMessage):
    conversation_history.append({"role": "user", "content": chat_message.message})

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # fast + generous free-tier limits, good for prototyping
        messages=conversation_history,
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})

    return {"reply": reply}


@app.get("/")
def health_check():
    return {"status": "Server is running"}