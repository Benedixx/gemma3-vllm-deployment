import uuid
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.service.gemma3_service import generate_chat_stream, generate_chat_once, init_engine

app = FastAPI()
engine = None

class Message(BaseModel):
    role: str
    content: str

class RequestPayload(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False

@app.on_event("startup")
async def preload_model():
    global engine
    try:
        print("Loading Gemma engine...")
        engine = init_engine()
        print("Gemma engine ready!")
    except Exception as e:
        print(f"Warm-up failed: {e}")

@app.post("/chat")
async def chat(payload: RequestPayload):
    try:
        messages = [m.dict() for m in payload.messages]

        if payload.stream:
            return StreamingResponse(
                generate_chat_stream(engine, messages, payload.temperature, payload.max_tokens),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            result = await generate_chat_once(
                engine=engine,
                messages=messages,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            )

            return {
                "id": f"marshall_chat-{uuid.uuid4()}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result},
                        "finish_reason": "stop",
                    }
                ],
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
