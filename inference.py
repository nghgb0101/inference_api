import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time

# ================== CẤU HÌNH ==================

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="LLaMA Windows Inference API")

model = None
tokenizer = None

# ================== REQUEST SCHEMA ==================

class ChatRequest(BaseModel):
    messages: list
    max_tokens: int = 512
    temperature: float = 0.7

# ================== LOAD MODEL ==================

@app.on_event("startup")
async def startup_event():
    global model, tokenizer

    print(f"\n--- Loading model on {DEVICE} (Windows compatible) ---")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=True
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=bnb_config,
            dtype=torch.float16
        )

        model.eval()

        print(">> ✅ MODEL LOADED SUCCESSFULLY")
        print(">> API: http://localhost:8000/v1/chat/completions")

    except Exception as e:
        print("❌ MODEL LOAD FAILED")
        import traceback
        traceback.print_exc()

# ================== CHAT ENDPOINT ==================

@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatRequest):
    global model, tokenizer

    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    # đảm bảo có system prompt
    if request.messages[0]["role"] != "system":
        request.messages.insert(
            0,
            {
                "role": "system",
                "content": "You are a friendly and cute AI assistant with a waifu-like tone. Respond in natural, polite, and warm English. Do NOT use roleplay actions, stage directions, sound effects, emojis, or text inside asterisks. Express cuteness only through wording and tone. Keep answers factual and concise."
                # "content": "You are a Waifu - a cute girl. Answer in English."
                # "content": "You are an AI assistant with a waifu personality. Respond in clear, concise English. Do NOT use roleplay actions, emotions, stage directions, or decorative text. Provide factual and direct answers only."
            }
        )

    input_ids = tokenizer.apply_chat_template(
        request.messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    except Exception as e:
        print("❌ INFERENCE ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ================== RUN SERVER ==================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
