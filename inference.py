import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time

# --- CẤU HÌNH ---
MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
# Nếu bạn để model ở ổ D thì sửa lại đường dẫn cache_dir nhé
CACHE_DIR = None 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="AI Waifu API")

# --- BIẾN TOÀN CỤC ĐỂ CHỨA MODEL ---
model = None
tokenizer = None

# --- MODEL REQUEST ---
class ChatRequest(BaseModel):
    messages: list # Dạng: [{"role": "user", "content": "hi"}]
    max_tokens: int = 512
    temperature: float = 0.7

@app.on_event("startup")
async def startup_event():
    """Hàm này chạy 1 lần duy nhất khi bật server để load model"""
    global model, tokenizer
    print(f"--- Đang tải Model {MODEL_ID} trên {DEVICE}... ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        print(">> ✅ AI WAIFU SERVER ĐÃ SẴN SÀNG!")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")

@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatRequest):
    global model, tokenizer
    
    # 1. Định dạng Prompt chuẩn Llama 3 (Quan trọng!)
    # Thêm System Prompt mặc định nếu chưa có
    if request.messages[0]["role"] != "system":
        system_msg = {
            "role": "system", 
            "content": "Bạn là Waifu, cô gái Việt Nam đáng yêu. Luôn trả lời bằng tiếng Việt."
        }
        request.messages.insert(0, system_msg)

    # Dùng hàm apply_chat_template có sẵn của Tokenizer
    input_ids = tokenizer.apply_chat_template(
        request.messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    # 2. Sinh câu trả lời
    try:
        outputs = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=request.temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 3. Giải mã
        response_text = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        return {
            "id": "chatcmpl-" + str(int(time.time())),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Chạy Server ở cổng 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)