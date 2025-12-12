import time
import orjson
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# --- 1. CẤU HÌNH ENGINE (GIỮ NGUYÊN TỐI ƯU) ---
MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

engine_args = AsyncEngineArgs(
    model=MODEL_ID,
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    enforce_eager=False,
    enable_prefix_caching=True,
    disable_log_stats=True
)

print("⏳ Init vLLM Engine...")
engine = AsyncLLMEngine.from_engine_args(engine_args)
print("✅ Ready!")

# --- 2. FASTAPI SETUP & OPTIMIZED JSON ---

class ORJSONResponse(JSONResponse):
    """Class này giúp return JSON nhanh hơn mặc định của FastAPI"""
    media_type = "application/json"
    def render(self, content: any) -> bytes:
        return orjson.dumps(content)

app = FastAPI(default_response_class=ORJSONResponse)

# Data Model (Đã bỏ field 'stream')
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=1024)
    temperature: Optional[float] = Field(default=0.7)

SYSTEM_PROMPT = "You are a helpful AI assistant."

# Warmup để tránh chậm request đầu
@app.on_event("startup")
async def startup_event():
    dummy_sampling = SamplingParams(temperature=0, max_tokens=1)
    req_id = random_uuid()
    try:
        async for _ in engine.generate(SYSTEM_PROMPT, dummy_sampling, req_id):
            pass
    except: pass

# --- 3. API ENDPOINT (FULL RESPONSE) ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    request_id = random_uuid()
    created_time = int(time.time())

    # Xử lý Prompt
    prompt = f"{SYSTEM_PROMPT}\n"
    for msg in request.messages:
        prompt += f"{msg.role.upper()}: {msg.content}\n"
    prompt += "ASSISTANT:"

    # Params
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=["</s>", "USER:", "user:"],
    )

    # --- LOGIC QUAN TRỌNG: CHỜ KẾT QUẢ CUỐI CÙNG ---
    final_output = None
    
    # Engine của vLLM bản chất là async generator, ta phải loop để lấy kết quả cuối
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        final_output = request_output
    
    # Lấy text hoàn chỉnh từ kết quả cuối cùng
    text_output = final_output.outputs[0].text

    # Trả về JSON một lần duy nhất
    return ORJSONResponse({
        "id": request_id,
        "object": "chat.completion",
        "created": created_time,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": text_output
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": len(final_output.outputs[0].token_ids),
            "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids)
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="uvloop")