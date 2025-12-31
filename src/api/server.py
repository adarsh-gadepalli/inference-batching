from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.core.dynamic_batcher import DynamicBatcher
from src.core.continuous_batcher import ContinuousBatcher
from src.models.nlp import NLPModel
from src.models.gen import GenerativeModel
import uvicorn
import os
import asyncio
import torch

# config
# BATCHING_TYPE: "NONE", "DYNAMIC", "CONTINUOUS"
BATCHING_TYPE = os.getenv("BATCHING_TYPE", "DYNAMIC").upper()
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
MAX_LATENCY_MS = float(os.getenv("MAX_LATENCY_MS", "10.0"))

MODEL_NAME = "gpt2" 

batcher = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher, model
    print(f"initializing server with batching_type={BATCHING_TYPE}...")
    
    model = GenerativeModel(model_name=MODEL_NAME)
    model.load()

    if BATCHING_TYPE == "CONTINUOUS":
        print("continuous batching enabled")
        batcher = ContinuousBatcher(model, max_batch_size=MAX_BATCH_SIZE)
        await batcher.start()
        
    elif BATCHING_TYPE == "DYNAMIC":
        print("dynamic batching enabled")

        batcher = DynamicBatcher(model, max_batch_size=MAX_BATCH_SIZE, max_latency_ms=MAX_LATENCY_MS)
        await batcher.start()
        
    else: 
        print("batching disabled (direct inference)")
        batcher = None
    
    yield
    
    print("shutting down...")
    if batcher:
        await batcher.stop()
        

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    result: str 
    # optional field for metrics
    padding_waste: float = 0.0
    
app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        if BATCHING_TYPE in ["DYNAMIC", "CONTINUOUS"]:
            if not batcher:
                raise HTTPException(status_code=503, detail="batcher not initialized")
            
            # tokenize here to unblock batcher loop
            # keep cpu work out of the consumer loop
            if model.tokenizer:
                input_ids = model.tokenizer.encode(request.text, return_tensors="pt").to(model.device).squeeze(0)
                # pass tensor (1d) to batcher
                result_obj = await batcher.predict(input_ids)
            else:
                 # fallback if no tokenizer
                result_obj = await batcher.predict(request.text)

            
            if isinstance(result_obj, tuple):
                return PredictResponse(result=str(result_obj[0]), padding_waste=result_obj[1])
            else:
                return PredictResponse(result=str(result_obj))
            
        else:
            # no batching
            if not model:
                raise HTTPException(status_code=503, detail="model not initialized")
            
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, model.predict, [request.text])
            # no padding waste in batch size 1 (technically 0%, irrelevant)
            return PredictResponse(result=results[0], padding_waste=0.0)
            
    except Exception as e:
        print(f"error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
