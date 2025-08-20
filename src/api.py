from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from pathlib import Path
from typing import List
from model_loader import stream_model_response, cancel_generation, reset_generation, prepare_models, get_model_from_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager





@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up API, preloading models...")
    prepare_models()
    yield  
    print("Shutting down API, cleaning up...")

app = FastAPI(lifespan=lifespan)

def sse_event(data: str) -> str:
    return f"data: {data}\n\n"

origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define base paths
base_path = Path(__file__).resolve().parent / "../../Models/Models_for_testing_app"
base_path_tokenizer = Path(__file__).resolve().parent / "../../Tokenizers/Tokenizers_for_testing_app"

# Request model to match C# JSON
class PromptRequest(BaseModel):
    modelName: str
    promptBody: str

# Helper: get list of models
def modelRepository() -> List[str]:
    models = []
    if base_path.exists() and base_path.is_dir():
        for model_dir in base_path.iterdir():
            if model_dir.is_dir():
                    models.append(model_dir.name)
    print("Discovered models:", models)
    return models

# Helper: find model path
def find_model_path(model_name: str):
    if not base_path.exists():
        return None
    model_path = base_path / model_name
    return model_path if model_path.exists() else None

# Helper: find tokenizer path
def find_tokenizer_path(model_name: str):
    if not base_path_tokenizer.exists():
        return None
    tokenizer_path = base_path_tokenizer / model_name
    return tokenizer_path if tokenizer_path.exists() else None

# GET: list available models
@app.get("/AIPlanat/getModelsRepository")
async def get_models_repository():
    models = modelRepository()
    return {"models": models}


# POST: use a model to generate response
@app.post("/AIPlanat/getModelResponse")
async def post_use_model(request: PromptRequest):
    prompt = request.promptBody
    model_name = request.modelName

    model, tokenizer = get_model_from_cache(request.modelName)
    
    reset_generation()
    
    def event_stream():
        yield sse_event(f'{{"event":"start","model":"{model_name}"}}')

        for chunk in stream_model_response(model, tokenizer, prompt):
            yield sse_event(chunk)

        yield sse_event('{"event":"end"}')

    return StreamingResponse(event_stream(), media_type="text/event-stream")
    


