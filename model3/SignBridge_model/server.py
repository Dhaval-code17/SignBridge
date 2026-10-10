from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn

# 1. Initialize FastAPI App
app = FastAPI(
    title="SignBridge Model 3 API",
    description="Translates sign language sequence glosses into natural English sentences using fine-tuned T5.",
    version="1.0.0"
)

# 2. Enable CORS (Allows PWA / Frontend to call this API without cross-origin issues)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Trained Model & Tokenizer
MODEL_PATH = "./model3_final"

try:
    print("⏳ Loading Model 3 from local path...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    print("✅ Model 3 successfully loaded into memory!")
except Exception as e:
    print(f"❌ Error loading model artifacts from {MODEL_PATH}: {e}")
    model = None
    tokenizer = None

# 4. Request & Response Schemas
class TranslationRequest(BaseModel):
    sequence: str = Field(
        ..., 
        example="ME COLLEGE GO TOMORROW", 
        description="Space-separated sign language tokens or glosses."
    )

class TranslationResponse(BaseModel):
    sign_sequence: str
    translation: str
    confidence: float

# 5. API Endpoints
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "SignBridge Model 3 - Sequence-to-Text Translation API"
    }

@app.post("/translate", response_model=TranslationResponse)
def translate_sequence(payload: TranslationRequest):
    if not model or not tokenizer:
        raise HTTPException(
            status_code=500, 
            detail="Model weights not loaded properly on the server."
        )
    
    clean_sequence = payload.sequence.strip()
    if not clean_sequence:
        raise HTTPException(
            status_code=400, 
            detail="Provided sign sequence cannot be empty."
        )

    # Prepare input prefix matching training setup
    input_text = "translate Sign to English: " + clean_sequence
    inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
    
    # Generate inference
    outputs = model.generate(
        **inputs, 
        max_length=64, 
        num_beams=4, 
        early_stopping=True
    )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return TranslationResponse(
        sign_sequence=clean_sequence,
        translation=translated_text,
        confidence=0.95  # Placeholder value for downstream pipeline integration
    )

# 6. Direct Execution Entrypoint
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)