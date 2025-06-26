from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf
import io

from pyngrok import ngrok
import nest_asyncio
import uvicorn
import threading

# Enable nested asyncio (required for Colab)
nest_asyncio.apply()

# Load model and processor
model_path = "facebook/wav2vec2-base-960h"  # or your custom checkpoint if uploaded
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define app
app = FastAPI()

# CORS for browser/API clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if sample_rate != 16000:
            return {"error": "Sample rate must be 16000 Hz"}

        input_values = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_values.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        transcription = processor.decode(predicted_ids[0])
        return {"transcription": transcription.strip()}
    except Exception as e:
        return {"error": str(e)}

# Start FastAPI in background thread
def start_ngrok_and_uvicorn():
    public_url = ngrok.connect(8000)
    print(f"Public URL: {public_url}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=start_ngrok_and_uvicorn).start()

