from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import io

app = FastAPI()

# Allow CORS (frontend upload support)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set frontend URL here for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load model and processor ===
model_path = r"C:\Users\DELL\Downloads\ASR_App\ASR_App\checkpoint-91890"

# Load processor and manually ensure PAD token is treated correctly
processor = Wav2Vec2Processor.from_pretrained(model_path)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = "[PAD]"

model = Wav2Vec2ForCTC.from_pretrained(model_path)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# === Root route ===
@app.get("/")
async def root():
    return {"message": "Sanskrit Speech-to-Text API is running"}

# === Transcription route ===
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Read the uploaded WAV file
        audio_bytes = await file.read()
        audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Convert stereo to mono if needed
        if len(audio_np.shape) > 1:
            audio_np = np.mean(audio_np, axis=1)

        # Pad audio if shorter than 1 sec
        if len(audio_np) < 16000:
            audio_np = np.pad(audio_np, (0, 16000 - len(audio_np)), mode='constant')

        # Tokenize audio input
        inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        # Model inference
        model.eval()
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Decode prediction
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        # Optional: remove leftover [PAD] just in case
        transcription = transcription.replace("[PAD]", "").strip()

        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
