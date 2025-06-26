from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import io

# Initialize FastAPI app
app = FastAPI()

# Allow CORS from all origins so the Tkinter app can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Wav2Vec2 model ===
model_path = "checkpoint-91890/"  # Path to your checkpoint folder
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read uploaded file into a float32 waveform
    audio_bytes = await file.read()
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # If sample rate isn't 16000, resample
    if sample_rate != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Run through model
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return {"transcription": transcription.strip()}
