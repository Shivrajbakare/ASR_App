import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf

app = FastAPI()

# Allow CORS from all origins (so your Tkinter app can call from anywhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor once on server start
model_path = "checkpoint-91890/"
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read uploaded audio file as bytes
    audio_bytes = await file.read()
    
    # Load audio with soundfile (to numpy float32 array)
    audio_input, sample_rate = sf.read(io.BytesIO(audio_bytes))
    
    # Resample to 16000 if needed (optional, add librosa if needed)
    if sample_rate != 16000:
        import librosa
        audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Convert to torch tensor input features
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    transcription = transcription.replace("[PAD]", "").strip()

    return {"transcription": transcription}
