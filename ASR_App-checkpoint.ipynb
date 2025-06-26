import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import numpy as np
import sounddevice as sd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from PIL import Image, ImageTk

import os
model_path=r"C:\Users\DELL\Downloads\ASR_App\ASR_App\checkpoint-91890"
# print(os.listdir("checkpoint-91890/"))

# Define the model path
# model_path = "checkpoint-91890/"


# Load the processor and model

processor = Wav2Vec2Processor.from_pretrained(model_path)
processor.save_pretrained("checkpoint-91890/")
model = Wav2Vec2ForCTC.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

recording = False  # Global variable to control the recording state
audio = None       # Global variable to store the final audio array

# Function to record audio in real-time
def record_audio(sample_rate=16000):
    global audio
    duration = 10  # Set a default duration for recording
    
    def record():
        global audio
        print("Recording...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        audio = audio.squeeze()  # Remove unnecessary dimensions
        print("Recording finished.")
    
    # Start recording in a separate thread
    threading.Thread(target=record).start()

# Function to stop recording and start transcription
def stop_recording():
    global recording, audio
    if recording:
        recording = False
        sd.stop()  # Stop the recording

        # Update UI before starting transcription
        app.status_label.config(text="Transcribing...", fg="blue")
        app.update_idletasks()  # Force UI to update

        try:
            transcription = transcribe(audio)
            app.transcription_text.delete(1.0, tk.END)
            app.transcription_text.insert(tk.END, "\n") 
            app.transcription_text.insert(tk.END, transcription)
            app.status_label.config(text="Recording finished and transcription completed.", fg="blue")
        except Exception as e:
            app.status_label.config(text="Error during transcription.", fg="red")
            messagebox.showerror("Transcription Error", str(e))

# Function to transcribe a single audio array
def transcribe(audio):
    if audio is None or len(audio) < 4000:
        return "Audio too short to transcribe. Please speak for at least 1 second."

    # Extract input features from the audio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    input_features = input_features.to(device)

    # Generate transcription
    model.eval()
    with torch.no_grad():
        logits = model(input_features).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the generated IDs to text
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    transcription = transcription.replace('[PAD]', '').strip()

    return transcription

# def transcribe(audio):
#     # Extract input features from the audio
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_values
#     input_features = input_features.to(device)

#     # Generate transcription
#     model.eval()
#     with torch.no_grad():
#         logits = model(input_features).logits
#         predicted_ids = torch.argmax(logits, dim=-1)

#     # Decode the generated IDs to text
#     transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
#     transcription = transcription.replace('[PAD]', '').strip()

#     return transcription

class AudioTranscriptionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MIPS Sanskrit ASR APP")
        self.geometry("800x600")  # Increased size of the app window
        # Load and display the logo using Pillow
        self.load_logo()
        # Add heading
        self.heading_label = tk.Label(self, text="MIPS Sanskrit ASR App", font=("Helvetica", 60, "bold"))
        self.heading_label.pack(pady=10)

       



        self.create_widgets()

    def load_logo(self):
        try:
            # Open and resize the image
            self.logo_image = Image.open("logo.png")  # Ensure this path is correct
            self.logo_image = self.logo_image.resize((100, 100), Image.LANCZOS)  # Resize to 100x100 pixels
            self.logo = ImageTk.PhotoImage(self.logo_image)
            self.logo_label = tk.Label(self, image=self.logo)
            self.logo_label.pack(pady=5)
        except Exception as e:
            print("Error loading image:", e)
            self.logo_label = tk.Label(self, text="Logo could not be loaded")
            self.logo_label.pack(pady=5)

    def create_widgets(self):
        # Button to start recording
        self.record_button = tk.Button(self, text="Start", command=self.start_recording, height=4, width=20,font=("Helvetica", 20, "bold"))
        self.record_button.pack(pady=10)

        # Button to stop recording
        self.stop_button = tk.Button(self, text="Stop", command=self.stop_recording, height=4, width=20,font=("Helvetica", 20, "bold"))
        self.stop_button.pack(pady=10)

        # Label to show status messages
        self.status_label = tk.Label(self, text="", fg="white")
        self.status_label.pack(pady=10)

        # Text widget to display transcription
        self.transcription_text = tk.Text(self, wrap=tk.WORD, height=100, width=400,font=("Helvetica", 30, "bold"))
        self.transcription_text.pack(pady=10)
        
    def start_recording(self):
        global recording
        if recording:
            messagebox.showwarning("Warning", "Recording already in progress.")
            return
        
        recording = True
        self.status_label.config(text="Recording started...", fg="white")
        self.update_idletasks()  # Ensure the GUI updates before recording starts
        record_audio()

    def stop_recording(self):
        stop_recording()  # Ensure recording stops and transcription is performed
        self.status_label.config(text="Recording stopped.", fg="white")

# Run the application
if __name__ == "__main__":
    app = AudioTranscriptionApp()
    app.mainloop()
