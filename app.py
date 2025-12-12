import streamlit as st
import torch
import librosa
import numpy as np
from step4_train_model import VoiceCNN  # Your model
from PIL import Image

# ---------------------
# PAGE CONFIG
# ---------------------
st.set_page_config(page_title="Voice Deepfake Detector",
                   page_icon="🎤",
                   layout="centered")

# ---------------------
# LOAD LOGO
# ---------------------
try:
    logo = Image.open("static/logo.png")
    st.image(logo, width=150)
except:
    st.warning("Logo not found. Make sure 'static/logo.png' exists.")

# ---------------------
# TITLE + ANIMATED NAME
# ---------------------
st.markdown("""
<h1 style='text-align:center; color:#00C3FF;'>🔍 Voice Deepfake Detection</h1>

<div style='text-align:center; font-size:22px; font-weight:bold;'>
✨ Developed by <span style='color:#FF007F; animation: glow 1s infinite;'>Manish Kumar</span> ✨
</div>

<style>
@keyframes glow {
  0% { text-shadow: 0 0 5px #FF007F; }
  50% { text-shadow: 0 0 20px #FF007F; }
  100% { text-shadow: 0 0 5px #FF007F; }
}
</style>
""", unsafe_allow_html=True)

st.write("Upload an audio file to test whether it is **Real or Fake**.")

# ---------------------
# LOAD MODEL
# ---------------------
MODEL_PATH = "voice_cnn_model_small.pth"

model = VoiceCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---------------------
# AUDIO PREPROCESSING (librosa)
# ---------------------
def preprocess_audio(file_path):
    wav, sr = librosa.load(file_path, sr=16000)

    # Generate Mel-spectrogram (same config as training)
    mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=400,
                                         hop_length=160, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)

    mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()
    return mel_tensor

# ---------------------
# FILE UPLOAD SECTION
# ---------------------
audio_file = st.file_uploader("Upload WAV/MP3 File", type=["wav", "mp3"])

if audio_file:
    
    with open("temp_input.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    st.success("Audio uploaded successfully!")
    st.audio(audio_file)

    mel = preprocess_audio("temp_input.wav")

    with torch.no_grad():
        output = model(mel)
        probs = torch.softmax(output, dim=1).numpy()[0]

    real_prob = probs[0]
    fake_prob = probs[1]

    # ---------------------
    # RESULT UI
    # ---------------------
    st.write("### 🎯 Detection Result")

    if fake_prob > real_prob:
        st.error(f"🚨 **FAKE VOICE DETECTED!**\nFake Probability: **{fake_prob:.4f}**")
    else:
        st.success(f"✅ **REAL VOICE**\nReal Probability: **{real_prob:.4f}**")

    st.progress(float(fake_prob))






