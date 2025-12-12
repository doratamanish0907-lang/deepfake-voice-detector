import streamlit as st
import torch
import torchaudio
import time
from step4_train_model import VoiceCNN

# ------------------------------------------
# Load Model
# ------------------------------------------
MODEL_PATH = "voice_cnn_model.pth"

model = VoiceCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ------------------------------------------
# Preprocess Audio
# ------------------------------------------
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=128
)

def preprocess_audio(wav):
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    mel = mel_spectrogram(wav)

    max_len = 1500
    if mel.shape[2] > max_len:
        mel = mel[:, :, :max_len]
    else:
        mel = torch.nn.functional.pad(mel, (0, max_len - mel.shape[2]))

    mel = mel.unsqueeze(0)
    return mel

# ------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="Deepfake Voice Detector",
    layout="centered",
    page_icon="🎤"
)

# ------------------------------------------
# Custom CSS For Premium UI
# ------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
}

/* Glowing Header */
@keyframes neonGlow {
    0% { text-shadow: 0 0 5px #00eaff; }
    50% { text-shadow: 0 0 20px #00c3ff; }
    100% { text-shadow: 0 0 5px #00eaff; }
}

.title {
    font-size: 38px;
    color: white;
    text-align: center;
    font-weight: bold;
    animation: neonGlow 2.5s infinite;
}

/* Manish Branding */
@keyframes glowName {
    0% { text-shadow: 0 0 4px #ff00c3; }
    50% { text-shadow: 0 0 18px #ff00e0; }
    100% { text-shadow: 0 0 4px #ff00c3; }
}

.creator {
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    color: #ffb3f7;
    animation: glowName 3s infinite;
}

/* Upload Button Styling */
button[kind="secondary"] {
    background-color: #00c3ff !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 10px !important;
}

/* Result Box */
.result-box {
    padding: 15px;
    border-radius: 12px;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# Header Section
# ------------------------------------------
st.image("static/logo.png", width=150)

st.markdown("<div class='title'>Deepfake Voice Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='creator'>✨ Created by Manish Kumar ✨</div><br>", unsafe_allow_html=True)

# ------------------------------------------
# Upload Section
# ------------------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("🎤 Upload your voice sample (WAV or MP3)", type=["wav", "mp3"])
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# Prediction Logic
# ------------------------------------------
if uploaded_file is not None:
    st.audio(uploaded_file)

    st.markdown("<div class='card'>Processing audio... ⏳</div>", unsafe_allow_html=True)
    time.sleep(1)

    try:
        wav, sr = torchaudio.load(uploaded_file)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        mel = preprocess_audio(wav)

        with torch.no_grad():
            output = model(mel)
            prob = torch.softmax(output, dim=1)

        real_score, fake_score = prob[0][0].item(), prob[0][1].item()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if fake_score > real_score:
            st.error(f"🚨 **Fake Voice Detected!**\n\nConfidence: **{fake_score:.4f}**")
        else:
            st.success(f"✅ **Real Voice**\n\nConfidence: **{real_score:.4f}**")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing audio: {e}")


