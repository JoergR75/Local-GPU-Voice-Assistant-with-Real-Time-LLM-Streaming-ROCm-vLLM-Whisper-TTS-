# 🤖 Local GPU Voice Assistant with Real-Time LLM Streaming (ROCm + vLLM + Whisper-STT + Piper-TTS)

A fully offline, GPU-accelerated AI voice assistant that streams LLM responses in real time with speech input and output on AMD ROCm hardware.

## 🚀 Overview

This project provides a fully local, GPU-accelerated AI voice assistant running on AMD ROCm hardware. It combines high-performance LLM inference with real-time speech input and spoken responses, all without relying on cloud services or external APIs.

The assistant uses vLLM for fast, streaming language model inference, Whisper for speech-to-text transcription, Piper-TTS for natural voice output, and Gradio for a browser-based chat interface. Responses are streamed token by token for low latency, and audio playback is automatically generated once the answer is complete.

The system runs entirely offline on Ubuntu 22.04 or 24.04 with ROCm 7.2 or newer and supports modern AMD GPUs across CDNA and RDNA generations. It is designed for private AI assistant use, on-device LLM experimentation, enterprise demos, and showcasing high-performance local inference on AMD hardware.

Everything runs 100% locally on an AMD GPU with ROCm support.

## 🧠 Features

- Text-based chat with streaming responses
- Voice input via microphone → Whisper → LLM
- AI voice responses using Piper-TTS
- Low-latency, high-speed inference with vLLM
- Customizable personality through system prompts
- Fully local GPU execution on AMD ROCm hardware
- Persistent chat history within the session

## 🏗 Architecture

**Pipeline Flow:**

Microphone → Whisper → Qwen3 4B (vLLM with real-time streaming) → Piper-TTS → Audio Playback

**Core Components**

- Model loading and inference via vllm.LLM
- Chat template handling with Hugging Face tokenizer
- Real-time token streaming from vLLM to the UI
- Piper-TTS wrapped for synchronous playback
- Gradio Blocks UI with:
- Chatbot display
- Text input
- Microphone input
- Autoplay audio responses

## ⚙️ Model Configuration
```python
MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507"

SamplingParams(
    max_tokens=160,
    temperature=0.2,
    top_p=0.9
)
```

- Short, sharp responses
- Dry humor personality
- Optimized for speed and responsiveness

## 🖥 Hardware & Platform

Tested on:

- AMD Ryzen™ AI MAX 390 w/ Radeon 8050S (Strix Halo)
- ROCm 7.2.1
- Ubuntu 22.04 / 24.04
- PyTorch 2.11 (Preview)
- vLLM 0.16

Designed specifically for AMD GPU acceleration.

## 🎭 Personality System

- Ruby is configured via a structured system prompt:
- Sharp wit
- Dry humor
- Short and confident replies
- Helpful first, funny second
- Occasional references to running locally and speed
- No long explanations unless requested

## 🚀 Installation

### 1️⃣ **System preperation**
Install the latest **RDNA4** architecture docker vLLM container for Ubuntu 24.04
```bash
docker pull rocm/vllm-dev:rocm7.2_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.14.0rc0
```

### 2️⃣ **Start the vLLM container**
```bash
sudo docker run -it \
    -p 7860:7860 \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --entrypoint /bin/bash \
    vllm/vllm-openai-strix-rocm:v0.18.0_2
```

| Flag / Option | Purpose |
|---------------|---------|
| `-p 7860:7860` | Exposes port 7860 (commonly used for web UIs or API endpoints) |
| `--device=/dev/kfd` | Grants access to the ROCm kernel driver (required for compute) |
| `--device=/dev/dri` | Passes the physical GPU device into the container |
| `--security-opt seccomp=unconfined` | Required to avoid ROCm-related syscall restrictions |
| `--entrypoint /bin/bash` |  |
| `--group-add video` | Ensures proper GPU access permissions inside the container |

rocm/vllm-dev:...
Uses the ROCm 7.2 vLLM development image with:
Ubuntu 24.04
Python 3.12
PyTorch 2.9
vLLM 0.14.0rc0

Notes
Adjust /dev/dri/cardX and /dev/dri/renderDX if your GPU uses different device IDs (ls /dev/dri/ to verify).
Ensure Docker is installed and the ROCm driver is properly configured on the host system.
For production use, consider adding volume mounts for model storage and persistent data.


### 3️⃣ **Update and install** the container environment
```bash
sudo apt update
sudo apt install nano -y
sudo apt install ffmpeg -y
python3 -m pip install --upgrade pip wheel
python3 -m pip install gradio
python3 -m pip install git+https://github.com/openai/whisper.git
cd /app
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz
cd /app/piper
chmod +x piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
wget --content-disposition \
"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true"
wget --content-disposition \
"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true"
cd ..
```

### 4️⃣ **Download** the Chat Agent script
```bash
wget https://raw.githubusercontent.com/JoergR75/Local-GPU-Voice-Assistant-with-Real-Time-LLM-Streaming-ROCm-vLLM-Whisper-TTS-/refs/heads/main/Ryzen_AI/chat_agent_stream_piper_vllm_ryzen_ai.py
```

### 3️⃣ **Run** the Chat Agent
```bash
python3 chat_agent_stream_piper_vllm_ryzen_ai.py
```
Starting the Qwen3 model, Piper-TTS, and Whisper for the first time will download their weights. This may take 5–10 minutes, depending on your internet connection. 

<img width="851" height="949" alt="image" src="https://github.com/user-attachments/assets/de55543f-a062-419f-8e8d-aacf1e9dd01e" />

### 4️⃣ Launch the Gradio web Agent from another device connected to same network
First, SSH into the web server and forward port **7860**:
```echo
ssh -L 7860:0.0.0.0:7860 ai1@pc1
```
or use the the server IP address
```echo
ssh -L 7860:0.0.0.0:7860 ai1@192.168.178.xxx
```
Now you can open **http://localhost:7860** in your local browser to access the Gradio Web Agent.

<img width="1173" height="1261" alt="{10E97761-0C3A-4E91-AF82-AD8DC0059DD9}" src="https://github.com/user-attachments/assets/240b3abd-d0d5-4c62-b02e-661d68bd2349" />
