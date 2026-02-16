# 🤖 Local GPU Voice Assistant with Real-Time LLM Streaming (ROCm + vLLM + Whisper + TTS)

A fully offline, GPU-accelerated AI voice assistant that streams LLM responses in real time with speech input and output on AMD ROCm hardware.

## 🚀 Overview

This project provides a fully local, GPU-accelerated AI voice assistant running on AMD ROCm hardware. It combines high-performance LLM inference with real-time speech input and spoken responses, all without relying on cloud services or external APIs.

The assistant uses vLLM for fast, streaming language model inference, Whisper for speech-to-text transcription, Edge-TTS for natural voice output, and Gradio for a browser-based chat interface. Responses are streamed token by token for low latency, and audio playback is automatically generated once the answer is complete.

The system runs entirely offline on Ubuntu 22.04 or 24.04 with ROCm 7.2 or newer and supports modern AMD GPUs across CDNA and RDNA generations. It is designed for private AI assistant use, on-device LLM experimentation, enterprise demos, and showcasing high-performance local inference on AMD hardware.

Everything runs 100% locally on an AMD GPU with ROCm support.

## 🧠 Features

- Text-based chat with streaming responses
- Voice input via microphone → Whisper → LLM
- AI voice responses using Edge-TTS
- Low-latency, high-speed inference with vLLM
- Customizable personality through system prompts
- Fully local GPU execution on AMD ROCm hardware
- Persistent chat history within the session

## 🏗 Architecture

**Pipeline Flow:**

Microphone → Whisper → Llama 3.3 (vLLM with real-time streaming) → Edge-TTS → Audio Playback

**Core Components**

- Model loading and inference via vllm.LLM
- Chat template handling with Hugging Face tokenizer
- Real-time token streaming from vLLM to the UI
- Async Edge-TTS wrapped for synchronous playback
- Gradio Blocks UI with:
- Chatbot display
- Text input
- Microphone input
- Autoplay audio responses

## ⚙️ Model Configuration
```python
MODEL_ID = "DavidAU/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning"

SamplingParams(
    max_tokens=256,
    temperature=0.8,
    top_p=0.9
)
```

- Short, sharp responses
- Dry humor personality
- Optimized for speed and responsiveness

## 🖥 Hardware & Platform

Tested on:

- AMD Radeon™ AI PRO R9700 (RDNA4) and Radeon™ PRO W7900 (RDNA3)
- ROCm 7.2
- Ubuntu 22.04 / 24.04
- PyTorch 2.11 (Preview)
- vLLM 0.14

Designed specifically for AMD GPU acceleration.

## 🎭 Personality System

- Eva is configured via a structured system prompt:
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
    rocm/vllm-dev:rocm7.2_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.14.0rc0
```

| Flag / Option | Purpose |
|---------------|---------|
| `-p 7860:7860` | Exposes port 7860 (commonly used for web UIs or API endpoints). |
| `--device=/dev/kfd` | Grants access to the ROCm kernel driver (required for compute). |
| `--device=/dev/dri` | Passes the physical GPU device into the container. |
| `--security-opt seccomp=unconfined` | Required to avoid ROCm-related syscall restrictions. |
| `--group-add video` | Ensures proper GPU access permissions inside the container. |

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
python3 -m pip install asyncio
python3 -m pip install edge-tts
```

### 4️⃣ **Download** the Chat Agent script
```bash
wget https://raw.githubusercontent.com/JoergR75/Local-GPU-Voice-Assistant-with-Real-Time-LLM-Streaming-ROCm-vLLM-Whisper-TTS-/refs/heads/main/chat_agent_stream_vllm.py
```

### 3️⃣ **Run** the Chat Agent
```bash
python3 chat_agent_stream_vllm.py
```

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

<img width="1531" height="1264" alt="{E30F3FDA-3E55-42A4-B889-BEE91AF7F30E}" src="https://github.com/user-attachments/assets/b7bcf3c8-9aed-47b3-9609-cde90c44c894" />
