#!/usr/bin/env python3
# ================================================================================================================
# Local GPU Voice Assistant (vLLM + Gradio + Whisper + Edge-TTS)
# ================================================================================================================
# Fully local, GPU-accelerated AI voice assistant with real-time token streaming.
# Runs entirely offline on AMD ROCm hardware using:
#
#   • vLLM            → fast LLM inference + streaming tokens
#   • Gradio          → web UI / chat interface
#   • OpenAI Whisper  → speech-to-text (STT)
#   • Microsoft Edge TTS → text-to-speech (TTS)
#
# Features:
#   • low-latency streaming responses
#   • microphone input (push-to-talk)
#   • live chatbot updates
#   • automatic speech playback after generation
#   • 100% local execution (no cloud services)
#
# ================================================================================================================
# REQUIREMENTS
# ---------------------------------------------------------------------------------------------------------------
# Operating Systems:
#   - Ubuntu 22.04 LTS (Jammy Jellyfish)
#   - Ubuntu 24.04 LTS (Noble Numbat)
#
# Tested Kernel Versions:
#   - 5.15.x
#   - 6.8.x
#
# Supported Hardware:
#   - AMD ROCm GPUs
#   - CDNA1 / CDNA2 / CDNA3 / CDNA4
#   - RDNA3 / RDNA4
#
# Software Stack:
#   - Python 3.10+
#   - ROCm 6.x+
#   - PyTorch (ROCm build)
#   - vLLM
#   - Gradio
#   - Whisper
#   - edge-tts
#
# ================================================================================================================
# EXECUTION DETAILS
# ---------------------------------------------------------------------------------------------------------------
# Author:            Joerg Roskowetz
# First Run:         ~10–20 minutes (model + container download depending on internet speed)
# Last Updated:      2026-02-15
# License:           Personal / Research use
# ================================================================================================================

import gradio as gr
import whisper
import tempfile
import asyncio
import edge_tts
from functools import partial

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# -----------------------------
# Model configuration
# -----------------------------
MODEL_ID = "DavidAU/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning"
# MODEL_ID = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
# MODEL_ID = "LeoLM/leo-hessianai-13b-chat" # German language model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# -----------------------------
# Whisper (Speech → Text)
# -----------------------------
whisper_model = whisper.load_model("base")

# -----------------------------
# Text to Speech (Edge-TTS)
# -----------------------------
VOICE_NAME = "en-US-AriaNeural"
# VOICE_NAME = "de-DE-KillianNeural"
# VOICE_NAME = "de-AT-IngridNeural"

async def speak_async(text):
    """Generate speech using edge-tts and save to a temp file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        path = f.name
    communicate = edge_tts.Communicate(text, voice=VOICE_NAME)
    await communicate.save(path)
    return path

def speak(text):
    """Sync wrapper for edge-tts"""
    return asyncio.run(speak_async(text))

# -----------------------------
# LLM Stream Chat Function with personality (vLLM)
# -----------------------------
def chat_llama_stream(llm, user_input, history):
    messages = []

    system_prompt = (
        "You are Eva, a local AI assistant running on AMD Radeon AI PRO R9700 graphics hardware. "
        "Your specification is: 32GB frame buffer, 640TB/s memory bandwidth, 128 AI accelerators, 64 compute units, 191 TerraFLOPs floating point 16 Matrix performance. "
        "Respond with sharp wit and dry humor. Keep replies short, clear, and confident. "
        "Be helpful first, funny second. "
        "Occasionally reference speed, efficiency, or running locally when relevant. "
        "No long explanations unless requested."
    )

    messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.8,
        top_p=0.9,
    )

    answer = ""

    history.append({"role": "assistant", "content": ""})

    # STREAM TOKENS
    for output in llm.generate([prompt], sampling_params):

        token = output.outputs[0].text
        answer += token
        history[-1]["content"] = answer

        # 🔥 DO NOT reset audio during stream
        yield history, history, gr.update()

    # FINISHED → create TTS
    audio_path = speak(answer.strip())

    # only now update audio
    yield history, history, audio_path

# -----------------------------
# Audio Input Handler
# -----------------------------
def text_to_chat_stream(llm, text, history):
    return chat_llama_stream(llm, text, history)

# -----------------------------
# Speech Input Handler
# -----------------------------
def speech_to_chat_stream(llm, audio, history):
    result = whisper_model.transcribe(audio)
    text = result["text"]

    return chat_llama_stream(llm, text, history)

# -----------------------------
# Main (REQUIRED for vLLM spawn)
# -----------------------------
if __name__ == "__main__":

    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        max_model_len=92000,
    )

    # -----------------------------
    # Gradio UI
    # -----------------------------
    with gr.Blocks(title="🦙 Llama 3.3 Local AI Agent | AMD ROCm 7.2") as demo:
        gr.Markdown("""
    # 🤖 Local GPU Voice Assistant with Real-Time LLM Streaming (ROCm + vLLM + Whisper + TTS)
    ### 🤖 Sarcastic • 🎙️ Voice-Enabled • ⚡ GPU-Accelerated • 100% local

    ## 🧠 Model Stack
    - **LLM:** Llama 3.3 8B Instruct
    - **ASR:** OpenAI Whisper (base, 74M parameters)
    - **Framework:** PyTorch 2.11 (Preview)
    - **Library:** vLLM 0.14
    - **UI:** Gradio Web Interface

    ## 🚀 Hardware & Platform
    Running fully local on:
    **AMD Radeon™ AI PRO R9700 (RDNA4)**
    Powered by **ROCm 7.2**
    Ubuntu 22.04 / 24.04

    ## 🎤 How to Use
    - 💬 Type your message
    - 🎙️ Or speak directly
    - ⚡ Everything runs locally on a single Radeon™ AI PRO R9700 GPU

    _No cloud. No API keys. Just pure local AMD AI power._

    ## 🔗 Resources
    [![ROCm](https://img.shields.io/badge/ROCm-7.2.0-ff6b6b?logo=amd)](https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html)
    [![Whisper GitHub repo](https://img.shields.io/badge/Whisper-GitHub_repo-blue)](https://github.com/JoergR75/whisper_rocm_transcribe/tree/main/whisper_gradio_web_ui)
    [![Gradio](https://img.shields.io/badge/Gradio-Quickstart-orange)](https://www.gradio.app/guides/quickstart)
    [![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%20(Preview)-ee4c2c?logo=pytorch)](https://pytorch.org/get-started/locally/)
    [![Docker](https://img.shields.io/badge/Docker-29.2.0-blue?logo=docker)](https://www.docker.com/)
    [![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20%7C%2024.04-e95420?logo=ubuntu)](https://ubuntu.com/download/server)
    [![AMD Radeon AI PRO R9700](https://img.shields.io/badge/AMD-RDNA4%20Radeon(TM)%20AI%20PRO%20R9700-8B0000?logo=amd)](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)

    ### 😏 Warning
    Responses may contain sarcasm, wit, and dangerously high GPU utilization.
    """)

        gr.Markdown("Talk or type. Audio runs fully local on one Radeon(TM) AI PRO R9700 GPU.")

        chatbot = gr.Chatbot()
        state = gr.State([])

        txt = gr.Textbox(
            label="Type your message",
            placeholder="Press Enter to send...",
            lines=1
        )

        mic = gr.Audio(
            label="Speak",
            type="filepath",
            sources=["microphone"]
        )

        audio_out = gr.Audio(label="AI Voice Reply", autoplay=True)

        # 1) TEXT
        def text_to_chat_stream(llm, text, history):
            yield from chat_llama_stream(llm, text, history)

        txt.submit(
            partial(text_to_chat_stream, llm),
            inputs=[txt, state],
            outputs=[chatbot, state, audio_out]
        )

        # 2) Stop recording → send speech automatically
        def speech_to_chat_stream(llm, audio, history):
            result = whisper_model.transcribe(audio)
            text = result["text"]

            yield from chat_llama_stream(llm, text, history)

        mic.stop_recording(
            partial(speech_to_chat_stream, llm),
            inputs=[mic, state],
            outputs=[chatbot, state, audio_out]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
