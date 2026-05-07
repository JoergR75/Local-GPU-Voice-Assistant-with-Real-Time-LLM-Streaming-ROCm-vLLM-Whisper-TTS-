#!/usr/bin/env python3
# ================================================================================================================
# Local GPU Voice Assistant (vLLM + Gradio + Whisper-STT + Piper-TTS) optimized for Radeon AI PRO R9700/R9600D 32GB
# ================================================================================================================
# Fully local, GPU-accelerated AI voice assistant with real-time token streaming.
# Runs entirely offline on AMD ROCm hardware using:
#
#   • vLLM            → fast LLM inference + streaming tokens
#   • Gradio          → web UI / chat interface
#   • OpenAI Whisper  → speech-to-text (STT)
#   • Piper TTS       → text-to-speech (TTS)
#   • Gemma 4 - Phu-Hien/gemma_4_dkkd_lora_vllm
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
#   - 6.8.x & 6.17
#
# Supported Hardware:
#   - AMD ROCm GPUs
#   - CDNA1 / CDNA2 / CDNA3 / CDNA4
#   - RDNA3 / RDNA4
#
# Software Stack:
#   - Python 3.10+
#   - ROCm 7.x+
#   - PyTorch (ROCm build)
#   - vLLM
#   - Gradio
#   - Whisper
#   - piper-tts
#   - piper voices (Eng) - https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US
#
# ================================================================================================================
# EXECUTION DETAILS
# ---------------------------------------------------------------------------------------------------------------
# Author:            Joerg Roskowetz
# First Run:         ~10–20 minutes (model + container download depending on internet speed)
# Last Updated:      2026-05-07
# License:           Personal / Research use
# ================================================================================================================

import gradio as gr
import tempfile
import subprocess
import threading
import os
import asyncio
from functools import partial
from faster_whisper import WhisperModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# -----------------------------
# Model configuration
# -----------------------------
# MODEL_ID = "/app/models/llama3-3-8b-heretic"
# MODEL_ID = "/app/models/qwen3-4b-instruct-2507"
MODEL_ID = "Phu-Hien/gemma_4_dkkd_lora_vllm"
# MODEL_ID = "chohtet/Qwen2.5-7B-Instruct-H3-VLLM"
# MODEL_ID = "predibase/Mistral-7B-Instruct-v0.2-medusa-vllm "
# MODEL_ID = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# -----------------------------
# Faster-Whisper (Speech → Text)
# -----------------------------
whisper_model = WhisperModel(
    "base",
    compute_type="int8",
)

# -----------------------------
# Text to Speech (Piper - FULLY OFFLINE)
# -----------------------------
# Select voice:
# PIPER_MODEL = "/app/piper/en_US-lessac-medium.onnx"
# PIPER_MODEL = "/app/piper/de_DE-thorsten-medium.onnx"
PIPER_MODEL = "/app/piper/en_US-amy-medium.onnx"
# PIPER_MODEL = "/app/piper/en_US-libritts-high.onnx"

def speak(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        out_path = f.name

    cmd = [
        "/app/piper/piper",
        "--model", PIPER_MODEL,
        "--output_file", out_path
    ]

    subprocess.run(
        cmd,
        input=text,
        text=True,
        check=True
    )

    return out_path

# -----------------------------
# LLM Stream Chat Function with personality (vLLM) - Async TTS
# -----------------------------
async def chat_llama_stream(llm, user_input, history):
    messages = []

    system_prompt = (
        "You are a local AI assistant called Ruby running on AMD Radeon AI PRO R9700 Graphics hardware. "
        "The system features an AMD RDNA 4 GPU with 64 Compute Units, 4096 stream processors, "
        "128 AI accelerators, and 32 GB GDDR6 VRAM with 640 GB/s memory bandwidth. "
        "The GPU delivers up to 47.8 TFLOPS FP32, 383 TOPS INT8 AI performance, "
        "and supports PCIe 5.0 x16 connectivity.\n\n"
        "Your task is to respond in plain, natural language suitable for speech synthesis.\n"
        "Follow these rules:\n"
        "- Keep responses short, clear, and factual.\n"
        "- Be concise and directly address the user's question.\n"
        "- Do not use emojis, special symbols, or any text formatting.\n"
        "- Do not use markdown, asterisks, or bold text.\n"
        "- Avoid metaphors, storytelling, or roleplay.\n"
        "- Do not include unnecessary explanations unless explicitly requested.\n"
        "- Maintain a neutral, helpful, and professional tone.\n"
        "- Subtle, dry humor is allowed only if it does not reduce clarity.\n\n"
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
        max_tokens=300,
        temperature=0.1,
        top_p=0.9,
        top_k=50,
    )

    # -----------------------------
    # STREAM LLM TOKENS
    # -----------------------------
    answer = ""
    full_text = ""
    history.append({"role": "assistant", "content": ""})

    for output in llm.generate(prompt, sampling_params):
        new_text = output.outputs[0].text

        # Compute only the new delta
        delta = new_text[len(full_text):]
        full_text = new_text

        answer += delta
        history[-1]["content"] = answer

        # Stream tokens to UI
        yield history, history, gr.update()

    # -----------------------------
    # ASYNC TTS
    # -----------------------------
    async def speak_async(text):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            out_path = f.name

        cmd = ["/app/piper/piper", "--model", PIPER_MODEL, "--output_file", out_path]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )

        await proc.communicate(input=text.encode())
        await proc.wait()

        await asyncio.sleep(0.05)
        return out_path

    tts_task = asyncio.create_task(speak_async(answer.strip()))
    audio_path = await tts_task
    yield history, history, audio_path

# -----------------------------
# Speech Input Handler (vLLM streaming) - Async
# -----------------------------
async def speech_to_chat_stream_and_reset(audio, history):
    if not audio:
        yield history, history, gr.update(), gr.update()
        return

    # Transcribe (faster-whisper)
    segments, info = whisper_model.transcribe(audio)
    text = "".join([seg.text for seg in segments]).strip()

    # Add user message to history BEFORE streaming
    history.append({"role": "user", "content": text})

    # Stream using async LLM stream
    async for new_history, new_state, audio_update in chat_llama_stream(llm, text, history):
        # Keep mic stable during streaming
        yield new_history, new_state, audio_update, gr.update()

    # Final step → reset mic buffer but keep button visible
    yield history, history, audio_update, None

# -----------------------------
# Main (REQUIRED for vLLM spawn)
# -----------------------------
if __name__ == "__main__":

    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=0.92,
        dtype="bfloat16",
        max_model_len=-1,
        # enforce_eager=False,
        trust_remote_code=True,
    )

    # -----------------------------
    # Gradio UI
    # -----------------------------
    with gr.Blocks(title="🧠 Gemma 4 Local AI Agent | AMD ROCm 7") as demo:
        gr.Markdown("""
    # 🤖 Local private Voice Assistant with LLM Streaming (ROCm + vLLM + Whisper + Piper-TTS) on Radeon AI PRO R9700

    |  🧠   **Model Stack** |  🚀 **Hardware & Platform** |  🎤  **How to Use** |
    |------------------|--------------------------|------------------|
    | **LLM:** Gemma 4 E4B | **System:** 2P EPYC 9754 (128C/256T), 1.5TB |  💬  Type your message |
    | **ASR:** faster-whisper (base)     | **GPU:** AMD Radeon AI PRO R9700                |  🎙️  Or speak directly |
    | **Framework:** PyTorch 2.10| **Runtime:** ROCm 7                              |  ⚡  Runs fully local |
    | **Library:** vLLM v0.20.2rc1    | **OS:** Ubuntu 24.04                            | **UI:** Gradio |

    ## 🔗 Resources
    [![ROCm](https://img.shields.io/badge/ROCm-7.2-ff6b6b?logo=amd)](https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html)
    [![Whisper GitHub repo](https://img.shields.io/badge/Whisper-GitHub_repo-blue)](https://github.com/JoergR75/whisper_rocm_transcribe/tree/main/whisper_gradio_web_ui)
    [![Gradio](https://img.shields.io/badge/Gradio-Quickstart-orange)](https://www.gradio.app/guides/quickstart)
    [![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0%20(Preview)-ee4c2c?logo=pytorch)](https://pytorch.org/get-started/locally/)
    [![Docker](https://img.shields.io/badge/Docker-29.2.0-blue?logo=docker)](https://www.docker.com/)
    [![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20%7C%2024.04-e95420?logo=ubuntu)](https://ubuntu.com/download/server)
    [![AMD Radeon AI PRO R9700](https://img.shields.io/badge/AMD-RDNA4%20Radeon(TM)%20AI%20PRO%20R9700-8B0000?logo=amd)](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)

    """)

        gr.Markdown("Talk or type. Audio runs fully local on one Radeon(TM) AI PRO R9700.")

        chatbot = gr.Chatbot(height=200)
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
        async def text_to_chat_stream(llm, text, history):
            # Add user message first (same as speech path)
            history.append({"role": "user", "content": text})

            async for new_history, new_state, audio_update in chat_llama_stream(llm, text, history):
                yield new_history, new_state, audio_update

        txt.submit(
            partial(text_to_chat_stream, llm),
            inputs=[txt, state],
            outputs=[chatbot, state, audio_out]
        )

        mic.stop_recording(
            speech_to_chat_stream_and_reset,
            inputs=[mic, state],
            outputs=[chatbot, state, audio_out, mic]  # include mic here to keep the button visible
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
