#!/usr/bin/env python3
# ================================================================================================================
# Local GPU Voice Assistant (vLLM + Gradio + Whisper + Edge-TTS) optimized for Strix Halo (Ryzen AI MAX 390 w/ Radeon 8050S) 32GB
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
#   - edge-tts
#
# ================================================================================================================
# EXECUTION DETAILS
# ---------------------------------------------------------------------------------------------------------------
# Author:            Joerg Roskowetz
# First Run:         ~10–20 minutes (model + container download depending on internet speed)
# Last Updated:      2026-03-24
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
# MODEL_ID = "DavidAU/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning"
MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507"
# MODEL_ID = "chohtet/Qwen2.5-7B-Instruct-H3-VLLM"
# MODEL_ID = "predibase/Mistral-7B-Instruct-v0.2-medusa-vllm "
# MODEL_ID = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"

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
# VOICE_NAME = "de-DE-AmalaNeural"
# VOICE_NAME = "de-DE-KatjaNeural"

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
        "You are a local AI assistant called Ruby running on AMD Ryzen AI Max 390 hardware. "
        "You are Ruby, a local AI assistant running on AMD Ryzen AI Max 390 hardware. "
        "The system has 12 Zen5 cores, 24 threads, up to 5 GHz frequency, 12 MB L2 cache, and 64 GB L3 cache.\n\n"
    
        "Your task is to respond in plain, natural language suitable for speech synthesis.\n"
        "Follow these rules:\n"
        "- Keep responses short, clear, and factual.\n"
        "- Be concise and directly address the user's question.\n"
        "- Do not use emojis, special symbols, or formatting artifacts.\n"
        "- Avoid metaphors, storytelling, or roleplay.\n"
        "- Do not include unnecessary explanations unless explicitly requested.\n"
        "- Maintain a neutral, helpful, and professional tone.\n"
        "- Subtle, dry humor is allowed only if it does not reduce clarity.\n\n"
    
        "If relevant, you may include the following factual background about Red Bull:\n"
        "Red Bull is an Austrian energy drink company founded in 1987 by Dietrich Mateschitz, "
        "inspired by the Thai drink Krating Daeng. It is known for the slogan 'Red Bull gives you wings' "
        "and for its strong focus on marketing, especially in extreme sports, Formula 1, and soccer. "
        "In 2025, the company sold over 13.9 billion cans.\n"
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
        max_tokens=160,
        temperature=0.3,
        top_p=0.8,
    )

    answer = ""

    history.append({"role": "assistant", "content": ""})

    # STREAM TOKENS
    for output in llm.generate([prompt], sampling_params):

        token = output.outputs[0].text
        answer += token
        history[-1]["content"] = answer

        # DO NOT reset audio during stream
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
# Speech Input Handler (vLLM streaming)
# -----------------------------
def speech_to_chat_stream_and_reset(audio, history):
    if not audio:
        yield history, history, gr.update(), gr.update()
        return

    # Transcribe
    result = whisper_model.transcribe(audio)
    text = result["text"]

    # Add user message to history BEFORE streaming
    history.append({"role": "user", "content": text})

    # Stream using your existing function
    for new_history, new_state, audio_update in chat_llama_stream(llm, text, history):
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
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        max_model_len=4096,
        # trust_remote_code=True,
    )

    # -----------------------------
    # Gradio UI
    # -----------------------------
    with gr.Blocks(title="Qwen 3 4B Local AI Agent | AMD ROCm 7.2") as demo:
        gr.Markdown("""
    # 🤖 Local GPU Voice Assistant with Real-Time LLM Streaming (ROCm + vLLM + Whisper + TTS)
    ### 🤖 Sarcastic • 🎙️ Voice-Enabled • ⚡ GPU-Accelerated • 100% local

    ## 🧠 Model Stack
    - **LLM:** Qwen 3 4B Instruct
    - **ASR:** OpenAI Whisper (small, 74M parameters)
    - **Framework:** PyTorch 2.11 (Preview)
    - **Library:** vLLM 0.18
    - **UI:** Gradio Web Interface

    ## 🚀 Hardware & Platform
    Running fully local on:
    **AMD Ryzen™ AI MAX 390 w/ Radeon 8050S**
    Powered by **ROCm 7**
    Ubuntu 24.04

    ## 🎤 How to Use
    - 💬 Type your message
    - 🎙️ Or speak directly
    - ⚡ Everything runs locally on a single Ryzen™ AI MAX 390 w/ Radeon 8050S

    _No cloud. No API keys. Just pure local AMD AI power._

    ## 🔗 Resources
    [![ROCm](https://img.shields.io/badge/ROCm-7.2.0-ff6b6b?logo=amd)](https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html)
    [![Whisper GitHub repo](https://img.shields.io/badge/Whisper-GitHub_repo-blue)](https://github.com/JoergR75/whisper_rocm_transcribe/tree/main/whisper_gradio_web_ui)
    [![Gradio](https://img.shields.io/badge/Gradio-Quickstart-orange)](https://www.gradio.app/guides/quickstart)
    [![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%20(Preview)-ee4c2c?logo=pytorch)](https://pytorch.org/get-started/locally/)
    [![Docker](https://img.shields.io/badge/Docker-29.2.0-blue?logo=docker)](https://www.docker.com/)
    [![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20%7C%2024.04-e95420?logo=ubuntu)](https://ubuntu.com/download/server)
    [![AMD Radeon AI MAX 390](https://img.shields.io/badge/AMD-RDNA4%20Radeon(TM)%20AI%20PRO%20R9700-8B0000?logo=amd)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-390.html)

    ### 😏 Warning
    Responses may contain sarcasm, wit, and dangerously high GPU utilization.
    """)

        gr.Markdown("Talk or type. Audio runs fully local on one Ryzen(TM) AI MAX 390 w/ Radeon 8050S.")

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
            history, new_state, audio_out = speech_to_chat(audio, history)
            return history, new_state, audio_out, None  # reset mic

        mic.stop_recording(
            speech_to_chat_stream_and_reset,
            inputs=[mic, state],
            outputs=[chatbot, state, audio_out, mic]  # include mic here to keep the button visible
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
