import os
import subprocess
import time

PIPER_BIN = "/home/arora/piper/piper/piper"
VOICE_MODEL = "/home/arora/piper/voices/en_US-amy-medium.onnx"
OUT_DIR = "/home/arora/assistant/data/out"

os.makedirs(OUT_DIR, exist_ok=True)

def speak(text: str) -> str:
    if not text.strip():
        return ""

    filename = f"tts_{int(time.time())}.wav"
    out_path = os.path.join(OUT_DIR, filename)

    cmd = [
        PIPER_BIN,
        "--model", VOICE_MODEL,
        "--output_file", out_path
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process.communicate(input=text, timeout=60)

        if os.path.exists(out_path):
            return out_path
        else:
            raise RuntimeError("TTS file not created")

    except Exception as e:
        print("TTS Error:", e)
        return ""
