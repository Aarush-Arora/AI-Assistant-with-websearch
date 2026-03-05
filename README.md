# Phenora --- AI Voice Assistant

Phenora is a **local‑first AI voice assistant backend** that turns your
spoken words into intelligent spoken answers. It listens, understands,
searches the web, reads articles, checks Wikipedia and weather, and
finally speaks back --- all while remembering your conversation.

This project was **developed with the help of AI tools** for
architecture, debugging, and documentation assistance.\
**ESP32 hardware integration is coming soon** -- stay tuned!

------------------------------------------------------------------------

## How It Works (In Simple Terms)

1.  You speak → audio file is uploaded.
2.  Phenora converts speech to text (Whisper).
3.  It figures out what you need:
    -   **Real‑time data** (news, weather, current events) → searches
        the web.
    -   **Knowledge** (time, math, history, general facts) → answers
        from its own brain (LLM).
    -   **Conversational** (chit‑chat) → just talks naturally.
4.  If web search is needed, it generates smart search queries, fetches
    articles, summarises them, and merges everything into one fluent
    answer.
5.  It keeps track of the conversation (SQLite) so you can ask
    follow‑ups.
6.  Finally, it turns the answer into speech (Piper TTS) and sends back
    the audio file.

------------------------------------------------------------------------

## Features

-   Voice → Text using **Faster‑Whisper** (local, fast, accurate)
-   Web search via **SearXNG** (private, self‑hosted)
-   Article scraping & summarisation
-   Wikipedia knowledge retrieval
-   Weather via **Open‑Meteo** (free, no API key)
-   LLM reasoning using **Groq** (fast, cheap, models like Llama 3)
-   Conversation memory via **SQLite**
-   Text‑to‑speech using **Piper** (local, high‑quality, many voices)
-   Real‑time context awareness (current time in multiple timezones)
-   FastAPI backend with simple API endpoints
-   Debug endpoint to inspect every step
-   **ESP32 integration coming soon!**

------------------------------------------------------------------------

## Architecture

Audio Input → Audio Normalization (ffmpeg) → Speech Recognition
(Whisper)\
→ Intent Detection → Web Search / Wikipedia / Weather\
→ Article Summaries → LLM Reasoning → Final Answer → Text To Speech

------------------------------------------------------------------------

## Tech Stack

-   Python 3.10+
-   FastAPI (web framework)
-   faster‑whisper (speech‑to‑text)
-   Groq API (LLM inference)
-   BeautifulSoup (web scraping)
-   SQLite (memory)
-   ffmpeg (audio processing)
-   Open‑Meteo API (weather)
-   SearXNG (meta‑search engine)
-   Piper (text‑to‑speech)

------------------------------------------------------------------------

## System Requirements

-   **Operating System**: Linux (Ubuntu 22.04 / 24.04 recommended),
    macOS, or Windows with WSL2 (Ubuntu)
-   **Python**: 3.10 or higher
-   **RAM**: 8 GB minimum, 16 GB recommended
-   **CPU**: 4 cores minimum, 8+ cores recommended
-   **Disk**: \~5 GB for models and temporary files
-   **Internet**: Required for Groq API, web search, and weather (unless
    you run everything offline)

If you are on **Windows**, install **WSL2** with Ubuntu (see
https://learn.microsoft.com/en-us/windows/wsl/install).

------------------------------------------------------------------------

## Installation

### 1. System Dependencies

Update your package list and install required tools:

``` bash
sudo apt update
sudo apt install python3 python3-venv python3-pip ffmpeg git curl
```

### 2. Clone the Repository

``` bash
git clone https://github.com/yourname/phenora
cd phenora
```

### 3. Python Virtual Environment

Create and activate a virtual environment:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Python Dependencies

All required Python packages are listed in requirements.txt. Install
them with:

``` bash
pip install -r requirements.txt
```

### 5. Install and Configure SearXNG (Web Search)

SearXNG is a private metasearch engine that Phenora uses to fetch live
web results.

#### Option A: Using Docker (easiest)

``` bash
sudo apt install docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Then run:

``` bash
docker run -d --name searxng -p 8080:8080 searxng/searxng
```

SearXNG will be available at http://localhost:8080.

------------------------------------------------------------------------

### 6. Install and Configure Piper (Text‑to‑Speech)

Create a directory:

``` bash
mkdir -p ~/piper
cd ~/piper
```

Download Piper:

``` bash
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
```

Download voice model:

``` bash
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Create `app/tts.py`:

``` python
import subprocess
import tempfile
import os

PIPER_PATH = "/home/yourusername/piper/piper"
MODEL_PATH = "/home/yourusername/piper/en_US-lessac-medium.onnx"

def speak(text: str) -> str:
    out_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = out_file.name
    out_file.close()

    cmd = [
        PIPER_PATH,
        "--model", MODEL_PATH,
        "--output_file", out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.communicate(input=text.encode("utf-8"))
    if proc.returncode != 0:
        raise RuntimeError("Piper TTS failed")
    return out_path
```

------------------------------------------------------------------------

### 7. Environment Variables (.env)

Create a `.env` file:

    GROQ_API_KEY=your_groq_api_key_here
    FAST_MODEL=llama-3.1-8b-instant
    ANSWER_MODEL=llama-3.3-70b-versatile
    SEARXNG_URL=http://localhost:8080

------------------------------------------------------------------------

## Running Phenora

Start server:

``` bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

    http://localhost:8000/docs

------------------------------------------------------------------------

## Using the API

Upload audio:

``` bash
curl -X POST http://localhost:8000/upload_audio -F "file=@speech.wav"
```

Process audio:

``` bash
curl -X POST http://localhost:8000/process
```

Debug endpoint:

``` bash
curl -X POST http://localhost:8000/debug_llm -H "Content-Type: application/json" -d '{"question": "what is the weather in London?"}'
```

------------------------------------------------------------------------

## Troubleshooting

**Whisper errors**

    sudo apt install ffmpeg

Use smaller model:

``` python
whisper_model = WhisperModel("base.en", compute_type="int8")
```

**LLM errors**

-   Check Groq API key
-   Ensure internet access

**SearXNG issues**

Check:

    docker ps

or open:

    http://localhost:8080

**Piper not working**

Test manually:

``` bash
echo "Hello" | ~/piper/piper --model ~/piper/en_US-lessac-medium.onnx --output_file test.wav
```

Make executable if needed:

``` bash
chmod +x ~/piper/piper
```

------------------------------------------------------------------------

## Contributing

-   Fork repository
-   Create feature branch

Examples:

    feature/new-feature
    fix/bug

------------------------------------------------------------------------


## Acknowledgements

-   FastAPI
-   Groq
-   faster-whisper
-   BeautifulSoup
-   SearXNG
-   Open-Meteo
-   Piper

------------------------------------------------------------------------

## Note

This project and parts of its implementation were developed with the
help of AI tools to accelerate development, debugging, and
documentation.

ESP32 hardware integration is coming soon -- we're working on a
companion device that will let you talk to Phenora from anywhere in your
home!
