from fastapi import FastAPI, UploadFile
from dotenv import load_dotenv
import os, shutil, glob, subprocess, re, asyncio, sqlite3, tempfile, requests, logging, contextlib
from typing import Optional, List, Dict
from app.tts import speak

from faster_whisper import WhisperModel
from groq import Groq
from bs4 import BeautifulSoup
from datetime import datetime
import pytz

# ===================== TIMEZONE CONTEXT INJECTOR =====================
def get_clock_context() -> str:
    """Inject current time across key timezones into every request."""
    zones = {
        "UTC":             "UTC",
        "New York (EST)":  "America/New_York",
        "London (GMT)":    "Europe/London",
        "Dubai (GST)":     "Asia/Dubai",
        "India (IST)":     "Asia/Kolkata",
        "Tokyo (JST)":     "Asia/Tokyo",
        "Sydney (AEST)":   "Australia/Sydney",
        "Melbourne (AEST)":"Australia/Melbourne",
    }
    lines = []
    for label, tz in zones.items():
        now = datetime.now(pytz.timezone(tz))
        lines.append(f"{label}: {now.strftime('%A, %d %B %Y %I:%M %p')}")
    return "\n".join(lines)

# ===================== LOGGING =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phenora")

# ===================== ENV =====================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FAST_MODEL   = os.getenv("FAST_MODEL")    # llama-3.1-8b-instant   — summarization + intent
ANSWER_MODEL = os.getenv("ANSWER_MODEL")  # llama-3.3-70b-versatile — final user-facing answer

# ===================== APP =====================
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IN_DIR   = os.path.join(DATA_DIR, "in")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IN_DIR, exist_ok=True)

# ===================== MODELS =====================
whisper_model = WhisperModel("medium.en", compute_type="int8")
llm_client    = Groq(api_key=GROQ_API_KEY)

# ===================== SQLITE =====================
DB_PATH = os.path.join(DATA_DIR, "assistant_memory.db")

def get_db() -> sqlite3.Connection:
    """Fresh connection per call — thread-safe."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT,
            role       TEXT,
            content    TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def save_message(user: str, role: str, content: str) -> None:
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
        (user, role, content)
    )
    conn.commit()
    conn.close()

def load_recent(user: str, limit: int = 6) -> List[Dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user, limit)
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

# ===================== AUDIO =====================
@contextlib.contextmanager
def normalized_wav(src: str):
    """Convert any audio to 16kHz mono WAV, always clean up on exit."""
    dst = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", dst],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        yield dst
    finally:
        if os.path.exists(dst):
            os.unlink(dst)

# ===================== RESPONSE SANITIZER =====================
def clean_response(text: str) -> str:
    """Strip reasoning traces and markdown — output must be plain speakable text."""

    # Remove complete <think>...</think> blocks
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)

    # Handle unclosed <think> block
    if "<think>" in text:
        before = text[:text.find("<think>")].strip()
        text = before if before else ""

    # Strip markdown bold / italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)

    # Strip markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Strip bullet points
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)

    # Strip numbered lists
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Strip inline label headers like "Key Developments:" on their own line
    text = re.sub(r"^[A-Z][^\n]{0,60}:\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple newlines into a single space
    text = re.sub(r"\n+", " ", text)

    return text.strip()

# ===================== HELPERS =====================
def truncate(text: str, max_chars: int) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text

# ===================== LLM =====================
async def call_llm(messages: List[Dict], max_tokens: int = 350, model: str = None) -> str:
    """
    model=None       → ANSWER_MODEL (llama-3.3-70b-versatile)
    model=FAST_MODEL → FAST_MODEL   (llama-3.1-8b-instant)
    Auto-falls back to FAST_MODEL if ANSWER_MODEL is rate-limited or exhausted.
    """
    use_model = model or ANSWER_MODEL

    async def _try(m: str) -> str:
        def _call():
            return llm_client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=0.5,
                max_tokens=max_tokens
            )
        r = await asyncio.to_thread(_call)
        return clean_response(r.choices[0].message.content)

    try:
        return await _try(use_model)
    except Exception as e:
        if use_model == ANSWER_MODEL and FAST_MODEL != ANSWER_MODEL:
            logger.warning(f"{ANSWER_MODEL} failed ({e}) — falling back to {FAST_MODEL}")
            return await _try(FAST_MODEL)
        raise

# ===================== WEATHER =====================
def extract_city(q: str) -> Optional[str]:
    q_lower = q.lower()
    patterns = [
        r"weather (?:in|at|for) ([a-zA-Z\s]+)",
        r"temperature (?:in|at|for) ([a-zA-Z\s]+)",
        r"\bin ([a-zA-Z\s]+)",
        r"\bat ([a-zA-Z\s]+)",
        r"\bfor ([a-zA-Z\s]+)",
    ]
    for p in patterns:
        m = re.search(p, q_lower)
        if m:
            city = m.group(1).strip()
            city = re.sub(r"\b(today|now|currently|please|tomorrow)\b", "", city).strip()
            if 2 < len(city) < 40:
                return city.title()
    return None

def get_weather(city: str) -> Optional[str]:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10
        ).json()

        if not geo.get("results"):
            logger.warning(f"No geo results for: {city}")
            return None

        result   = geo["results"][0]
        lat      = result["latitude"]
        lon      = result["longitude"]
        resolved = result.get("name", city)

        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=10
        ).json()

        cw = w.get("current_weather")
        if not cw:
            return None

        return (
            f"Right now in {resolved}, the temperature is {cw['temperature']}°C "
            f"with wind speed of {cw['windspeed']} km/h."
        )
    except Exception as e:
        logger.warning(f"Weather fetch failed for {city}: {e}")
        return None

# ===================== WIKIPEDIA =====================
def wikipedia_search(query: str) -> Optional[str]:
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" +
            query.replace(" ", "_"),
            headers={"User-Agent": "Phenora/1.0"},
            timeout=10
        )
        if r.status_code == 200:
            extract = r.json().get("extract")
            if extract:
                logger.info(f"Wikipedia direct hit: {query}")
                return extract

        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "format":   "json",
                "srlimit":  1
            },
            headers={"User-Agent": "Phenora/1.0"},
            timeout=10
        ).json()

        results = search.get("query", {}).get("search", [])
        if not results:
            return None

        title = results[0]["title"]
        logger.info(f"Wikipedia fallback hit: {title}")

        r2 = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" +
            title.replace(" ", "_"),
            headers={"User-Agent": "Phenora/1.0"},
            timeout=10
        )
        if r2.status_code == 200:
            return r2.json().get("extract")

        return None
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}")
        return None

# ===================== WEB SCRAPING =====================
def read_article(url: str) -> str:
    try:
        html = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        ).text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "header"]):
            tag.decompose()

        content_tags = soup.find_all(["p", "h2", "h3", "li", "blockquote", "article"])
        paragraphs = [
            t.get_text().strip()
            for t in content_tags
            if len(t.get_text().strip()) > 40
        ]

        return " ".join(paragraphs[:25])
    except Exception as e:
        logger.warning(f"Article read failed ({url}): {e}")
        return ""

# ===================== SEARXNG =====================
def searxng_search_urls(query: str) -> List[Dict]:
    """Return list of {url, title, date} sorted freshest first."""
    try:
        r = requests.get(
            "http://localhost:8080/search",
            params={
                "q":          query,
                "format":     "json",
                "language":   "en",
                "time_range": "month"
            },
            timeout=10
        ).json()

        results = []
        for item in r.get("results", []):
            url = item.get("url", "")
            if not url:
                continue
            if any(x in url for x in ["youtube.com", "twitter.com", "facebook.com", "instagram.com"]):
                continue
            results.append({
                "url":   url,
                "title": item.get("title", ""),
                "date":  item.get("publishedDate", "")
            })

        results.sort(key=lambda x: x["date"] or "", reverse=True)
        return results[:15]
    except Exception as e:
        logger.warning(f"SearXNG failed for '{query}': {e}")
        return []

# ===================== INTENT DETECTOR =====================
async def detect_intent(transcript: str) -> str:
    """
    Classify the question to decide which pipeline to run.
    REALTIME_DATA  → needs web search (news, current events, live data)
    KNOWLEDGE      → model + clock context is enough (time, math, facts, general advice)
    CONVERSATIONAL → chit-chat, greetings, personal questions
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the user's question into exactly one category.\n"
                "Output ONLY the category name, nothing else.\n\n"
                "REALTIME_DATA — needs current news, live events, recent developments, "
                "sports scores, breaking news, weather, current prices, "
                "vacation spots considering current safety/events\n"
                "KNOWLEDGE — time zones, math, science, history, definitions, "
                "how things work, general recommendations not tied to current events\n"
                "CONVERSATIONAL — chit-chat, greetings, personal questions, opinions\n\n"
                "Examples:\n"
                "what time is it in Tokyo → KNOWLEDGE\n"
                "convert 3pm EST to IST → KNOWLEDGE\n"
                "what is the speed of light → KNOWLEDGE\n"
                "best movies to watch → KNOWLEDGE\n"
                "what is happening in Gaza right now → REALTIME_DATA\n"
                "who won the game last night → REALTIME_DATA\n"
                "top news today → REALTIME_DATA\n"
                "what is the weather in Dubai → REALTIME_DATA\n"
                "best vacation spot with good weather today → REALTIME_DATA\n"
                "how are you → CONVERSATIONAL\n"
                "what is your name → CONVERSATIONAL"
            )
        },
        {"role": "user", "content": transcript}
    ]
    result = await call_llm(messages, max_tokens=10, model=FAST_MODEL)
    intent = result.strip().upper()

    if intent not in ("REALTIME_DATA", "KNOWLEDGE", "CONVERSATIONAL"):
        logger.warning(f"Unknown intent '{intent}' — defaulting to REALTIME_DATA")
        intent = "REALTIME_DATA"

    logger.info(f"[Intent] {intent}")
    return intent

# ===================== QUERY OPTIMIZER =====================
async def optimize_search_queries(transcript: str) -> List[str]:
    """Generate 3 targeted search queries using the answer model for reliability."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query generator. You only output search queries, nothing else.\n"
                "Do NOT explain yourself. Do NOT say you cannot do something.\n"
                "Do NOT write sentences. ONLY output 3 short search queries.\n\n"
                "Format — exactly 3 lines, one query per line:\n"
                "<query 1>\n"
                "<query 2>\n"
                "<query 3>\n\n"
                "Rules:\n"
                "- Query 1: most recent news angle + year 2025\n"
                "- Query 2: specific names, places, or events in the question\n"
                "- Query 3: broader context angle\n"
                "- Max 6 words each. No punctuation. No numbering. No explanations.\n\n"
                "Examples:\n"
                "User: what happened with Khamenei and the US\n"
                "Khamenei death US Iran 2025\n"
                "Ayatollah Khamenei latest news\n"
                "US Iran conflict 2025\n\n"
                "User: top 5 key news topics today\n"
                "top world news today 2025\n"
                "breaking news headlines March 2025\n"
                "major global events this week\n\n"
                "User: best vacation destination good weather today\n"
                "best weather vacation destinations March 2025\n"
                "warm sunny travel destinations now\n"
                "top holiday spots good climate March"
            )
        },
        {"role": "user", "content": transcript}
    ]
    # Use answer model — more reliable for diverse question types
    result = await call_llm(messages, max_tokens=60, model=ANSWER_MODEL)

    queries = []
    for line in result.strip().split("\n"):
        line = line.strip().strip('"').strip("'")
        if line and len(line) < 60 and len(line.split()) <= 8:
            queries.append(line)

    if not queries:
        logger.warning("Query optimizer returned invalid output — using transcript fallback")
        words = transcript.strip().rstrip("?").strip()
        queries = [
            f"{words[:50]} 2025",
            f"{words[:50]} latest",
            f"{words[:50]} today"
        ]

    logger.info(f"Optimized queries: {queries}")
    return queries[:3]

# ===================== PER-ARTICLE SUMMARIZER =====================
async def summarize_article(article_text: str, query: str) -> Optional[str]:
    """Condense one article to 3-4 key sentences — uses fast model."""
    if not article_text or len(article_text) < 100:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You are a news summarizer.\n"
                "Extract only the facts directly relevant to the user's question.\n"
                "Output 3 to 4 plain sentences. No bullet points, no headers.\n"
                "If the article contains nothing relevant, output exactly: IRRELEVANT"
            )
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nArticle:\n{truncate(article_text, 2000)}"
        }
    ]
    result = await call_llm(messages, max_tokens=150, model=FAST_MODEL)
    if not result or "IRRELEVANT" in result:
        return None
    return result

# ===================== BATCHED SUMMARIZER =====================
async def summarize_articles_batched(articles: List[str], query: str, batch_size: int = 3) -> List[Optional[str]]:
    """
    Summarize in small batches with pause between each.
    Prevents TPM spikes on llama-3.1-8b-instant (6,000 TPM limit).
    """
    summaries = []
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        logger.info(f"Summarizing batch {i // batch_size + 1} ({len(batch)} articles) with {FAST_MODEL}...")
        batch_results = await asyncio.gather(
            *[summarize_article(article, query) for article in batch]
        )
        summaries.extend(batch_results)
        if i + batch_size < len(articles):
            await asyncio.sleep(1.5)
    return summaries

# ===================== LOCATION EXTRACTOR =====================
async def extract_locations(text: str) -> List[str]:
    """Pull country and city names out of text for safety checking."""
    messages = [
        {
            "role": "system",
            "content": (
                "Extract all country and city names from the text.\n"
                "Output only the names, one per line, nothing else.\n"
                "If none found, output exactly: NONE"
            )
        },
        {"role": "user", "content": text}
    ]
    result = await call_llm(messages, max_tokens=50, model=FAST_MODEL)
    if not result or "NONE" in result:
        return []
    return [l.strip() for l in result.strip().split("\n") if l.strip()]

# ===================== SAFETY CONTEXT CHECK =====================
async def check_location_safety(locations: List[str]) -> Optional[str]:
    """
    Check if any recommended locations have active conflicts,
    travel warnings, or military strikes.
    """
    if not locations:
        return None

    safety_queries = [f"{loc} travel warning conflict 2025" for loc in locations[:3]]

    all_results = await asyncio.gather(
        *[asyncio.to_thread(searxng_search_urls, q) for q in safety_queries]
    )

    seen  = set()
    urls  = []
    for result_list in all_results:
        for item in result_list:
            if item["url"] not in seen:
                seen.add(item["url"])
                urls.append(item)

    urls = urls[:6]
    if not urls:
        return None

    articles = await asyncio.gather(
        *[asyncio.to_thread(read_article, u["url"]) for u in urls]
    )

    summaries_raw = await asyncio.gather(
        *[summarize_article(a, f"travel warnings conflict danger {' '.join(locations)}")
          for a in articles]
    )
    summaries = [s for s in summaries_raw if s]

    if not summaries:
        return None

    logger.warning(f"Safety context found for: {locations}")
    return "\n".join(summaries)

# ===================== SUMMARY REDUCER =====================
async def merge_summaries(
    query: str,
    summaries: List[str],
    weather: Optional[str],
    history: List[Dict],
    safety_context: Optional[str] = None
) -> str:
    """Merge all summaries + model knowledge into one final spoken answer."""
    combined = "\n\n".join(f"[Source {i+1}]: {s}" for i, s in enumerate(summaries))
    clock    = get_clock_context()

    system_content = (
        "You are Phenora, an intelligent voice assistant.\n\n"
        "You are given summaries of recent news articles and asked to answer a question.\n\n"
        "Rules:\n"
        "- Prioritize the most recent and specific facts from the sources.\n"
        "- Fill in historical context and background from your own knowledge.\n"
        "- Merge both into one single fluent answer.\n"
        "- Never use markdown — no bullet points, bold, headers, or numbered lists.\n"
        "- Write in plain flowing sentences only. Your response will be spoken aloud.\n"
        "- Aim for 4 to 6 sentences. Never shorter, never longer.\n"
        "- Start your answer immediately with the substance — no preamble.\n"
        "- Never say 'according to', 'sources say', 'based on', 'I found', or 'the user is asking'.\n"
        "- Never narrate what you are doing. Just answer.\n"
        "- Always complete your answer fully.\n\n"
        f"Current Date and Time:\n{clock}"
    )

    if weather:
        system_content += f"\n\nWeather context if relevant: {weather}"

    if safety_context:
        system_content += (
            f"\n\nCRITICAL SAFETY CONTEXT — read before answering:\n{safety_context}\n"
            "If any recommended location has active conflict, travel warnings, or military strikes, "
            "you MUST mention this clearly before recommending it. "
            "Do not recommend locations that are currently under attack or have active danger."
        )

    messages = (
        [{"role": "system", "content": system_content}]
        + history
        + [{"role": "user", "content": f"Question: {query}\n\nNews Summaries:\n{combined}"}]
    )

    return await call_llm(messages, max_tokens=350, model=ANSWER_MODEL)

# ===================== FALLBACK ANSWER (no web data / knowledge questions) =====================
async def generate_final_answer(
    query: str,
    weather: Optional[str],
    history: List[Dict]
) -> str:
    """Used for KNOWLEDGE, CONVERSATIONAL intents, or when web search returns nothing."""
    clock = get_clock_context()

    system_content = (
        "You are Phenora, an intelligent voice assistant.\n\n"
        "Answer the user's message directly and naturally.\n\n"
        "Rules:\n"
        "- Never use markdown — no bullet points, bold, headers, or numbered lists.\n"
        "- Write in plain flowing sentences only. Your response will be spoken aloud.\n"
        "- Aim for 4 to 6 sentences. Never shorter, never longer.\n"
        "- Start your answer immediately with the substance — no preamble.\n"
        "- Never say 'the user is asking', 'based on your question', 'according to', or 'I found'.\n"
        "- Never narrate what you are doing. Just answer.\n"
        "- Always complete your answer fully.\n\n"
        f"Current Date and Time:\n{clock}"
    )

    if weather:
        system_content += f"\n\nWeather data: {weather}"

    messages = (
        [{"role": "system", "content": system_content}]
        + history
        + [{"role": "user", "content": query}]
    )

    return await call_llm(messages, max_tokens=350, model=ANSWER_MODEL)

# ===================== API ROUTES =====================
@app.post("/upload_audio")
async def upload_audio(file: UploadFile):
    path = os.path.join(IN_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Uploaded: {file.filename}")
    return {"status": "ok"}


@app.post("/process")
async def process_audio(device_id: Optional[str] = None):
    user = device_id or "local_user"

    # --- Find latest uploaded file ---
    files = glob.glob(os.path.join(IN_DIR, "*"))
    if not files:
        return {"error": "no audio"}

    latest = max(files, key=os.path.getctime)

    # --- Transcribe ---
    with normalized_wav(latest) as wav:
        segments, _ = whisper_model.transcribe(wav, beam_size=5)
        transcript  = " ".join(s.text for s in segments).strip()

    # --- Clean up uploaded file + Zone.Identifier ---
    os.unlink(latest)
    zone_id = latest + ":Zone.Identifier"
    if os.path.exists(zone_id):
        os.unlink(zone_id)

    logger.info(f"[{user}] Transcript: {transcript}")
    save_message(user, "user", transcript)

    history = load_recent(user, limit=6)

    # --- Detect intent first ---
    intent = await detect_intent(transcript)

    # ================================================================
    # KNOWLEDGE / CONVERSATIONAL — skip web search entirely
    # ================================================================
    if intent in ("KNOWLEDGE", "CONVERSATIONAL"):
        logger.info(f"[{user}] Skipping web search for intent: {intent}")
        final = await generate_final_answer(transcript, None, history)

    # ================================================================
    # REALTIME_DATA — full web pipeline
    # ================================================================
    else:
        possible_city = extract_city(transcript)
        weather_task  = (
            asyncio.to_thread(get_weather, possible_city)
            if possible_city
            else asyncio.sleep(0, result=None)
        )

        queries, weather_data = await asyncio.gather(
            optimize_search_queries(transcript),
            weather_task
        )

        all_url_results = await asyncio.gather(
            *[asyncio.to_thread(searxng_search_urls, q) for q in queries]
        )

        seen_urls   = set()
        unique_urls = []
        for url_list in all_url_results:
            for item in url_list:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    unique_urls.append(item)

        unique_urls = unique_urls[:9]
        logger.info(f"[{user}] Unique URLs to scrape: {len(unique_urls)}")

        raw_articles = await asyncio.gather(
            *[asyncio.to_thread(read_article, u["url"]) for u in unique_urls]
        )

        summaries_raw = await summarize_articles_batched(raw_articles, transcript, batch_size=3)
        summaries     = [s for s in summaries_raw if s]
        logger.info(f"[{user}] Relevant summaries: {len(summaries)} / {len(unique_urls)}")

        # --- Safety check for location-based recommendations ---
        safety_context = None
        if summaries:
            combined_text = " ".join(summaries)
            locations     = await extract_locations(combined_text)
            if locations:
                logger.info(f"[{user}] Checking safety for: {locations}")
                safety_context = await check_location_safety(locations)

        # --- Final answer ---
        if summaries:
            final = await merge_summaries(transcript, summaries, weather_data, history, safety_context)
        else:
            logger.info(f"[{user}] No web data — falling back to model knowledge")
            final = await generate_final_answer(transcript, weather_data, history)

    logger.info(f"[{user}] Answer: {final}")
    save_message(user, "assistant", final)

    audio = speak(final)
    return {"answer": final, "audio": audio}


# ===================== DEBUG =====================
@app.post("/debug_llm")
async def debug_llm(payload: dict):
    """
    Test endpoint — inspect every stage of the pipeline.
    POST {"question": "your question here"}
    """
    transcript = payload.get("question", "")

    intent  = await detect_intent(transcript)
    queries = await optimize_search_queries(transcript)

    all_url_results = await asyncio.gather(
        *[asyncio.to_thread(searxng_search_urls, q) for q in queries]
    )

    seen_urls   = set()
    unique_urls = []
    for url_list in all_url_results:
        for item in url_list:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                unique_urls.append(item)

    unique_urls  = unique_urls[:9]
    raw_articles = await asyncio.gather(
        *[asyncio.to_thread(read_article, u["url"]) for u in unique_urls]
    )

    summaries_raw = await summarize_articles_batched(raw_articles, transcript, batch_size=3)
    summaries     = [s for s in summaries_raw if s]

    def _raw_call():
        return llm_client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": transcript}
            ],
            temperature=0.5,
            max_tokens=350
        )

    r   = await asyncio.to_thread(_raw_call)
    raw = r.choices[0].message.content

    return {
        "intent":             intent,
        "fast_model":         FAST_MODEL,
        "answer_model":       ANSWER_MODEL,
        "queries":            queries,
        "urls_found":         len(unique_urls),
        "url_list":           [u["url"] for u in unique_urls],
        "articles_scraped":   len([a for a in raw_articles if a]),
        "summaries_kept":     len(summaries),
        "summaries_preview":  [s[:200] for s in summaries[:3]],
        "raw_llm_output":     raw,
        "after_clean":        clean_response(raw),
        "has_think_tags":     "<think>" in raw,
        "think_stripped":     "<think>" not in clean_response(raw),
        "clock_context":      get_clock_context(),
    }
