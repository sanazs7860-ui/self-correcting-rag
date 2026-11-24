import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)   # .env always wins over old system variables


def _clean(v):
    return (v or "").strip().strip("\"'")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

GEMINI_API_KEY = _clean(os.getenv("GEMINI_API_KEY"))
GROQ_API_KEY = _clean(os.getenv("GROQ_API_KEY"))
TAVILY_API_KEY = _clean(os.getenv("TAVILY_API_KEY"))
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Models
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GRADER_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
GENERATOR_PROVIDER = "groq"
GROQ_GENERATOR_MODEL = os.getenv("GROQ_GENERATOR_MODEL", "openai/gpt-oss-120b")
GEMINI_GENERATOR_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
DEFAULT_MAX_RETRIES = 1