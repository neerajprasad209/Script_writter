import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from config.path import ENV_PATH
from utils.logger import logger

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")


def get_gemini_llm():
    """
    Initialize Gemini LLM using LangChain.

    Returns:
        ChatGoogleGenerativeAI: Gemini LLM instance
    """

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )

        logger.info("Gemini LLM initialized successfully")

        return llm

    except Exception:
        logger.exception("Failed to initialize Gemini LLM")
        raise