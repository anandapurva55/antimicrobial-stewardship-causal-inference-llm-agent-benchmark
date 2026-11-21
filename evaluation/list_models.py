"""List the Gemini models your API key can use for text generation.

    python evaluation/list_models.py

Each trial stores the model it asked for (`model_requested`) and the name the API returned (`model_reported`)
in results/runs/<model>/<prompt>/trial_<k>/meta.json. Model documentation: https://ai.google.dev/gemini-api/docs/models
"""
import os
import sys

from dotenv import load_dotenv
from google import genai

if __name__ == "__main__":
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set (put it in .env)")
    for m in genai.Client(api_key=key).models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name.removeprefix("models/"))
