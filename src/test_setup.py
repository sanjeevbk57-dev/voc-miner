"""Check that the API key, Claude and the Play Store scraper all work.

Run from the voc-miner folder:  python src/test_setup.py
"""
from dotenv import load_dotenv
load_dotenv()

import anthropic
from google_play_scraper import reviews

client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=50,
    messages=[{"role": "user", "content": "Say 'setup works'"}])
print("API:", msg.content[0].text)

data, _ = reviews("com.zeptoconsumerapp", lang="en", country="in", count=3)
print("Scraper:", len(data), "reviews fetched")
