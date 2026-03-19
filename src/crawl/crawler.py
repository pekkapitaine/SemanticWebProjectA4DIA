"""
Crawler for AI Research domain.

Fetches pages from seed URLs, respects robots.txt, extracts main content
using trafilatura, and stores results in JSONL format.

Output: data/crawler_output.jsonl
"""

import json
import logging
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED_URLS = [
    # HuggingFace blog — crawl-friendly, rich in AI research content
    "https://huggingface.co/blog/bert-101",
    "https://huggingface.co/blog/large-language-models",
    "https://huggingface.co/blog/rlhf",
    "https://huggingface.co/blog/llama2",
    "https://huggingface.co/blog/falcon",
    "https://huggingface.co/blog/mixtral",
    "https://huggingface.co/blog/open-llm-leaderboard-v2",
    "https://huggingface.co/blog/gemma",
    "https://huggingface.co/blog/starcoder",
    "https://huggingface.co/blog/inference-endpoints-llm",
]

OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "crawler_output.jsonl"
MIN_WORD_COUNT = 500
CRAWL_DELAY = 1.5  # seconds between requests (polite crawling)
USER_AGENT = "SemanticWebBot/1.0 (educational project; respectful crawler)"

# ---------------------------------------------------------------------------
# Robots.txt helper
# ---------------------------------------------------------------------------

_robots_cache: dict[str, RobotFileParser] = {}


def can_fetch(url: str) -> bool:
    """Return True if robots.txt allows fetching this URL."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(urljoin(base, "/robots.txt"))
        try:
            rp.read()
        except Exception:
            # If robots.txt is unreachable, assume allowed
            _robots_cache[base] = None
            return True
        _robots_cache[base] = rp
    rp = _robots_cache[base]
    if rp is None:
        return True
    return rp.can_fetch(USER_AGENT, url)


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def fetch_and_extract(url: str, client: httpx.Client) -> dict | None:
    """
    Fetch a URL and extract main content.
    Handles Wikipedia API JSON responses separately from regular HTML pages.
    Returns a dict with url, title, text, word_count, or None if extraction fails.
    """
    # Wikipedia API is explicitly designed for programmatic access (Allow: /w/api.php)
    is_wikipedia_api = "wikipedia.org/w/api.php" in url
    if not is_wikipedia_api and not can_fetch(url):
        logger.warning(f"Blocked by robots.txt: {url}")
        return None

    # Wikipedia REST API — extract plain text from JSON response
    if is_wikipedia_api:
        return _fetch_wikipedia_api(url, client)

    try:
        response = client.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Fetch error for {url}: {e}")
        return None

    html = response.text
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        output_format="json",
        with_metadata=True,
    )

    if not extracted:
        logger.warning(f"trafilatura returned nothing for {url}")
        return None

    data = json.loads(extracted)
    text = data.get("text", "")
    word_count = len(text.split())

    if word_count < MIN_WORD_COUNT:
        logger.info(f"Skipped (too short, {word_count} words): {url}")
        return None

    return {
        "url": url,
        "title": data.get("title", ""),
        "text": text,
        "word_count": word_count,
        "date": data.get("date", ""),
        "hostname": data.get("hostname", ""),
    }


def _fetch_wikipedia_api(url: str, client: httpx.Client) -> dict | None:
    """Fetch a Wikipedia API URL and extract plain text from the JSON response."""
    try:
        response = client.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Wikipedia API error for {url}: {e}")
        return None

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        title = page.get("title", "")
        text = page.get("extract", "")
        if not text:
            logger.warning(f"No extract for Wikipedia page: {title}")
            return None
        word_count = len(text.split())
        if word_count < MIN_WORD_COUNT:
            logger.info(f"Skipped Wikipedia page (too short, {word_count} words): {title}")
            return None
        return {
            "url": url,
            "title": title,
            "text": text,
            "word_count": word_count,
            "date": "",
            "hostname": "en.wikipedia.org",
        }
    return None


# ---------------------------------------------------------------------------
# Main crawl loop
# ---------------------------------------------------------------------------

def crawl(seed_urls: list[str] = SEED_URLS, output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0

    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        with open(output_path, "w", encoding="utf-8") as out_file:
            for url in seed_urls:
                logger.info(f"Crawling: {url}")
                result = fetch_and_extract(url, client)
                if result:
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    saved += 1
                    logger.info(f"Saved ({result['word_count']} words): {url}")
                time.sleep(CRAWL_DELAY)

    logger.info(f"Done. {saved}/{len(seed_urls)} pages saved to {output_path}")


if __name__ == "__main__":
    crawl()
