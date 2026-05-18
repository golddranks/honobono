"""Fetch episodes 1-20 of https://ncode.syosetu.com/n9629ex/ into ./episodes/.

File format matches `read_episode` in tools/common.py:
    line 1: heading (episode subtitle)
    line 2: blank
    line 3+: body paragraphs separated by blank lines
"""

import html
import re
import sys
import time
import urllib.request
from pathlib import Path

EPISODES_DIR = Path(__file__).resolve().parent.parent / "episodes"
NCODE = "n9629ex"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

TITLE_RE = re.compile(r'<h1 class="p-novel__title[^"]*">(.*?)</h1>', re.S)
BODY_RE = re.compile(r'<div class="js-novel-text p-novel__text">(.*?)</div>', re.S)
PARA_RE = re.compile(r'<p id="L\d+">(.*?)</p>', re.S)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse(page: str) -> tuple[str, str]:
    tm = TITLE_RE.search(page)
    bm = BODY_RE.search(page)
    if not tm or not bm:
        raise ValueError("missing title or body — likely an error/busy page")
    title = html.unescape(tm.group(1)).strip()
    paragraphs = [
        # <br /> inside a <p> means a blank line within that paragraph block.
        html.unescape(re.sub(r"<br\s*/?>", "", p)).strip()
        for p in PARA_RE.findall(bm.group(1))
    ]
    body = "\n".join(paragraphs)
    return title, body


def fetch_episode(seq: int) -> tuple[str, str]:
    # The site's /1/ is a world-info preface; 1話 lives at /2/, etc.
    url = f"https://ncode.syosetu.com/{NCODE}/{seq + 1}/"
    for attempt in range(5):
        try:
            return parse(fetch(url))
        except (ValueError, OSError) as e:
            wait = 5 * (attempt + 1)
            print(f"  ep{seq}: {type(e).__name__}, retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"ep{seq}: gave up after retries")


def main() -> int:
    EPISODES_DIR.mkdir(exist_ok=True)
    for seq in range(1, 21):
        out = EPISODES_DIR / f"{seq:03d}_{NCODE}_ep{seq:03d}.txt"
        if out.exists():
            print(f"ep{seq}: already present, skipping", flush=True)
            continue
        title, body = fetch_episode(seq)
        out.write_text(f"{title}\n\n{body}\n", encoding="utf-8")
        print(f"ep{seq}: {title} ({len(body)} chars)", flush=True)
        time.sleep(1.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
