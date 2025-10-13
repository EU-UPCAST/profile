# scrape_ai_news.py
# Scrapes TLDR:AI newsletter issues into CSV format.
# Strategy: enumerate date-based issue URLs and fetch full HTML/text content.

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Iterable, List, Optional, Tuple, Dict
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import feedparser
from dateutil import parser as dateparser

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (AI-news-scraper)"
)

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def norm_dt(dt) -> Optional[datetime]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc)
    try:
        return dateparser.parse(str(dt)).astimezone(timezone.utc)
    except Exception:
        return None

def within(dt: Optional[datetime], since: Optional[datetime], until: Optional[datetime]) -> bool:
    if dt is None:
        return True  # keep undated items unless user is strict; easier for later manual filtering
    if since and dt < since:
        return False
    if until and dt > until:
        return False
    return True

@dataclass
class Item:
    source: str
    date: Optional[datetime]
    title: str
    url: str
    summary: str
    page_html: str = ""
    page_text: str = ""

def feed_candidates_from_html(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    feeds = []
    for link in soup.find_all("link", rel=lambda v: v and "alternate" in v):
        t = (link.get("type") or "").lower()
        if any(x in t for x in ["rss", "atom", "json"]):
            href = link.get("href")
            if href:
                feeds.append(requests.compat.urljoin(base_url, href))
    return list(dict.fromkeys(feeds))

def try_get(url: str, session: requests.Session, timeout=20) -> Optional[requests.Response]:
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception:
        return None
    return None

def discover_feeds(base_url: str, session: requests.Session) -> List[str]:
    feeds = []

    # common defaults by platform
    candidates = [
        base_url.rstrip("/") + "/feed",           # Substack, many CMS
        base_url.rstrip("/") + "/rss",            # common
        base_url.rstrip("/") + "/rss.xml",        # common
        base_url.rstrip("/") + "/atom.xml",       # common
        base_url.rstrip("/") + "/index.xml",      # Hugo/others
        base_url.rstrip("/") + "/feed.xml",       # Jekyll/others
    ]

    # fetch homepage and parse <link rel="alternate">
    home = try_get(base_url, session)
    if home is not None:
        feeds.extend(feed_candidates_from_html(home.text, base_url))

    for c in candidates:
        r = try_get(c, session)
        if r is not None and r.headers.get("content-type", "").lower().find("html") == -1:
            feeds.append(c)

    # de-dup
    feeds = list(dict.fromkeys(feeds))
    return feeds

def harvest_from_feed(source_name: str, feed_url: str, since: Optional[datetime], until: Optional[datetime]) -> List[Item]:
    parsed = feedparser.parse(feed_url)
    items: List[Item] = []
    for e in parsed.entries:
        dt = None
        for cand in [getattr(e, "published", None), getattr(e, "updated", None), getattr(e, "created", None)]:
            dt = norm_dt(cand)
            if dt:
                break
        title = getattr(e, "title", "") or ""
        link = getattr(e, "link", "") or ""
        summary = BeautifulSoup(getattr(e, "summary", "") or getattr(e, "content", [{}])[0].get("value", ""), "html.parser").get_text(" ", strip=True)
        if within(dt, since, until):
            items.append(Item(source=source_name, date=dt, title=title.strip(), url=link.strip(), summary=summary.strip()))
    return items

# ---------- Fallback scrapers (simple, resilient, minimal selectors) ----------








def scrape_tldr_ai_by_dates(session: requests.Session, since: Optional[datetime], until: Optional[datetime]) -> List[Item]:
    """Enumerate TLDR issue URLs by date instead of relying on archive listings."""
    items: List[Item] = []
    if until is None:
        until = datetime.now(timezone.utc)
    if since is None:
        since = until - timedelta(days=365)

    start = since.date()
    end = until.date()
    if start > end:
        start, end = end, start

    templates = [
        "https://tldr.tech/ai/{date}",
        "https://www.tldrnewsletter.com/ai/{date}",
        "https://www.tldrnewsletter.com/newsletters/ai/{date}",
    ]

    seen_urls = set()

    for offset in range((end - start).days + 1):
        current = start + timedelta(days=offset)
        slug = current.strftime('%Y-%m-%d')
        for tpl in templates:
            url = tpl.format(date=slug)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            resp = try_get(url, session)
            if resp is None:
                continue
            html = resp.text
            title = extract_title(html) or f"TLDR AI {slug}"
            text_content = extract_main_text(html, url=url)
            dt = datetime.combine(current, datetime.min.time(), tzinfo=timezone.utc)
            summary = text_content[:500] if text_content else ""
            items.append(Item(
                source="TLDR: AI",
                date=dt,
                title=title,
                url=url,
                summary=summary,
                page_html=html,
                page_text=text_content,
            ))
            break
    return items




# ---------- Source definitions ----------

@dataclass
class Source:
    name: str
    base_urls: List[str]
    preferred_feeds: List[str]  # we’ll try these first
    fallback_scraper: callable

SOURCES: List[Source] = [
    Source(
        name="TLDR: AI",
        base_urls=[],
        preferred_feeds=[],
        fallback_scraper=scrape_tldr_ai_by_dates,
    ),
]

def collect_source(source: Source, since: Optional[datetime], until: Optional[datetime], session: requests.Session, delay: float = 0.0) -> List[Item]:
    all_items: List[Item] = []
    tried_feed = set()

    # Try preferred feeds first
    for f in source.preferred_feeds:
        if f in tried_feed:
            continue
        tried_feed.add(f)
        items = harvest_from_feed(source.name, f, since, until)
        if items:
            all_items.extend(items)

    # If still empty, attempt discovery from base URLs
    if not all_items:
        for base in source.base_urls:
            for feed in discover_feeds(base, session):
                if feed in tried_feed:
                    continue
                tried_feed.add(feed)
                items = harvest_from_feed(source.name, feed, since, until)
                if items:
                    all_items.extend(items)
            if all_items:
                break

    # If still empty, fall back to archive scrape
    if not all_items and source.fallback_scraper:
        all_items.extend(source.fallback_scraper(session, since, until))

    # basic de-dup by url
    dedup: Dict[str, Item] = {}
    for it in all_items:
        if it.url and it.url not in dedup:
            dedup[it.url] = it

    # politeness
    if delay > 0:
        time.sleep(delay)

    return list(dedup.values())



def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for selector in ["meta[property='og:title']", "meta[name='twitter:title']"]:
        tag = soup.select_one(selector)
        if tag and tag.get('content'):
            return tag['content'].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find('h1')
    if h1:
        return h1.get_text(' ', strip=True)
    return ""


def extract_main_text(html: str, url: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    candidates = []
    seen_texts = set()

    def maybe_add(node, min_len: int = 200):
        if not node:
            return
        text = node.get_text(" ", strip=True)
        if not text:
            return
        text = re.sub(r"\s+", " ", text)
        if len(text) >= min_len:
            trimmed = text[:20000]
            if trimmed in seen_texts:
                return
            seen_texts.add(trimmed)
            candidates.append((len(text), trimmed))

    article = soup.find("article")
    maybe_add(article)
    main = soup.find("main")
    maybe_add(main)

    host = ""
    if url:
        host = urlparse(url).netloc.lower()
    if host:
        if "tldrnewsletter" in host or host.endswith("tldr.tech"):
            # TLDR issues tend to place the full content inside main sections; keep generous candidates
            for attr in ["data-testid", "data-test", "class", "id"]:
                for node in soup.find_all(attrs={attr: re.compile(r"(body|content|newsletter|issue)", re.I)}):
                    maybe_add(node, min_len=100)
            if main:
                for node in main.find_all(["section", "div", "article"]):
                    maybe_add(node, min_len=100)

    for selector in ["section", "div"]:
        for node in soup.find_all(selector):
            maybe_add(node)

    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:20000]


def enrich_with_full_page(items: List[Item], session: requests.Session, delay: float = 0.0, timeout: float = 30.0) -> None:
    """Populate each item with the HTML contents and extracted text of its URL.

    We memoize per-URL to avoid duplicate requests when entries share the same link.
    """
    cache: Dict[str, Tuple[str, str]] = {}
    for item in items:
        url = item.url
        if not url:
            item.page_html = ""
            item.page_text = ""
            continue
        if item.page_html:
            if not item.page_text:
                item.page_text = extract_main_text(item.page_html, url=url)
            cache[url] = (item.page_html, item.page_text)
            continue
        if url in cache:
            html, text_extract = cache[url]
            item.page_html = html
            item.page_text = text_extract
            continue
        resp = try_get(url, session, timeout=timeout)
        if resp is None:
            html = ""
            text_extract = ""
        else:
            html = resp.text
            text_extract = extract_main_text(html, url=url)
        item.page_html = html
        item.page_text = text_extract
        cache[url] = (html, text_extract)
        if delay > 0:
            time.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description="Scrape TLDR:AI newsletter issues into CSV.")
    ap.add_argument("--since", type=str, default=None, help="Earliest date (YYYY-MM-DD)")
    ap.add_argument("--until", type=str, default=None, help="Latest date (YYYY-MM-DD)")
    ap.add_argument("--limit-per-source", type=int, default=None, help="Max items to keep per source")
    ap.add_argument("--out", type=str, default="ai_news.csv", help="Output CSV path")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay (seconds) between sources for politeness")
    ap.add_argument("--page-delay", type=float, default=0.0, help="Delay (seconds) between fetching full pages")
    args = ap.parse_args()

    since = norm_dt(args.since) if args.since else None
    until = norm_dt(args.until) if args.until else None

    session = make_session()
    items: List[Item] = []
    for src in SOURCES:
        sys.stderr.write(f"Collecting: {src.name}\n")
        got = collect_source(src, since, until, session, delay=args.delay)
        if args.limit_per_source is not None:
            # Sort by date desc before truncating
            got.sort(key=lambda x: x.date or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
            got = got[: args.limit_per_source]
        items.extend(got)
        sys.stderr.write(f"  -> {len(got)} items\n")

    # Final sort by date desc (undated go last)
    items.sort(key=lambda x: x.date or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    enrich_with_full_page(items, session, delay=args.page_delay)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "date", "title", "url", "summary", "page_html", "page_text"])
        for it in items:
            w.writerow([
                it.source,
                it.date.isoformat() if it.date else "",
                it.title,
                it.url,
                it.summary,
                it.page_html,
                it.page_text
            ])

    print(f"Wrote {len(items)} rows to {args.out}")

if __name__ == "__main__":
    main()

