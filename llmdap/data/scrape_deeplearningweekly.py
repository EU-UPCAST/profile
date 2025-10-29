#!/usr/bin/env python3
"""
Scrape the DeepLearningWeekly newsletter archive into plain text files.

Usage (from repo root):
    python scripts/scrape_deeplearningweekly.py --output-dir data/deeplearningweekly
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

ARCHIVE_URL = "https://www.deeplearningweekly.com/archive"
ARCHIVE_API_BATCH_SIZE = 50
REQUEST_TIMEOUT = 15


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/118.0 Safari/537.36"
            )
        }
    )
    return session


def fetch(session: requests.Session, url: str) -> BeautifulSoup:
    logging.debug("Fetching %s", url)
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_links(archive_soup: BeautifulSoup) -> List[str]:
    links: List[str] = []
    for anchor in archive_soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href:
            continue
        if href.startswith("/"):
            href = f"https://www.deeplearningweekly.com{href}"
        if href.startswith("https://www.deeplearningweekly.com/p/"):
            links.append(href.split("#", 1)[0])
    # Preserve ordering while removing duplicates.
    seen = set()
    unique_links = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    logging.info("Found %d newsletter links on initial page", len(unique_links))
    return unique_links


def collect_archive_links(
    session: requests.Session,
    archive_url: str,
    delay: float,
) -> List[str]:
    parsed = urlparse(archive_url)
    api_url = urlunparse(
        parsed._replace(path="/api/v1/archive", params="", query="", fragment="")
    )

    links: List[str] = []
    seen = set()
    offset = 0

    while True:
        logging.info(
            "Fetching archive batch offset=%d limit=%d",
            offset,
            ARCHIVE_API_BATCH_SIZE,
        )
        response = session.get(
            api_url,
            params={"offset": offset, "limit": ARCHIVE_API_BATCH_SIZE},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 404 and parsed.netloc.startswith("www."):
            # Some custom domains omit the www prefix for API calls.
            alt_netloc = parsed.netloc[4:]
            api_url = urlunparse(
                parsed._replace(
                    netloc=alt_netloc,
                    path="/api/v1/archive",
                    params="",
                    query="",
                    fragment="",
                )
            )
            logging.debug("Retrying archive API without www: %s", api_url)
            offset = 0
            links.clear()
            seen.clear()
            continue

        response.raise_for_status()

        try:
            batch = response.json()
        except ValueError as exc:
            logging.error("Invalid JSON from archive API: %s", exc)
            break

        if not isinstance(batch, list):
            logging.error("Unexpected archive API response type: %s", type(batch))
            break

        if not batch:
            logging.info("Archive API returned no more posts at offset %d", offset)
            break

        new_links = 0
        for entry in batch:
            if not isinstance(entry, dict):
                continue
            url = entry.get("canonical_url")
            if not url and entry.get("slug"):
                url = urlunparse(
                    parsed._replace(
                        path=f"/p/{entry['slug']}",
                        params="",
                        query="",
                        fragment="",
                    )
                )
            if not url:
                continue
            url = url.split("#", 1)[0].strip()
            if url and url not in seen:
                links.append(url)
                seen.add(url)
                new_links += 1

        logging.info(
            "Archive batch offset=%d returned %d items (%d new)",
            offset,
            len(batch),
            new_links,
        )

        offset += len(batch)
        if new_links == 0:
            break
        time.sleep(delay)

    if not links:
        # Fallback to scraping the single archive page if API is unavailable.
        logging.warning(
            "Archive API yielded no links; falling back to page scraping mode."
        )
        soup = fetch(session, archive_url)
        links = extract_links(soup)

    return links


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = path.split("/")[-1]
    slug = slug or "deeplearningweekly"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug)
    slug = slug.strip("-")
    return slug or "deeplearningweekly"


def find_content_container(soup: BeautifulSoup) -> Tag:
    # Try a few common blog/newsletter containers.
    selectors = [
        ("article", {}),
        ("div", {"data-testid": "post-content"}),
        ("div", {"class": re.compile(r"(post|article|body|content)", re.I)}),
        ("main", {}),
    ]
    for name, attrs in selectors:
        node = soup.find(name, attrs=attrs)
        if node:
            return node
    body = soup.find("body")
    if not isinstance(body, Tag):
        raise ValueError("Unable to locate newsletter body in page")
    return body


def serialize_node(node: Tag, depth: int = 0) -> List[str]:
    lines: List[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                lines.append(text)
            continue

        if not isinstance(child, Tag):
            continue

        name = child.name.lower()

        if name in {"script", "style"}:
            continue

        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            heading = child.get_text(separator=" ", strip=True)
            if heading:
                lines.append(f"{'#' * level} {heading}")
                lines.append("")
            continue

        if name == "p":
            entries = extract_paragraph_entries(child)
            if entries:
                for entry in entries:
                    lines.append(f"### {entry.heading}")
                    if entry.href:
                        lines.append(f"[Link] {entry.href}")
                    for description in entry.description:
                        for desc_line in description.splitlines():
                            stripped = desc_line.strip()
                            if stripped:
                                lines.append(stripped)
                    lines.append("")
                continue

            entry_heading = extract_entry_heading(child)
            if entry_heading:
                heading_text, href = entry_heading
                lines.append(f"### {heading_text}")
                if href:
                    lines.append(f"[Link] {href}")
                lines.append("")
                continue

            category_heading = extract_category_heading(child)
            if category_heading:
                lines.append(f"## {category_heading}")
                lines.append("")
                continue
            text = child.get_text(separator="\n", strip=True)
            if text:
                lines.append(text)
                lines.append("")
            continue

        if name in {"ul", "ol"}:
            ordered = name == "ol"
            lines.extend(serialize_list(child, ordered=ordered, depth=depth))
            lines.append("")
            continue

        if name == "blockquote":
            quote_lines = [
                f"> {line}" if line else ">"
                for line in child.get_text(separator="\n", strip=True).splitlines()
            ]
            lines.extend(quote_lines)
            lines.append("")
            continue

        if name == "br":
            lines.append("")
            continue

        if name == "img":
            alt_text = child.get("alt", "").strip()
            src = child.get("src", "").strip()
            # Capture minimal image metadata so the reader can recover context.
            if alt_text:
                lines.append(f"[Image] {alt_text}")
            elif src:
                lines.append(f"[Image] {src}")
            lines.append("")
            continue

        # Fallback: recurse for composite containers (div, span, etc.).
        lines.extend(serialize_node(child, depth=depth))

    # Collapse consecutive blank lines while preserving paragraph spacing.
    collapsed: List[str] = []
    previous_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = is_blank
    return collapsed


def serialize_list(node: Tag, ordered: bool, depth: int = 0) -> List[str]:
    lines: List[str] = []
    index = 1
    for item in node.find_all("li", recursive=False):
        prefix = f"{index}." if ordered else "-"
        index += 1
        item_lines = serialize_node(item, depth=depth + 1)
        while item_lines and not item_lines[0].strip():
            item_lines.pop(0)
        if not item_lines:
            continue

        first_line = item_lines[0]
        indent = "  " * depth
        lines.append(f"{indent}{prefix} {first_line}")
        for continuation in item_lines[1:]:
            if continuation.strip():
                lines.append(f"{indent}    {continuation}")
            else:
                lines.append("")
    return lines


def extract_entry_heading(paragraph: Tag) -> Optional[Tuple[str, str]]:
    anchors = paragraph.find_all("a")
    if len(anchors) != 1:
        return None

    anchor = anchors[0]
    text = anchor.get_text(separator=" ", strip=True)
    if not text:
        return None

    paragraph_text = paragraph.get_text(separator=" ", strip=True)
    if paragraph_text != text:
        return None

    href = anchor.get("href", "").strip()
    return text, href


@dataclass
class ParagraphEntry:
    heading: str
    href: str
    description: List[str]


def extract_paragraph_entries(paragraph: Tag) -> Optional[List[ParagraphEntry]]:
    anchors = paragraph.find_all("a")
    if not anchors:
        return None

    # Require anchors to be wrapped in <strong> to avoid intro paragraphs with inline links.
    if any(anchor.find_parent("strong") is None for anchor in anchors):
        return None

    entries: List[ParagraphEntry] = []
    current: Optional[ParagraphEntry] = None

    for child in paragraph.children:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text and current is not None:
                current.description.append(text)
            continue

        if not isinstance(child, Tag):
            continue

        name = child.name.lower()

        if name == "strong":
            anchor = child.find("a")
            if anchor:
                heading = anchor.get_text(separator=" ", strip=True)
                if not heading:
                    current = None
                    continue
                href = anchor.get("href", "").strip()
                current = ParagraphEntry(heading=heading, href=href, description=[])
                entries.append(current)
            else:
                text = child.get_text(separator=" ", strip=True)
                if text and current is not None:
                    current.description.append(text)
            continue

        if name == "a":
            heading = child.get_text(separator=" ", strip=True)
            if heading:
                href = child.get("href", "").strip()
                current = ParagraphEntry(heading=heading, href=href, description=[])
                entries.append(current)
            continue

        if name == "br":
            continue

        text = child.get_text(separator="\n", strip=True)
        if text and current is not None:
            current.description.append(text)

    entries = [entry for entry in entries if entry.heading]
    return entries if entries else None


def extract_category_heading(paragraph: Tag) -> Optional[str]:
    if paragraph.find("a"):
        return None

    strong_tags = paragraph.find_all("strong")
    if not strong_tags:
        return None

    text = paragraph.get_text(separator=" ", strip=True)
    strong_text = " ".join(
        tag.get_text(separator=" ", strip=True) for tag in strong_tags
    ).strip()

    if text and text == strong_text:
        return text
    return None


@dataclass
class Newsletter:
    url: str
    title: str
    lines: List[str]


def extract_newsletter(session: requests.Session, url: str) -> Newsletter:
    soup = fetch(session, url)
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else slug_from_url(url)
    container = find_content_container(soup)
    lines = serialize_node(container)
    if lines and lines[0].startswith("#"):
        # Avoid duplicated title if heading already present.
        return Newsletter(url=url, title=title, lines=lines)

    heading_line = f"# {title}"
    content = [heading_line, ""]
    content.extend(lines)
    return Newsletter(url=url, title=title, lines=content)


def write_newsletter(newsletter: Newsletter, output_dir: Path, overwrite: bool) -> Path:
    slug = slug_from_url(newsletter.url)
    output_path = output_dir / f"{slug}.txt"
    if output_path.exists() and not overwrite:
        logging.info("Skipping %s (already exists)", output_path.name)
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(newsletter.lines).rstrip() + "\n"
    output_path.write_text(content, encoding="utf-8")
    logging.info("Saved %s", output_path)
    return output_path


def scrape(
    archive_url: str,
    output_dir: Path,
    limit: Optional[int],
    delay: float,
    overwrite: bool,
) -> None:
    session = build_session()
    links = collect_archive_links(session, archive_url, delay=delay)
    if limit is not None:
        links = links[:limit]

    for idx, url in enumerate(links, start=1):
        logging.info("Processing %d/%d: %s", idx, len(links), url)
        try:
            newsletter = extract_newsletter(session, url)
            write_newsletter(newsletter, output_dir, overwrite=overwrite)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to process %s: %s", url, exc)
        time.sleep(delay)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DeepLearningWeekly newsletter archive as plain text.",
    )
    parser.add_argument(
        "--archive-url",
        default=ARCHIVE_URL,
        help=f"Archive page to scrape (default: {ARCHIVE_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/deeplearningweekly"),
        help="Directory where newsletter text files will be stored.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of newsletters to download (useful for testing).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between requests to avoid overloading the site.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    try:
        scrape(
            archive_url=args.archive_url,
            output_dir=args.output_dir,
            limit=args.limit,
            delay=max(args.delay, 0.0),
            overwrite=args.overwrite,
        )
    except requests.RequestException as exc:
        logging.error("Network error: %s", exc)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Unexpected error: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
