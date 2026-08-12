from __future__ import annotations
from urllib.parse import urljoin
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, List, Tuple
 
import requests
from bs4 import BeautifulSoup, NavigableString
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
ARXIV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/"
 
REQUEST_DELAY_SECONDS = 1.0
 
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------
 
def fetch_soup(url: str) -> Optional[BeautifulSoup]:
    """
    Download *url* and return a BeautifulSoup parse tree.
 
    Retries up to MAX_RETRIES times on transient HTTP errors (5xx) or
    connection problems.  Returns None if every attempt fails.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
 
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            if status == 404:
                log.debug("404 – paper does not exist: %s", url)
                return None          # not a transient error; stop retrying
            log.warning("HTTP %s on attempt %d/%d for %s", status, attempt, MAX_RETRIES, url)
 
        except requests.exceptions.RequestException as exc:
            log.warning("Request error on attempt %d/%d for %s: %s", attempt, MAX_RETRIES, url, exc)
 
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS)
 
    log.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None
 
 
# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
 
def parse_title_abstract(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Extract the paper title and abstract text from a parsed arXiv HTML page.
 
    Returns a ``(title, abstract)`` tuple.  Either value is an empty string
    when the expected element cannot be found.
    """
    title_tag = soup.find("h1", class_="ltx_title_document")
    title = ""
    if title_tag:
        title = ' '.join(extract_text_with_math(title_tag).split())
 
    abstract_div = soup.find("div", class_="ltx_abstract")
    abstract = ""
    if abstract_div:
        abstract_p = abstract_div.find("p", class_="ltx_p")
        if abstract_p:
            abstract = ' '.join(extract_text_with_math(abstract_p).split())
 
    return title, abstract
 
 
def _resolve_image_url(raw_src: str, paper_url: str) -> str:
    """
     Return an absolute URL for *raw_src*, resolving relative paths against *paper_url*.
     Works correctly for ar5iv's absolute paths (starting with '/html/...').
    """
    return urljoin(paper_url, raw_src)
 
 
# --- Revised caption extraction ---
def _outer_caption(fig_tag: BeautifulSoup) -> str:
    for tag in fig_tag.children:
        if tag.name == "figcaption" and "ltx_caption" in tag.get("class", []):
            raw = extract_text_with_math(tag)
            return ' '.join(raw.split())
    return ""
 
# --- Math‑aware text extraction ---
def extract_text_with_math(element) -> str:
    if isinstance(element, NavigableString):
        return str(element)
    if element.name == 'math':
        alt = element.get('alttext', '')
        return f'${alt}$' if alt else element.get_text()
    return ''.join(extract_text_with_math(child) for child in element.children)
 
 
def parse_figures(soup: BeautifulSoup, paper_url: str) -> list[dict]:
    rows: list[dict] = []
    sequential_idx = 0
 
    top_level_figures = [
        fig for fig in soup.find_all("figure", class_="ltx_figure")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]
 
    for fig in top_level_figures:
        sequential_idx += 1
        outer_caption = _outer_caption(fig)
 
        num_match = re.search(r"\bFigure\s+(\d+)", outer_caption, re.IGNORECASE)
        fig_id = int(num_match.group(1)) if num_match else sequential_idx
 
        panels = fig.find_all("figure", class_="ltx_figure_panel")
 
        if panels:
            # First, try to extract individual sub‑captions from each panel
            panel_captions = []
            for panel in panels:
                sub = _outer_caption(panel)
                panel_captions.append(sub)
 
            # If all panel captions are empty, fall back to splitting the outer caption
            if all(c == "" for c in panel_captions):
                # Parse the outer caption for "(a) ... (b) ..."
                sub_parts = split_subcaptions(outer_caption)
                # If we found sub‑labels, use them; otherwise, fall back to simple letters
                if sub_parts:
                    # Ensure we have at least as many parts as panels; if not, pad with empty texts
                    while len(sub_parts) < len(panels):
                        sub_parts.append(("", ""))
                    # Use the sub‑captions from the split
                    panel_captions = [text for _, text in sub_parts[:len(panels)]]
                    # Also set sub_ids from the letters
                    panel_sub_ids = [letter for letter, _ in sub_parts[:len(panels)]]
                    # If some letters are missing, use order-based letters
                    for i, (label, text) in enumerate(sub_parts):
                        if not label:
                            panel_sub_ids[i] = chr(ord('a') + i)
                else:
                    # No labels found; just use order-based letters and no sub‑captions
                    panel_sub_ids = [chr(ord('a') + i) for i in range(len(panels))]
                    panel_captions = ["" for _ in panels]
            else:
                # We have individual captions; extract sub_ids from each
                panel_sub_ids = []
                for sub in panel_captions:
                    sub_match = re.search(r"\(([a-z])\)", sub, re.IGNORECASE)
                    panel_sub_ids.append(sub_match.group(1).lower() if sub_match else None)
 
            # Now iterate over panels and build rows
            for idx, panel in enumerate(panels):
                img_tag = panel.find("img")
                img_src = (
                    _resolve_image_url(img_tag["src"], paper_url)
                    if img_tag and img_tag.get("src")
                    else None
                )
                sub_caption = panel_captions[idx] if idx < len(panel_captions) else ""
                sub_id = panel_sub_ids[idx] if idx < len(panel_sub_ids) else None
 
                rows.append({
                    "figure_id": fig_id,
                    "sub_id": sub_id,
                    "source": img_src,
                    "caption": outer_caption,
                    "sub_caption": sub_caption,
                })
        else:
            # Simple figure
            img_tag = fig.find("img")
            img_src = (
                _resolve_image_url(img_tag["src"], paper_url)
                if img_tag and img_tag.get("src")
                else None
            )
            rows.append({
                "figure_id": fig_id,
                "sub_id": None,
                "source": img_src,
                "caption": outer_caption,
                "sub_caption": None,
            })
 
    return rows


def parse_figure_references(soup: BeautifulSoup) -> dict[str, list[str]]:
    figure_numbers: list[str] = []
    references: dict[str, list[str]] = {}
    top_level_figures = [
        fig for fig in soup.find_all("figure", class_="ltx_figure")
        if "ltx_figure_panel" not in fig.get("class", [])
    ]
    for seq_idx, fig in enumerate(top_level_figures, start=1):
        caption = _outer_caption(fig)
        match = re.search(r"\bFigure\s+(\d+)", caption, re.IGNORECASE)
        fig_num = match.group(1) if match else str(seq_idx)
        figure_numbers.append(fig_num)
        references[f"Figure {fig_num}"] = []
    all_paragraphs = soup.find_all("p", class_="ltx_p")
    for fig_num in figure_numbers:
        key = f"Figure {fig_num}"
        pattern = re.compile(
            rf"\bfig(?:ure)?\.?\s*{re.escape(fig_num)}(?:[a-z]|\([a-z]\)|-[a-z])?\b",
            re.IGNORECASE,
        )
        for para in all_paragraphs:
            if para.find_parent("figure") is not None:
                continue
            text = extract_text_with_math(para)
            text = ' '.join(text.split())  # normalise whitespace
            if pattern.search(text) and text not in references[key]:
                references[key].append(text)
    return references


import xml.etree.ElementTree as ET
 
 
def parse_categories(soup: BeautifulSoup, paper_id) -> list[str]:
    """
    Extract arXiv subject categories. Extracts the paper ID from the HTML header
    and cross-references the official arXiv API since categories are missing
    from the raw HTML text body.
    """
    categories = []
 
    try:
        log.info("Fetching categories via arXiv API for ID: %s", paper_id)
        api_url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
        api_resp = requests.get(api_url, timeout=10)
 
        if api_resp.status_code == 200:
            root = ET.fromstring(api_resp.content)
            # Parse all <category term="..."/> tags from the Atom XML feed
            for category_tag in root.findall(".//{http://www.w3.org/2005/Atom}category"):
                term = category_tag.get("term")
                # Keeps valid primary/secondary categories (e.g., cs.HC, stat.ML)
                if term and "." in term and term not in categories:
                    categories.append(term)
    except Exception as e:
        log.warning("arXiv API metadata fallback query failed: %s", e)
 
    return categories


def extract_paragraphs(soup: BeautifulSoup) -> list[str]:
    """
    Extract the text of every ``<p>`` (arXiv/ar5iv marks these with class
    ``ltx_p``) in the paper body, in document order, with math converted to
    LaTeX-ish text via extract_text_with_math and whitespace normalised.
 
    Empty paragraphs are skipped.
    """
    paragraphs: list[str] = []
    for p in soup.find_all("p", class_="ltx_p"):
        text = extract_text_with_math(p)
        text = ' '.join(text.split())
        if text:
            paragraphs.append(text)
    return paragraphs


def split_subcaptions(caption: str) -> List[Tuple[str, str]]:
    """
    Split a compound figure caption like
    "Figure 1: (a) First panel (b) Second panel."
    into a list of (label, text) for each sub‑panel.
 
    Returns a list of (letter, subcaption_text) pairs.
    """
    # Pattern: (a) ... followed by another (letter) or end of string
    pattern = re.compile(r'\(([a-z])\)\s*([^)]*?)(?=\s*\([a-z]\)|$)')
    matches = pattern.findall(caption)
    # Clean up: remove leading/trailing whitespace from each text
    return [(letter, text.strip()) for letter, text in matches]


# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------
 
def process_paper(
    paper_id: str,
    year: str,
    month: str,
    output_dir: Optional[Path] = None,
) -> bool:
    """
    Scrape a single arXiv paper and save all data.
    
    Saves:
    1. JSON file with all data (paper_id + enumerator for paragraphs)
    2. HTML file with the full paper
 
    Parameters
    ----------
    paper_id:
        The five-digit arXiv sequence number, e.g. ``"12325"``.
        Combined with *year* and *month* it forms the full ID ``YYMM.NNNNN``
        (e.g. ``"2510.12325"``).
    year:
        Two-digit year string, e.g. ``"25"``.
    month:
        Two-digit month string, e.g. ``"10"``.
    output_dir:
        Optional directory to save files. If None, saves in current directory.
 
    Returns
    -------
    bool
        True if the paper was found and processed, False otherwise.
    """
    # arXiv IDs follow the format YYMM.NNNNN  (e.g. 2510.12325)
    full_id = f"{year}{month}.{paper_id}"
    paper_url = ARXIV_HTML_BASE + full_id
    log.info("Processing %s …", paper_url)
 
    soup = fetch_soup(paper_url)
 
    if soup is None:
        return False
 
    # Determine save location
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        save_dir = output_dir
    else:
        save_dir = Path.cwd()
    
    # ------------------------------------------------------------------ parse
    title, abstract = parse_title_abstract(soup)
    figures = parse_figures(soup, paper_url)
    fig_references = parse_figure_references(soup)
    categories = parse_categories(soup, full_id)
    paragraphs = extract_paragraphs(soup)
    
    # ------------------------------------------------------------------ save HTML
    html_path = save_dir / f"{full_id}.html"
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(str(soup))
    log.info(f"  ✓ HTML saved to: {html_path}")
    
    # ------------------------------------------------------------------ save JSON with all data
    json_path = save_dir / f"{full_id}.json"
    
    # Build the complete data structure
    paper_data = {
        "paper_id": full_id,
        "title": title,
        "abstract": abstract,
        "categories": categories,
        # Paragraphs with paper_id + enumerator (starts from 1)
        "paragraphs": {
            f"{full_id}_{i}": text 
            for i, text in enumerate(paragraphs, start=1)
        },
        "figures": figures,
        "figure_references": fig_references,
        "metadata": {
            "url": paper_url,
            "num_paragraphs": len(paragraphs),
            "num_figures": len(figures),
            "num_categories": len(categories),
        }
    }
    
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(paper_data, fh, indent=2, ensure_ascii=False)
    
    log.info(f"  ✓ JSON saved to: {json_path}")
    log.info(f"    {len(paragraphs)} paragraphs, {len(figures)} figures, {len(categories)} categories")
    
    return True
 
 
# ---------------------------------------------------------------------------
# Batch scraping
# ---------------------------------------------------------------------------
 
def scrape_month(
    year: str,
    month: str,
    output_dir: Optional[Path] = None,
    max_papers: Optional[int] = None,
    start_id: int = 1,
) -> None:
    """
    Iterate over arXiv paper IDs for a given *year* / *month* and scrape each
    one until a 404 is returned (indicating no more papers exist for that
    month/year combination).
 
    Parameters
    ----------
    year:
        Two-digit year, e.g. ``"25"``.
    month:
        Two-digit month, e.g. ``"10"``.
    output_dir:
        Optional directory for output files. If None, saves in current directory.
    max_papers:
        Stop after processing this many papers (useful for testing). Pass
        None to scrape until arXiv returns a 404.
    start_id:
        The numeric ID to begin from (default 1). Useful for resuming an
        interrupted run.
    """
    log.info("=== Scraping %s/%s (starting at %s.%05d) ===", year, month, year + month, start_id)
    processed = 0
 
    for numeric_id in range(start_id, 100_000):
        paper_id = f"{numeric_id:05d}"
        found = process_paper(paper_id, year, month, output_dir)
 
        if not found:
            log.info("No paper found for id %s – assuming end of %s/%s.", paper_id, year, month)
            break
 
        processed += 1
        if max_papers is not None and processed >= max_papers:
            log.info("Reached max_papers limit (%d). Stopping.", max_papers)
            break
 
        time.sleep(REQUEST_DELAY_SECONDS)
 
    log.info("Done. Processed %d paper(s) for %s/%s.", processed, year, month)


def scrape_range(
    years: list[str],
    months: list[str],
    output_dir: Optional[Path] = None,
    max_papers_per_month: Optional[int] = None,
) -> None:
    """
    Scrape every (year, month) combination in *years* × *months*.
 
    Parameters
    ----------
    years:
        List of two-digit year strings, e.g. ``["24", "25"]``.
    months:
        List of two-digit month strings, e.g. ``["01", "02", …, "12"]``.
    output_dir:
        Optional directory for output files. If None, saves in current directory.
    max_papers_per_month:
        Cap on papers scraped per month (useful for testing).
    """
    for year in years:
        for month in months:
            scrape_month(year, month, output_dir, max_papers=max_papers_per_month)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Scrape arXiv papers for a given month.")
    parser.add_argument("year", help="Two-digit year (e.g. 24 for 2024)")
    parser.add_argument("month", help="Two-digit month (e.g. 10 for October)")
    args = parser.parse_args()
 
    # No output_dir specified - saves in current directory
    scrape_month(args.year, args.month)