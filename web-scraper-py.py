"""
utils/web_scraper.py - Web scraping and sitemap utilities
"""
import xml.etree.ElementTree as ET
from typing import List

import requests
from bs4 import BeautifulSoup


def parse_sitemap(url: str) -> List[str]:
    """Parse a sitemap and return all URLs"""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    
    root = ET.fromstring(r.text)
    ns = {"sm": root.tag.split("}")[0].strip("{")}
    
    # Check for URLs in the sitemap
    locs = root.findall("sm:url/sm:loc", ns) or root.findall("sm:sitemap/sm:loc", ns)
    urls = [e.text.strip() for e in locs if e.text]
    
    # If this is a sitemap index, recursively parse sub-sitemaps
    if root.find("sm:sitemap", ns) is not None:
        nested = []
        for sub in urls:
            try:
                nested.extend(parse_sitemap(sub))
            except Exception:
                pass  # Skip failed sub-sitemaps
        return nested
    
    return urls


def download_page(url: str) -> str:
    """Download and extract text from a web page"""
    r = requests.get(url, timeout=15, headers={"User-Agent": "RAG-Chatbot-Crawler"})
    r.raise_for_status()
    
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Remove script and style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    
    # Extract text
    return soup.get_text("\n", strip=True)