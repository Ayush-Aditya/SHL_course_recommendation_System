"""
SHL Product Catalog Scraper
Scrapes all Individual Test Solutions from the SHL product catalog.
Uses requests + BeautifulSoup for listing pages and detail pages.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os

BASE_URL = "https://www.shl.com/products/product-catalog/"
DETAIL_BASE = "https://www.shl.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

# Test type code to label mapping
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def scrape_catalog_page(start: int, catalog_type: int = 1) -> list[dict]:
    """Scrape one page of the catalog listing.
    type=1 is Individual Test Solutions, type=2 is Pre-packaged Job Solutions.
    """
    url = f"{BASE_URL}?start={start}&type={catalog_type}"
    print(f"  Fetching listing page: {url}")
    
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    items = []
    
    # Find all table rows in the catalog - each product is in a table row
    # The catalog uses a custom table structure
    # Look for product links in the Individual Test Solutions section
    
    # Find all links that point to product-catalog/view/
    product_links = soup.find_all("a", href=re.compile(r"/products/product-catalog/view/"))
    
    for link in product_links:
        name = link.get_text(strip=True)
        href = link.get("href", "")
        if not href or not name:
            continue
            
        # Build full URL
        if href.startswith("/"):
            full_url = DETAIL_BASE + href
        else:
            full_url = href
            
        # Get the parent row to extract other columns
        row = link.find_parent("tr")
        if not row:
            # Try finding parent with class pattern
            row = link.find_parent(["div", "tr", "li"])
            
        remote_testing = False
        adaptive_irt = False
        test_type_codes = []
        
        if row:
            cells = row.find_all("td")
            if len(cells) >= 4:
                # Column 0: Name (link)
                # Column 1: Remote Testing (green dot = yes)
                # Column 2: Adaptive/IRT (green dot = yes)
                # Column 3: Test Type (colored badges)
                
                # Remote testing - check for green dot (span with specific class or img)
                remote_cell = cells[1]
                if remote_cell.find("span") or remote_cell.get_text(strip=True):
                    remote_testing = True
                    
                # Adaptive/IRT
                adaptive_cell = cells[2]
                if adaptive_cell.find("span") or adaptive_cell.get_text(strip=True):
                    adaptive_irt = True
                    
                # Test types - look for badge elements
                type_cell = cells[3]
                # Test types are shown as small badge elements with single letters
                badges = type_cell.find_all("span")
                for badge in badges:
                    code = badge.get_text(strip=True).upper()
                    if code in TEST_TYPE_MAP:
                        test_type_codes.append(code)
                        
                # If no spans found, try the raw text
                if not test_type_codes:
                    raw = type_cell.get_text(strip=True)
                    for char in raw:
                        if char.upper() in TEST_TYPE_MAP and char.upper() not in test_type_codes:
                            test_type_codes.append(char.upper())
        
        items.append({
            "name": name,
            "url": full_url,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "test_type": test_type_codes,
        })
    
    return items


def scrape_detail_page(url: str) -> dict:
    """Scrape a product detail page for additional metadata."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        detail = {}
        
        # Get og:description for the description
        og_desc = soup.find("meta", property="og:description")
        if og_desc:
            content = og_desc.get("content", "")
            # The format is usually "Name: Description..."
            if ": " in content:
                detail["description"] = content.split(": ", 1)[1].strip()
            else:
                detail["description"] = content.strip()
        
        # Try to get title for fallback name
        og_title = soup.find("meta", property="og:title")
        if og_title:
            detail["og_title"] = og_title.get("content", "").replace(" | SHL", "").strip()
        
        # Look for structured data in the page body
        # The detail pages have sections like "Assessment Length", "Languages", etc.
        # These may be in specific containers
        body_text = soup.get_text()
        
        # Try to find duration/assessment length
        duration_match = re.search(r'(?:Assessment Length|Duration)[:\s]*(\d+\s*minutes?)', body_text, re.IGNORECASE)
        if duration_match:
            detail["duration"] = duration_match.group(1).strip()
            
        # Try to find languages
        lang_section = None
        for heading in soup.find_all(["h3", "h4", "strong", "dt"]):
            if "language" in heading.get_text(strip=True).lower():
                lang_section = heading.find_next(["dd", "p", "div", "ul"])
                break
        
        if lang_section:
            detail["languages"] = lang_section.get_text(strip=True)
            
        return detail
        
    except Exception as e:
        print(f"    Warning: Failed to scrape detail page {url}: {e}")
        return {}


def scrape_all_individual_tests() -> list[dict]:
    """Scrape all Individual Test Solutions (type=1), all 32 pages."""
    all_items = []
    seen_urls = set()
    
    # There are 32 pages, 12 items per page
    for page in range(32):
        start = page * 12
        print(f"\n--- Page {page + 1}/32 (start={start}) ---")
        
        items = scrape_catalog_page(start, catalog_type=1)
        
        new_items = []
        for item in items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                new_items.append(item)
        
        print(f"  Found {len(new_items)} new items (total unique: {len(seen_urls)})")
        all_items.extend(new_items)
        
        time.sleep(0.5)  # Be polite
    
    print(f"\n=== Total unique items from listings: {len(all_items)} ===")
    
    # Now fetch detail pages for additional metadata
    print("\n--- Fetching detail pages for descriptions ---")
    for i, item in enumerate(all_items):
        print(f"  [{i+1}/{len(all_items)}] {item['name']}")
        detail = scrape_detail_page(item["url"])
        
        if "description" in detail:
            item["description"] = detail["description"]
        else:
            item["description"] = ""
            
        if "duration" in detail:
            item["duration"] = detail["duration"]
            
        if "languages" in detail:
            item["languages_raw"] = detail["languages"]
            
        # Add test type labels
        item["test_type_labels"] = [TEST_TYPE_MAP.get(c, c) for c in item.get("test_type", [])]
        
        time.sleep(0.3)  # Be polite
    
    return all_items


def main():
    print("SHL Product Catalog Scraper")
    print("=" * 60)
    
    items = scrape_all_individual_tests()
    
    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Saved {len(items)} items to {output_path} ===")
    
    # Print summary
    type_counts = {}
    for item in items:
        for t in item.get("test_type", []):
            type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\nTest type distribution:")
    for code, count in sorted(type_counts.items()):
        print(f"  {code} ({TEST_TYPE_MAP.get(code, '?')}): {count}")


if __name__ == "__main__":
    main()
