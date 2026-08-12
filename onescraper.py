"""
Scrape a single arXiv paper for a given year/month.
 
Usage:
    python3 scraper.py <year> <month> [paper_number]
 
Example:
    python3 scraper.py 24 08          # scrapes paper 00001 (default)
    python3 scraper.py 24 08 00005    # scrapes paper 00005
"""
import argparse
import json
from pathlib import Path
 
import scraper_functions
 
parser = argparse.ArgumentParser(description="Scrape a single arXiv paper.")
parser.add_argument("year", help="Two-digit year (e.g. 24 for 2024)")
parser.add_argument("month", help="Two-digit month (e.g. 08 for August)")
parser.add_argument(
    "paper_number",
    nargs="?",
    default="00001",
    help="5-digit paper sequence number (default: 00001)",
)
args = parser.parse_args()
 
# No output_dir specified - saves in current directory
found = scraper_functions.process_paper(
    paper_id=args.paper_number,
    year=args.year,
    month=args.month,
    output_dir=None,  # This saves in current directory
)
 
if not found:
    print("Paper not found (404) or all retries failed check the year/month/number.")
else:
    full_id = f"{args.year}{args.month}.{args.paper_number}"
    json_path = Path.cwd() / f"{full_id}.json"
    html_path = Path.cwd() / f"{full_id}.html"
    
    print(f"\n✅ Saved files:")
    print(f"   📄 JSON: {json_path}")
    print(f"   📄 HTML: {html_path}")
    
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            
        print(f"\n📊 Paper Summary:") 
        print(f"   Title: {data['title'][:100]}...")
        print(f"   Categories: {', '.join(data['categories'])}")
        print(f"   📝 {data['metadata']['num_paragraphs']} paragraphs")
        print(f"   🖼️  {data['metadata']['num_figures']} figures")
        
        print("\n📝 First 2 paragraphs (sanity check):\n")
        for key, text in list(data['paragraphs'].items())[:2]:
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"   {key}: {preview}\n")