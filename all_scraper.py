"""
Scrape ALL arXiv papers for a given year/month.
 
Usage:
    python3 scraper.py <year> <month>
 
Example:
    python3 scraper.py 24 08    # scrapes ALL papers from August 2024
"""
import argparse
from pathlib import Path

import scraper_functions

parser = argparse.ArgumentParser(description="Scrape ALL arXiv papers for a given month.")
parser.add_argument("year", help="Two-digit year (e.g. 24 for 2024)")
parser.add_argument("month", help="Two-digit month (e.g. 08 for August)")
parser.add_argument(
    "--max_papers",
    type=int,
    default=None,
    help="Maximum number of papers to scrape (for testing)",
)
parser.add_argument(
    "--start_id",
    type=int,
    default=1,
    help="Starting paper ID (default: 1, useful for resuming)",
)
args = parser.parse_args()

print(f"📚 Scraping ALL papers from {args.year}/{args.month}...")
print(f"   Starting at paper {args.start_id:05d}")
if args.max_papers:
    print(f"   Max papers: {args.max_papers}")
print()

# Scrape ALL papers from this month
scraper_functions.scrape_month(
    year=args.year,
    month=args.month,
    output_dir=None,  # Saves in current directory
    max_papers=args.max_papers,
    start_id=args.start_id,
)

print(f"\n✅ Done! Check your current directory for {args.year}{args.month}.*.json and .html files")