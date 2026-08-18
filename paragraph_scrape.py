"""
Extract paragraphs from arXiv papers for a given year/month using the arXiv API.
 
Usage:
    python3 paragraph_extractor.py <year> <month>
 
Example:
    python3 paragraph_extractor.py 24 08    # extracts paragraphs from August 2024 papers
"""
import argparse
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import re
from datetime import datetime, timedelta

def get_paper_ids_from_api(year, month, max_results=1000):
    """Get list of paper IDs from arXiv API for a given month"""
    
    # Convert two-digit year to four-digit
    full_year = 2000 + int(year)
    
    # Calculate date range for the month
    start_date = datetime(full_year, int(month), 1)
    if int(month) == 12:
        end_date = datetime(full_year + 1, 1, 1)
    else:
        end_date = datetime(full_year, int(month) + 1, 1)
    
    # Format dates for API query
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    # Build API query
    query = f'submittedDate:[{start_str} TO {end_str}]'
    api_url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}&sortBy=submittedDate&sortOrder=ascending"
    
    print(f"   Querying arXiv API: {api_url}")
    
    try:
        response = requests.get(api_url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            print(f"   ⚠️ API returned status: {response.status_code}")
            return []
        
        # Parse the XML response
        soup = BeautifulSoup(response.content, 'xml')
        
        # Find all entry tags
        entries = soup.find_all('entry')
        paper_ids = []
        
        for entry in entries:
            # Find the id tag
            id_tag = entry.find('id')
            if id_tag:
                # Extract paper ID from URL like http://arxiv.org/abs/2408.12345
                paper_id = id_tag.text.split('/')[-1]
                paper_ids.append(paper_id)
        
        print(f"   Found {len(paper_ids)} papers via API")
        if paper_ids:
            print(f"   First 5 paper IDs: {paper_ids[:5]}")
        
        return paper_ids
        
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        return []

def extract_paragraphs_from_paper(paper_id, html_content):
    """Extract all paragraphs from a paper's HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all <p> tags
    paragraphs = soup.find_all('p')
    
    # Extract text from each paragraph
    extracted_paragraphs = []
    for idx, p in enumerate(paragraphs, start=1):
        text = p.get_text().strip()
        if text:  # Only add non-empty paragraphs
            extracted_paragraphs.append({
                'paper_id': paper_id,
                'paragraph_id': idx,
                'content': text
            })
    
    return extracted_paragraphs

def scrape_papers_for_paragraphs(year, month, output_dir=None, max_papers=None, start_id=1):
    """Scrape arXiv papers and extract paragraphs using the API"""
    
    # Set up output directory
    if output_dir is None:
        folder_name = f"{month}_{year}_paragraphs"
        output_dir = Path.cwd() / folder_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📚 Extracting paragraphs from papers for {year}/{month}...")
    print(f"   Output directory: {output_dir}")
    
    # Get list of papers using the API
    paper_ids = get_paper_ids_from_api(year, month)
    
    if not paper_ids:
        print("   ❌ No papers found for this month via API!")
        print("   📝 Try:")
        print("      - Checking if the month has papers at: https://arxiv.org")
        print("      - Using a different month (e.g., 24 07 for July 2024)")
        return []
    
    # Filter by start_id
    if start_id > 1:
        paper_ids = paper_ids[start_id-1:]
    
    # Limit papers if max_papers is specified
    if max_papers:
        paper_ids = paper_ids[:max_papers]
    
    print(f"   Processing {len(paper_ids)} papers...")
    print()
    
    all_paragraphs = []
    failed_papers = []
    no_paragraphs_papers = []
    
    for idx, paper_id in enumerate(paper_ids, 1):
        try:
            print(f"  [{idx}/{len(paper_ids)}] 📄 Processing paper: {paper_id}")
            
            # First try HTML version
            html_url = f"https://arxiv.org/html/{paper_id}"
            print(f"      Fetching HTML: {html_url}")
            
            html_response = requests.get(html_url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if html_response.status_code == 200:
                # Extract paragraphs from HTML
                paragraphs = extract_paragraphs_from_paper(paper_id, html_response.text)
            else:
                # Try PDF version if HTML isn't available
                print(f"      HTML not available, trying PDF...")
                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                print(f"      💡 PDF available at: {pdf_url}")
                print(f"      ⚠️ Paragraph extraction from PDF not supported in this version")
                paragraphs = []
            
            if paragraphs:
                # Save paragraphs for this paper
                for para in paragraphs:
                    filename = f"{paper_id}_{para['paragraph_id']:04d}.json"
                    filepath = output_dir / filename
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(para, f, indent=2, ensure_ascii=False)
                    
                    all_paragraphs.append(para)
                
                print(f"      ✅ Found {len(paragraphs)} paragraphs")
            else:
                print(f"      ⚠️ No paragraphs found (HTML not available)")
                no_paragraphs_papers.append(paper_id)
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            failed_papers.append(paper_id)
        
        # Be polite - delay between requests
        time.sleep(2)
    
    # Save summary
    summary = {
        'year': year,
        'month': month,
        'total_papers_found': len(paper_ids),
        'total_papers_processed': len(paper_ids) - len(failed_papers),
        'papers_with_paragraphs': len(paper_ids) - len(failed_papers) - len(no_paragraphs_papers),
        'failed_papers': failed_papers,
        'papers_with_no_paragraphs': no_paragraphs_papers,
        'total_paragraphs': len(all_paragraphs)
    }
    
    summary_file = output_dir / f"extraction_summary_{year}{month}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Extraction complete!")
    print(f"   📝 Total paragraphs extracted: {len(all_paragraphs)}")
    print(f"   📄 Total papers found: {len(paper_ids)}")
    print(f"   ✅ Papers processed: {len(paper_ids) - len(failed_papers)}")
    print(f"   ❌ Failed papers: {len(failed_papers)}")
    print(f"   📄 Papers with no paragraphs: {len(no_paragraphs_papers)}")
    print(f"   💾 Files saved in: {output_dir}")
    
    return all_paragraphs

def main():
    parser = argparse.ArgumentParser(description="Extract paragraphs from arXiv papers for a given month.")
    parser.add_argument("year", help="Two-digit year (e.g. 24 for 2024)")
    parser.add_argument("month", help="Two-digit month (e.g. 08 for August)")
    parser.add_argument(
        "--max_papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (for testing)",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=1,
        help="Starting paper index (default: 1, useful for resuming)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for paragraphs (default: creates month_year_paragraphs folder)",
    )
    
    args = parser.parse_args()
    
    print(f"📚 Extracting paragraphs from papers for {args.year}/{args.month}...")
    print(f"   Starting at paper index {args.start_id}")
    if args.max_papers:
        print(f"   Max papers: {args.max_papers}")
    print()
    
    # Extract paragraphs from papers
    scrape_papers_for_paragraphs(
        year=args.year,
        month=args.month,
        output_dir=args.output_dir,
        max_papers=args.max_papers,
        start_id=args.start_id,
    )

if __name__ == "__main__":
    main()