#!/usr/bin/env python3
"""
Parallel Patent Field Extractor v1.4
- All functions in one file
- Fixed imports
- Complete error handling
"""

import os
import json
import re
import argparse
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from psutil import virtual_memory
from typing import List, Dict, Optional

# --------------------------
# Core Extraction Functions
# --------------------------

def parse_classification_codes(text: str) -> List[str]:
    """Clean and split classification codes"""
    return [line.strip() for line in text.split('\n') if line.strip()]

def normalize_text(text: Optional[str]) -> Optional[str]:
    """Clean whitespace and normalize newlines"""
    if not text:
        return None
    return ' '.join(text.split())

def normalize_multi_text(text: str) -> List[str]:
    """Normalize multi-line fields (APPLICANT/INVENTOR)"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return list(dict.fromkeys(lines))  # Remove duplicates while preserving order

def parse_claims(text: str) -> List[str]:
    """Extract claims with multiple handling strategies"""
    if not text:
        return []
    
    # Strategy 1: Structured <claim> tags
    if '<claim>' in text.lower():
        soup = BeautifulSoup(text, 'html.parser')
        return [c.get_text(' ', strip=True) for c in soup.find_all('claim')]
    
    # Strategy 2: Numbered claims (1., 2., etc.)
    numbered_claims = re.split(r'\n\s*\d+\.\s*', text)
    if len(numbered_claims) > 1:
        return [c.strip() for c in numbered_claims if c.strip()]
    
    # Strategy 3: Fallback - split by newlines
    return [c.strip() for c in text.split('\n') if c.strip()]

def extract_fields(file_path: str) -> Dict[str, Optional[str|List[str]]]:
    """Main extraction function"""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    text_field = soup.find('text')
    if not text_field:
        raise ValueError("No <TEXT> tag found")

    return {
        'docno': get_text(soup, 'docno', required=True),
        'date': get_text(text_field, 'date'),
        'ipcr': parse_classification_codes(get_text(text_field, 'ipcr-classifications', default="")),
        'cpc': parse_classification_codes(get_text(text_field, 'cpc-classifications', default="")),
        'title': normalize_text(get_text(text_field, 'title', required=True)),
        'abstract': normalize_text(get_text(text_field, 'abstract')),
        'applicant': normalize_multi_text(get_text(text_field, 'applicant', default="")),
        'inventor': normalize_multi_text(get_text(text_field, 'inventor', default="")),
        'description': normalize_text(get_text(text_field, 'description')),
        'claims': parse_claims(get_text(text_field, 'claims', default=""))
    }

def get_text(soup: BeautifulSoup, tag: str, default: Optional[str] = None, required: bool = False) -> str:
    """Safe tag extraction with validation"""
    element = soup.find(tag)
    if not element:
        if required:
            raise ValueError(f"Required tag <{tag}> not found")
        return default
    return element.get_text(' ', strip=True)

# --------------------------
# Parallel Processing Setup
# --------------------------

def get_safe_worker_count(requested_workers: int) -> int:
    """Adjust workers based on available RAM"""
    mem = virtual_memory()
    safe_workers = min(
        requested_workers,
        cpu_count(),
        max(1, int(mem.available / (512 * 1024 * 1024)))  # Added missing )
    )  # This closes the min()
    print(f"Using {safe_workers} workers (RAM available: {mem.available//(1024*1024)}MB)")
    return safe_workers

def process_file(args) -> None:
    """Worker function for parallel processing"""
    file_path, output_dir = args
    
    try:
        if virtual_memory().available < (256 * 1024 * 1024):
            raise MemoryError("Low system memory")
            
        patent = extract_fields(file_path)
        output_path = os.path.join(output_dir, f"{patent['docno']}.json.tmp")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(patent, f, indent=2, ensure_ascii=False)
        os.rename(output_path, output_path[:-4])
        
    except Exception as e:
        return (os.path.basename(file_path), str(e))
    return None

def batch_extract(input_dir: str, output_dir: str, workers: int) -> None:
    """Main processing function"""
    os.makedirs(output_dir, exist_ok=True)
    error_log = os.path.join(output_dir, "extraction_errors.log")
    
    files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith('.txt') and 
           not os.path.exists(os.path.join(output_dir, f"{os.path.splitext(f)[0]}.json"))
    ]
    
    print(f"Processing {len(files)} files")
    
    errors = []
    batch_size = min(1000, len(files))
    
    for i in range(0, len(files), batch_size):
        batch = [(f, output_dir) for f in files[i:i+batch_size]]
        
        with Pool(processes=get_safe_worker_count(workers)) as pool:
            batch_errors = filter(None, pool.map(process_file, batch))
            errors.extend(batch_errors)
        
        if errors:
            with open(error_log, 'a') as f:
                for filename, error in errors[-10:]:
                    f.write(f"{filename}\t{error}\n")
    
    print(f"Completed with {len(errors)} errors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data", help="Input directory")
    parser.add_argument("--output", default="extracted", help="Output directory")
    parser.add_argument("--workers", type=int, default=2, help="Max workers")
    args = parser.parse_args()
    
    print(f"Starting extraction with {args.workers} workers")
    batch_extract(args.input, args.output, args.workers)
