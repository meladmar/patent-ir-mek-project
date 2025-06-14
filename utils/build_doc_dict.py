import os
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer

# Configuration
INPUT_DIR = "/media/lucid/hitachi/MB204IR/WPI_60K/extracted"
OUTPUT_JSON = "/home/lucid/Desktop/eval/doc_dict_optimized.json"
MAX_TOKENS = 512  # Strict token limit
WORD_LIMIT = 300  # Word limit for long sections

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def clean_text(text, word_limit=None):
    """Clean text with optional word truncation"""
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(filter(None, [clean_text(x) for x in text]))
    
    text = str(text)
    # Remove unwanted patterns
    text = re.sub(r'<[^>]+>|\[\w+\]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Word-level truncation if requested
    if word_limit:
        words = text.split()[:word_limit]
        text = ' '.join(words)
    
    return text

def truncate_by_tokens(text, max_tokens=MAX_TOKENS):
    """Ensure text stays within token limits"""
    tokens = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_string(tokens[:max_tokens])

def process_file(file_path):
    """Process a patent file with optimized text handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Core content with smart truncation
        doc = {
            "doc_id": data.get("docno", os.path.basename(file_path).replace(".json", "")),
            "title": clean_text(data.get("title", "")),
            "abstract": clean_text(data.get("abstract", "")),
            "claims": truncate_by_tokens(clean_text(data.get("claims", "")), MAX_TOKENS//3),
            "description": truncate_by_tokens(clean_text(data.get("description", "")), MAX_TOKENS//2)
        }
        
        # Create search-optimized text
        search_text = " ".join([
            doc["title"],
            doc["abstract"],
            doc["claims"][:WORD_LIMIT//2],  # Additional word limit
            doc["description"][:WORD_LIMIT]
        ])
        doc["search_text"] = truncate_by_tokens(search_text)
        
        return doc if len(doc["search_text"].split()) >= 30 else None
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def build_doc_dict():
    """Build optimized patent dictionary"""
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    doc_dict = {}
    
    for file in tqdm(files, desc="Processing patents"):
        processed = process_file(os.path.join(INPUT_DIR, file))
        if processed:
            doc_dict[processed["doc_id"]] = processed
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Processed {len(doc_dict)} patents")
    print(f"📝 Average description length: {sum(len(d['description'].split()) for d in doc_dict.values())//len(doc_dict)} words")
    print(f"🔍 Search text length: {len(next(iter(doc_dict.values()))['search_text'].split())} words (avg)")

if __name__ == "__main__":
    build_doc_dict()
