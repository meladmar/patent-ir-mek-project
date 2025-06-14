# ✅ Rebuild BM25 Index from Clean doc_dict.json

import json
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import os
import re
import spacy


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def lemmatize_text(text):
    """Convert words to base forms (devices → device, fabricating → fabricate)"""
    doc = nlp(text.lower())  # Process lowercase text
    return [token.lemma_ for token in doc if token.is_alpha]  # Keep only alphabetic tokens

# === Paths (edit if needed)
DOC_DICT_PATH = "/home/lucid/Desktop/eval/doc_dict.json"
BM25_OUTPUT_PATH = "/home/lucid/Desktop/eval/bm25_index_v2.pkl"

# === Load clean doc_dict
with open(DOC_DICT_PATH) as f:
    doc_dict = json.load(f)

# === Preprocess: use search_text if available, else compose it
corpus = []
doc_ids = []

for doc_id, content in tqdm(doc_dict.items(), desc="Preparing corpus"):
    if isinstance(content, dict):
        text = content.get("search_text")
        if not text:
            text = " ".join(
                filter(None, [
                    content.get("title"),
                    content.get("abstract"),
                    content.get("claims"),
                    content.get("description")
                ])
            )
    else:
        text = str(content)  # fallback

    tokens = lemmatize_text(text)  
    if len(tokens) >= 5:
        corpus.append(tokens)
        doc_ids.append(doc_id)

# === Build BM25
print("\n⚙️ Building BM25 index...")
#bm25 = BM25Okapi(corpus)
# === Build BM25 (Optimized for Patents)
bm25 = BM25Okapi(
    corpus,
    k1=1.6,    # More aggressive term saturation (patents have repeated terms)
    b=0.9,     # Stronger length normalization (accounts for varying patent lengths)
    epsilon=0.25  # Better handling of long documents
)
# === Save index
with open(BM25_OUTPUT_PATH, "wb") as f:
    pickle.dump((bm25, doc_ids), f)

print(f"\n✅ Saved BM25 index → {BM25_OUTPUT_PATH}")
print(f"📄 Documents indexed: {len(doc_ids)}")

