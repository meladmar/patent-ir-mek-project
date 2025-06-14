import os
import json
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

def truncate_words(text, limit):
    return ' '.join(text.split()[:limit])

def build_bm25_index(json_dir, output_path):
    documents = {}
    for fname in tqdm(sorted(os.listdir(json_dir))):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(json_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)

            doc_id = data.get("docno", fname.replace(".json", ""))
            title = data.get("title", "")
            abstract = truncate_words(data.get("abstract", ""), 100)
            claims = truncate_words(data.get("claims", ""), 300)

            full_text = f"{title} {abstract} {claims}".strip()
            if full_text:
                documents[doc_id] = full_text
        except Exception as e:
            print(f"⚠️ Error parsing {fname}: {e}")

    # Tokenize
    tokenized_corpus = [tokenizer.tokenize(doc.lower()) for doc in documents.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_ids = list(documents.keys())

    # Save index
    with open(output_path, 'wb') as f:
        pickle.dump((bm25, doc_ids, documents), f)

    print(f"✅ Saved BM25 index with {len(documents)} documents to {output_path}")

# Example call
build_bm25_index(
    json_dir='/content/drive/MyDrive/WPI_60K/extracted',  # your full JSON dataset
    output_path='/content/drive/MyDrive/WPI_60K/bm25_index.pkl'
)

