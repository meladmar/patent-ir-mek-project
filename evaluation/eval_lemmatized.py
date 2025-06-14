import os
import json
import pickle
import numpy as np
import pandas as pd
import pytrec_eval
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ===== Configuration =====
variant = 'title_abstract_claims'  # Options: title_abstract_claims, title_abstract_claims_description
TOP_K = 1000
BM25_WEIGHT = 3  # BM25 weight
SPLADE_WEIGHT = 4  # SPLADE weight
FAISS_WEIGHT = 3  # FAISS weight

# ===== Paths =====
base = "/home/lucid/Desktop/eval"  # Update if needed to /content/drive/MyDrive/WPI_60K/
bm25_index_path = f"{base}/bm25_index.pkl"
splade_jsonl_path = f"{base}/splade_vectors_v3.jsonl"
topics_dir = f"{base}/test_topics"
qrels_path = f"{base}/Prior_art_qrels.txt"
documents_path = f"{base}/doc_dict.json"
faiss_index_path = f"{base}/wpi60k_faiss_cosine.index"
faiss_docnos_path = f"{base}/wpi60k_docnos_cosine.npy"

# ===== Initialize Models =====
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===== Query Processing =====
def lemmatize_query(query_text):
    """Consistent query lemmatization matching BM25 index"""
    doc = nlp(query_text.lower())
    return [token.lemma_ for token in doc if token.is_alpha]

def extract_query(xml_path, variant):
    """Extract query components from patent XML"""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    def safe_extract(tag):
        el = root.find(f".//{tag}")
        return el.text.strip() if el is not None and el.text else ""

    title = safe_extract("invention-title")
    abstract = " ".join(safe_extract("abstract").split()[:500])
    claims = " ".join(safe_extract("claims").split()[:500])
    description = " ".join(safe_extract("description").split()[:500])

    if variant == 'title_abstract_claims':
        return f"{title} {abstract} {claims}"
    elif variant == 'title_abstract_claims_description':
        return f"{title} {abstract} {claims} {description}"
    elif variant == 'abstract_claims':
        return f"{abstract} {claims}"    
    elif variant == 'title_claims_description':
        return f"{title} {claims} {description}"    
    return ""

# ===== Scoring Functions =====
def normalize_scores(scores):
    """Normalize scores to [0,1] range"""
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

def hybrid_score(query_text, bm25, doc_ids, splade_vectors, faiss_index, faiss_docnos):
    """Enhanced with query lemmatization and dynamic weighting"""
    # Consistent lemmatization
    query_lemmas = lemmatize_query(query_text)
    query_tokens = tokenizer.tokenize(" ".join(query_lemmas))  # SPLADE-compatible
    
    # BM25 with lemmatized query
    bm25_raw = bm25.get_scores(query_lemmas)  # Now using lemmas
    bm25_norm = normalize_scores(bm25_raw)
    
    # SPLADE (unchanged but benefits from alignment)
    splade_raw = np.zeros(len(doc_ids))
    for i, doc_id in enumerate(doc_ids):
        vec = splade_vectors.get(doc_id, {})
        splade_raw[i] = sum(vec.get(tok, 0) for tok in query_tokens)
    splade_norm = normalize_scores(splade_raw)
    
    # FAISS (now with query expansion)
    expanded_query = f"{query_text} {' '.join(query_lemmas)}"  # Boosts recall
    query_embedding = embedding_model.encode([expanded_query])[0]
    faiss_scores, faiss_indices = faiss_index.search(np.array([query_embedding]).astype('float32'), TOP_K)
    faiss_scores_dict = {faiss_docnos[i]: float(s) for i,s in zip(faiss_indices[0], faiss_scores[0])}
    faiss_raw = np.array([faiss_scores_dict.get(did, 0) for did in doc_ids])
    faiss_norm = normalize_scores(faiss_raw)
    
    # Dynamic weighting based on query length
    query_len = len(query_tokens)
    dynamic_weights = [
        BM25_WEIGHT * min(1, 15/query_len),  # Favors shorter queries
        SPLADE_WEIGHT,
        FAISS_WEIGHT * min(2, query_len/10)   # Favors longer queries
    ]
    
    combined = (dynamic_weights[0] * bm25_norm + 
               dynamic_weights[1] * splade_norm + 
               dynamic_weights[2] * faiss_norm)
    
    top_indices = np.argsort(combined)[-TOP_K:][::-1]
    return [(doc_ids[i], combined[i]) for i in top_indices]

# ===== Main Evaluation =====
def run_hybrid_eval(variant):
    print(f"\n✨ Running hybrid evaluation for: {variant}")
    print(f"⚖️ Weights - BM25: {BM25_WEIGHT}, SPLADE: {SPLADE_WEIGHT}, FAISS: {FAISS_WEIGHT}")
    
    # Load resources
    print("Loading resources...")
    with open(bm25_index_path, 'rb') as f:
        bm25, doc_ids = pickle.load(f)
    
    splade_vectors = {}
    with open(splade_jsonl_path) as f:
        for line in tqdm(open(splade_jsonl_path), desc="Loading SPLADE vectors"):
            item = json.loads(line)
            splade_vectors[item['id']] = item['vector']
    
    faiss_index = faiss.read_index(faiss_index_path)
    faiss_docnos = np.load(faiss_docnos_path, allow_pickle=True)
    
    with open(documents_path) as f:
        documents = json.load(f)
    
    # Load qrels
    qrels = defaultdict(dict)
    with open(qrels_path) as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels[qid][docid] = int(rel)

    # Process queries
    run = {}
    xml_files = sorted(f for f in os.listdir(topics_dir) if f.endswith('.xml'))
    
    for xml_file in tqdm(xml_files, desc="Evaluating queries"):
        qid = xml_file.replace('.xml', '')
        query_text = extract_query(os.path.join(topics_dir, xml_file), variant)
        
        if query_text:
            results = hybrid_score(query_text, bm25, doc_ids, splade_vectors, faiss_index, faiss_docnos)
            run[qid] = {doc_id: float(score) for doc_id, score in results}

 # Evaluate
    metrics = {
    'recall_10', 'recall_100', 'recall_1000',
    'ndcg_cut_10', 'ndcg_cut_100', 'ndcg_cut_1000',
    'map_cut_10', 'map_cut_100', 'map_cut_1000'
    }
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    eval_results = evaluator.evaluate(run)
    
    # Print results
    # Group metrics for better readability
    print("\n📊 Evaluation Results:")
    print("=== Recall ===")
    print(f"R@10           : {np.mean([res.get('recall_10', 0) for res in eval_results.values()]):.4f}")
    print(f"R@100          : {np.mean([res.get('recall_100', 0) for res in eval_results.values()]):.4f}")
    print(f"R@1000         : {np.mean([res.get('recall_1000', 0) for res in eval_results.values()]):.4f}")

    print("\n=== NDCG ===")
    print(f"NDCG@10        : {np.mean([res.get('ndcg_cut_10', 0) for res in eval_results.values()]):.4f}")
    print(f"NDCG@100       : {np.mean([res.get('ndcg_cut_100', 0) for res in eval_results.values()]):.4f}")
    print(f"NDCG@1000      : {np.mean([res.get('ndcg_cut_1000', 0) for res in eval_results.values()]):.4f}")

    print("\n=== MAP ===")
    print(f"MAP@10         : {np.mean([res.get('map_cut_10', 0) for res in eval_results.values()]):.4f}")
    print(f"MAP@100        : {np.mean([res.get('map_cut_100', 0) for res in eval_results.values()]):.4f}")
    print(f"MAP@1000       : {np.mean([res.get('map_cut_1000', 0) for res in eval_results.values()]):.4f}")

if __name__ == "__main__":
    run_hybrid_eval(variant)
