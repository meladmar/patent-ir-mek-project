# SET WEIGHT > 0 ON YOUR MODEL'S CHOICE
import os
import json
import pickle
import numpy as np
import faiss
import pytrec_eval
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import spacy

# ========== Configuration ==========
VARIANT = 'title_abstract_claims'
TOP_K = 1000
BM25_WEIGHT = 0  # Set to 0 for disabling
SPLADE_WEIGHT = 2
FAISS_WEIGHT = 0

# ========== Paths ==========
BASE = "/home/lucid/Desktop/eval"
BM25_PATH = f"{BASE}/bm25_index.pkl"
SPLADE_JSONL = f"{BASE}/splade_vectors_v3.jsonl"
FAISS_INDEX = f"{BASE}/wpi60k_faiss_cosine.index"
FAISS_DOCNOS = f"{BASE}/wpi60k_docnos_cosine.npy"
TOPICS_DIR = "/media/lucid/hitachi/MB204IR/WPI_60K/2500_topics/topics"
QRELS_PATH = "/media/lucid/hitachi/MB204IR/WPI_60K/2500_topics/test_qrels_for_all_topics.txt"
DOC_DICT_PATH = f"{BASE}/doc_dict.json"

# ========== Initialization ==========
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") if FAISS_WEIGHT > 0 else None
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ========== Helpers ==========
def lemmatize(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha]

def normalize(scores):
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

def extract_query(xml_path, variant):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    def safe(tag):
        el = root.find(f".//{tag}")
        return el.text.strip() if el is not None and el.text else ""

    title = safe("invention-title")
    abstract = " ".join(safe("abstract").split()[:500])
    claims = " ".join(safe("claims").split()[:500])
    description = " ".join(safe("description").split()[:500])

    if variant == 'title_abstract_claims':
        return f"{title} {abstract} {claims}"
    elif variant == 'title_abstract_claims_description':
        return f"{title} {abstract} {claims} {description}"
    elif variant == 'abstract_description':
        return f"{abstract} {description}"
    return ""

# ========== Core Scoring ==========
def hybrid_score(query, bm25, doc_ids, splade_vecs, faiss_idx, faiss_ids):
    lemmas = lemmatize(query)
    tokens = tokenizer.tokenize(" ".join(lemmas))

    bm25_scores, splade_scores, faiss_scores = np.zeros(len(doc_ids)), np.zeros(len(doc_ids)), np.zeros(len(doc_ids))

    if BM25_WEIGHT:
        raw = bm25.get_scores(lemmas)
        bm25_scores = normalize(raw)

    if SPLADE_WEIGHT:
        for i, doc_id in enumerate(doc_ids):
            vec = splade_vecs.get(doc_id, {})
            splade_scores[i] = sum(vec.get(t, 0) for t in tokens)
        splade_scores = normalize(splade_scores)

    if FAISS_WEIGHT:
        emb = embedding_model.encode([query])[0].astype('float32')
        sims, idxs = faiss_idx.search(np.array([emb]), TOP_K)
        temp = {faiss_ids[i]: s for i, s in zip(idxs[0], sims[0])}
        faiss_scores = np.array([temp.get(doc, 0) for doc in doc_ids])
        faiss_scores = normalize(faiss_scores)

    combined = (BM25_WEIGHT * bm25_scores + SPLADE_WEIGHT * splade_scores + FAISS_WEIGHT * faiss_scores)
    top_ids = np.argsort(combined)[-TOP_K:][::-1]
    return [(doc_ids[i], combined[i]) for i in top_ids]

# ========== Main Evaluation ==========
def run():
    print(f"\n✨ Running hybrid evaluation for: {VARIANT}")
    print(f"⚖️ BM25: {BM25_WEIGHT}, SPLADE: {SPLADE_WEIGHT}, FAISS: {FAISS_WEIGHT}")

    # Load resources
    with open(BM25_PATH, 'rb') as f:
        bm25, doc_ids = pickle.load(f)

    splade_vecs = {}
    if SPLADE_WEIGHT:
        with open(SPLADE_JSONL) as f:
            for line in tqdm(f, desc="Loading SPLADE vectors"):
                j = json.loads(line)
                splade_vecs[j['id']] = j['vector']

    if FAISS_WEIGHT:
        faiss_idx = faiss.read_index(FAISS_INDEX)
        faiss_ids = np.load(FAISS_DOCNOS, allow_pickle=True)
    else:
        faiss_idx = faiss_ids = None

    with open(QRELS_PATH) as f:
        qrels = defaultdict(dict)
        for line in f:
            qid, _, did, rel = line.strip().split()
            qrels[qid][did] = int(rel)

    run = {}
    for fname in tqdm(sorted(os.listdir(TOPICS_DIR)), desc="Evaluating queries"):
        if not fname.endswith(".xml"): continue
        qid = fname.replace(".xml", "")
        query = extract_query(os.path.join(TOPICS_DIR, fname), VARIANT)
        results = hybrid_score(query, bm25, doc_ids, splade_vecs, faiss_idx, faiss_ids)
        run[qid] = {doc_id: float(score) for doc_id, score in results}

    metrics = {
        'recall_10', 'recall_100', 'recall_1000',
        'ndcg_cut_10', 'ndcg_cut_100', 'ndcg_cut_1000',
        'map_cut_10', 'map_cut_100', 'map_cut_1000'
    }

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    scores = evaluator.evaluate(run)

    print("\n📊 Results:")
    for m in ['recall_10', 'recall_100', 'recall_1000']:
        print(f"{m.upper():<15}: {np.mean([v.get(m, 0) for v in scores.values()]):.4f}")
    for m in ['ndcg_cut_10', 'ndcg_cut_100', 'ndcg_cut_1000']:
        print(f"{m.upper():<15}: {np.mean([v.get(m, 0) for v in scores.values()]):.4f}")
    for m in ['map_cut_10', 'map_cut_100', 'map_cut_1000']:
        print(f"{m.upper():<15}: {np.mean([v.get(m, 0) for v in scores.values()]):.4f}")

if __name__ == "__main__":
    run()

