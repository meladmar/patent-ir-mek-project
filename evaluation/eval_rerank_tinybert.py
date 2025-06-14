import os
import json
import pickle
import numpy as np
import pytrec_eval
import faiss
import spacy
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Optional reranker
USE_RERANKER = True  # Set to False to disable reranking
RERANKER_TOP_K = 100
FINAL_TOP_K = 750

if USE_RERANKER:
    from sentence_transformers import CrossEncoder
    #reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512)    

# ====== Configuration ======
variant = 'title_abstract_claims'
BM25_WEIGHT = 4
SPLADE_WEIGHT = 4
FAISS_WEIGHT = 2

base = "/home/lucid/Desktop/eval"
bm25_index_path = f"{base}/bm25_index.pkl"
splade_jsonl_path = f"{base}/splade_vectors_v3.jsonl"
topics_dir = f"{base}/test_topics"
qrels_path = f"{base}/Prior_art_qrels.txt"
documents_path = f"{base}/doc_dict.json"
faiss_index_path = f"{base}/wpi60k_faiss_cosine.index"
faiss_docnos_path = f"{base}/wpi60k_docnos_cosine.npy"

# ====== NLP + Tokenizer ======
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ====== Functions ======
def lemmatize_query(text):
    return [t.lemma_ for t in nlp(text.lower()) if t.is_alpha]

def extract_query(xml_path, variant):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    def grab(tag):
        el = root.find(f".//{tag}")
        return el.text.strip() if el is not None and el.text else ""
    title, abstract = grab("invention-title"), grab("abstract")
    claims, description = grab("claims"), grab("description")
    if variant == 'title_abstract_claims': return f"{title} {abstract} {claims}"
    if variant == 'title_abstract_claims_description': return f"{title} {abstract} {claims} {description}"
    if variant == 'abstract_claims': return f"{abstract} {claims}"
    if variant == 'title_claims_description': return f"{title} {claims} {description}"
    return ""

def normalize_scores(scores):
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

def hybrid_score(query_text, bm25, doc_ids, splade_vectors, faiss_index, faiss_docnos):
    lemmas = lemmatize_query(query_text)
    tokens = tokenizer.tokenize(" ".join(lemmas))
    
    bm25_raw = bm25.get_scores(lemmas)
    bm25_norm = normalize_scores(bm25_raw)

    splade_raw = np.zeros(len(doc_ids))
    for i, doc_id in enumerate(doc_ids):
        vec = splade_vectors.get(doc_id, {})
        splade_raw[i] = sum(vec.get(tok, 0) for tok in tokens)
    splade_norm = normalize_scores(splade_raw)

    expanded_query = f"{query_text} {' '.join(lemmas)}"
    query_emb = embedding_model.encode([expanded_query])[0]
    faiss_scores, faiss_indices = faiss_index.search(np.array([query_emb]).astype('float32'), len(doc_ids))
    faiss_score_map = {faiss_docnos[i]: float(s) for i, s in zip(faiss_indices[0], faiss_scores[0])}
    faiss_raw = np.array([faiss_score_map.get(did, 0) for did in doc_ids])
    faiss_norm = normalize_scores(faiss_raw)

    qlen = len(tokens)
    weights = [BM25_WEIGHT * min(1, 15/qlen), SPLADE_WEIGHT, FAISS_WEIGHT * min(2, qlen/10)]
    hybrid = weights[0] * bm25_norm + weights[1] * splade_norm + weights[2] * faiss_norm

    top_k = RERANKER_TOP_K if USE_RERANKER else FINAL_TOP_K
    top_indices = np.argsort(hybrid)[-top_k:][::-1]
    return [(doc_ids[i], hybrid[i]) for i in top_indices]

def run_hybrid_eval(variant):
    print(f"\n✨ Running hybrid evaluation for: {variant} (reranker={'on' if USE_RERANKER else 'off'})")
    print(f"⚖️ Weights - BM25: {BM25_WEIGHT}, SPLADE: {SPLADE_WEIGHT}, FAISS: {FAISS_WEIGHT}")

    with open(bm25_index_path, 'rb') as f:
        bm25, doc_ids = pickle.load(f)
    with open(splade_jsonl_path) as f:
        splade_vectors = {json.loads(l)['id']: json.loads(l)['vector'] for l in f}
    faiss_index = faiss.read_index(faiss_index_path)
    faiss_docnos = np.load(faiss_docnos_path, allow_pickle=True)
    with open(documents_path) as f:
        documents = json.load(f)

    qrels = defaultdict(dict)
    with open(qrels_path) as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels[qid][docid] = int(rel)

    run = {}
    for xml_file in tqdm(sorted(f for f in os.listdir(topics_dir) if f.endswith('.xml')), desc="Evaluating queries"):
        qid = xml_file.replace('.xml', '')
        query_text = extract_query(os.path.join(topics_dir, xml_file), variant)
        if not query_text:
            continue

        hybrid = hybrid_score(query_text, bm25, doc_ids, splade_vectors, faiss_index, faiss_docnos)

        if USE_RERANKER:
            #doc_texts = [(query_text, documents[doc_id].get("search_text", "")) for doc_id, _ in hybrid]
            #reranked = reranker.predict(doc_texts)
            #sorted_docs = sorted(zip([doc for doc, _ in hybrid], reranked), key=lambda x: x[1], reverse=True)
            #run[qid] = {doc: float(score) for doc, score in sorted_docs[:FINAL_TOP_K]}
            #try this:
            doc_texts = []
            for doc_id, _ in hybrid:
                    doc_entry = documents.get(doc_id, {})
                    if isinstance(doc_entry, dict):
                    	text = doc_entry.get("search_text", "")
                    else:
                    	text = doc_entry  # If it's a plain string
                    doc_texts.append((query_text, text))
            
            reranked_scores = reranker.predict(doc_texts)
            sorted_docs = sorted(zip([doc_id for doc_id, _ in hybrid], reranked_scores), key=lambda x: x[1], reverse=True)
            run[qid] = {doc: float(score) for doc, score in sorted_docs[:FINAL_TOP_K]}
            
        else:
            run[qid] = {doc: float(score) for doc, score in hybrid[:FINAL_TOP_K]}

    metrics = {
        'recall_10', 'recall_100', 'recall_1000',
        'ndcg_cut_10', 'ndcg_cut_100', 'ndcg_cut_1000',
        'map_cut_10', 'map_cut_100', 'map_cut_1000'
    }

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)

    print("\n📊 Evaluation Results:")
    for group, keys in [
        ("Recall", ['recall_10', 'recall_100', 'recall_1000']),
        ("NDCG", ['ndcg_cut_10', 'ndcg_cut_100', 'ndcg_cut_1000']),
        ("MAP", ['map_cut_10', 'map_cut_100', 'map_cut_1000'])
    ]:
        print(f"=== {group} ===")
        for k in keys:
            print(f"{k.upper():14}: {np.mean([v.get(k, 0) for v in results.values()]):.4f}")

if __name__ == "__main__":
    run_hybrid_eval(variant)

