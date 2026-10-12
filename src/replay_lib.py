import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataB"
CACHE = Path.home() / ".cache" / "huggingface"

_DOC_FILES = {
    "camera": "final_camera_reviews",
    "ev": "final_ev_reviews",
    "shoes": "final_shoe_reviews",
}


def load_corpus(domain):
    path = DATA / domain / "comments" / f"{_DOC_FILES[domain]}.jsonl"
    out = []
    for i, line in enumerate(path.open(), 1):
        rec = json.loads(line)
        rec["_line"] = i
        rec["_source"] = _DOC_FILES[domain]
        out.append(rec)
    return out


def load_pool(domain, kind):
    name = "adversarial_simple_100.jsonl" if kind == "simple" else "comments_adversarial.jsonl"
    path = DATA / domain / "adversarial" / name
    source = name.split(".")[0]
    out = []
    for i, line in enumerate(path.open(), 1):
        rec = json.loads(line)
        rec["_line"] = i
        rec["_source"] = source
        out.append(rec)
    return out


def format_text(rec, sep="\n"):
    parts = [f"title: {rec.get('title', '')}", f"text: {rec.get('text', '')}"]
    for i, r in enumerate(rec.get("replies", []) or []):
        parts.append(f"replies[{i}]: {r}")
    return sep.join(parts)


def load_queries(seed_json, count=None):
    d = json.load(open(seed_json))
    if count is None:
        return [r["query"] for r in d["counts"]["0"]["runs"]]
    return [r["query"] for r in d["counts"][str(count)]["runs"]]


def observed_injected(seed_json, count):
    d = json.load(open(seed_json))
    c = d["counts"][str(count)]
    sem, sim = set(), set()
    for r in c["runs"]:
        for it in r["retrieved"]:
            if it["source"] == "comments_adversarial":
                sem.add(it["line"])
            elif "simple" in it["source"]:
                sim.add(it["line"])
    return sorted(sem), sorted(sim)


class Embedder:
    def __init__(self, model_id="BAAI/bge-small-en-v1.5", pooling="mean", batch_size=64):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.pooling = pooling
        self.batch_size = batch_size

    def encode(self, texts):
        out = []
        torch = self.torch
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                if self.pooling == "mean":
                    v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                else:
                    v = h[:, 0]
                v = torch.nn.functional.normalize(v, p=2, dim=1)
                out.append(v.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


class FlatIndex:
    def __init__(self, vecs):
        import faiss
        self.index = faiss.IndexFlatL2(vecs.shape[1])
        self.index.add(vecs)

    def search(self, qvecs, k):
        d, i = self.index.search(qvecs, k)
        return d, i


def retrieve_blocks(corpus_recs, injected_recs, queries, embedder, chunk_fn, k=10):
    corpus_texts = [chunk_fn(format_text(r)) for r in corpus_recs]
    inj_texts = [chunk_fn(format_text(r)) for r in injected_recs]
    nodes = []
    for rec, chunks in [(r, c) for r, c in zip(corpus_recs, corpus_texts)]:
        for ch in chunks:
            nodes.append((rec["_source"], rec["_line"], ch))
    for rec, chunks in [(r, c) for r, c in zip(injected_recs, inj_texts)]:
        for ch in chunks:
            nodes.append((rec["_source"], rec["_line"], ch))
    vecs = embedder.encode([n[2] for n in nodes])
    idx = FlatIndex(vecs)
    qvecs = embedder.encode(queries)
    d, i = idx.search(qvecs, k)
    return nodes, d, i


def chunk_texts(texts, size=512, overlap=64):
    from llama_index.core.node_parser import SentenceSplitter
    sp = SentenceSplitter(chunk_size=size, chunk_overlap=overlap)
    out = []
    for t in texts:
        out.extend(sp.split_text(t))
    return out


def nodes_of(recs):
    nodes = []
    for rec in recs:
        for ch in chunk_texts([format_text(rec)]):
            nodes.append((rec["_source"], rec["_line"], ch))
    return nodes


class VecCache:
    def __init__(self, embedder):
        self.embedder = embedder
        self.map = {}

    def vecs_for(self, texts):
        miss = [t for t in texts if t not in self.map]
        if miss:
            vs = self.embedder.encode(miss)
            for t, v in zip(miss, vs):
                self.map[t] = v
        import numpy as np
        return np.vstack([self.map[t] for t in texts])


def build_index(node_list, cache):
    import faiss
    mat = cache.vecs_for([t for _, _, t in node_list])
    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)
    return index, len(node_list)


def exposure_of(index, base_n, qvecs, k=10):
    d, i = index.search(qvecs, k)
    counts = [(row >= base_n).sum() for row in i]
    inj_rows = []
    for row in i:
        inj_rows.append([(int(j), j >= base_n) for j in row])
    return sum(counts) / (len(counts) * k), inj_rows


def solve_brc_q(eta, omega, N, rho):
    from scipy.optimize import brentq
    f = lambda q: omega * eta * eta * (1 - q) ** 2 * (N - q) - q * rho * rho
    if f(1e-12) <= 0:
        return 0.0
    return brentq(f, 1e-12, 1 - 1e-12, xtol=1e-14)


def r_star(omega, N, rho, qbar):
    import numpy as np
    eta = rho * np.sqrt(qbar) / ((1 - qbar) * np.sqrt(omega * (N - qbar)))
    return 1.0 - eta
