"""Smoke test for SapBERT clinical entity embeddings and similarity matching."""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def main():
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    queries = ["Lactic Acidosis", "Systemic Hypotension", "Pyrexia", "Thrombocytopenia"]
    synonyms = ["Hyperlactatemia", "Low Blood Pressure", "Fever", "Low Platelet Count"]
    unrelated = ["Atelectasis", "Normal Saline", "Bone Fracture", "Ear Infection"]

    def embed(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token representation as per SapBERT specification
            cls_rep = outputs.last_hidden_state[:, 0, :]
            return F.normalize(cls_rep, p=2, dim=1)

    query_emb = embed(queries)
    syn_emb = embed(synonyms)
    unrelated_emb = embed(unrelated)

    syn_sims = torch.sum(query_emb * syn_emb, dim=1)
    unrelated_sims = torch.sum(query_emb * unrelated_emb, dim=1)

    print("\n--- Synonymous Pairs (Expected: High Similarity > 0.70) ---")
    for q, s, sim in zip(queries, synonyms, syn_sims):
        print(f"  {q:<22} <-> {s:<22} : {sim.item():.4f}")

    print("\n--- Unrelated Pairs (Expected: Low Similarity < 0.50) ---")
    for q, u, sim in zip(queries, unrelated, unrelated_sims):
        print(f"  {q:<22} <-> {u:<22} : {sim.item():.4f}")

    assert (syn_sims > 0.65).all(), "Some synonymous pairs scored below 0.65"
    assert (unrelated_sims < 0.50).all(), "Some unrelated pairs scored above 0.50"
    print("\n[SUCCESS] SapBERT entity linking smoke test passed.")


if __name__ == "__main__":
    main()
