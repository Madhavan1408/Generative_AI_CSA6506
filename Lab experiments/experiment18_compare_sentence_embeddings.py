"""Experiment 18: Compare two sentence embeddings using cosine similarity.

Note: Base BERT is not specifically trained as a sentence-similarity model;
the score is a simple demonstration of mean-pooled embedding similarity.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

sentences = [
    "I love learning artificial intelligence.",
    "I enjoy studying artificial intelligence.",
]
sentence_vectors = []

with torch.no_grad():
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        token_embeddings = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        sentence_vectors.append(pooled)

similarity = F.cosine_similarity(sentence_vectors[0], sentence_vectors[1]).item()
print("Sentence 1:", sentences[0])
print("Sentence 2:", sentences[1])
print("Cosine similarity:", round(similarity, 4))
