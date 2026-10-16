"""Experiment 17: Generate BERT contextual embeddings and show dimensions."""
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

sentence = "Students are learning machine learning."
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state
print("Number of tokens:", embeddings.shape[1])
print("Embedding dimension:", embeddings.shape[2])
print("Embedding tensor shape:", tuple(embeddings.shape))
