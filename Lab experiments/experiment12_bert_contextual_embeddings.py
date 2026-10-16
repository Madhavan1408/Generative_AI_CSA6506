"""Experiment 12: Generate contextual embeddings using pretrained BERT."""
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

sentence = "AI helps students learn faster."
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state
print("Embedding shape:", tuple(embeddings.shape))
print("First token embedding, first 5 values:")
print(embeddings[0, 0, :5])
