"""Experiment 11: Display BERT tokens and token IDs for a sentence."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
sentence = "Machine learning is powerful."
encoded = tokenizer(sentence)

print("Sentence:", sentence)
print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
print("Token IDs:", encoded["input_ids"])
