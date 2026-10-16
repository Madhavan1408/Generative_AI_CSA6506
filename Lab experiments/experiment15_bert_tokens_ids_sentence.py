"""Experiment 15: Tokenize a sentence and display tokens and IDs."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
sentence = "Deep learning is interesting."
encoded = tokenizer(sentence)

print("Sentence:", sentence)
print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
print("Token IDs:", encoded["input_ids"])
