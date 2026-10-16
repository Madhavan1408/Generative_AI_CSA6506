"""Experiment 8: Tokenize a sentence using a BERT tokenizer."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
sentence = "Artificial intelligence is fascinating."

print("Original sentence:", sentence)
print("Tokens:", tokenizer.tokenize(sentence))
