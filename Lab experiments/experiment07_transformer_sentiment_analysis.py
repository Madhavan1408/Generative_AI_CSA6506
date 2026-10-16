"""Experiment 7: Perform sentiment analysis with a pretrained Transformer."""
from transformers import pipeline

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)

sentence = "I really enjoyed learning artificial intelligence."
print("Input:", sentence)
print("Prediction:", classifier(sentence)[0])
