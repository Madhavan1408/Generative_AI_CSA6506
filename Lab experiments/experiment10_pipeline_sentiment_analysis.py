"""Experiment 10: Sentiment analysis using the Hugging Face pipeline API."""
from transformers import pipeline

sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
sentence = "The new AI application is very useful."
prediction = sentiment(sentence)[0]

print("Sentence:", sentence)
print("Prediction:", prediction["label"])
print("Confidence:", round(prediction["score"], 4))
