"""Experiment 14: Answer a question using a pretrained Transformer."""
from transformers import pipeline

qa = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad",
)

context = (
    "Python is a programming language. It is widely used in "
    "artificial intelligence and data science."
)
question = "Where is Python widely used?"
result = qa(question=question, context=context)

print("Context:", context)
print("Question:", question)
print("Answer:", result["answer"])
print("Score:", round(result["score"], 4))
