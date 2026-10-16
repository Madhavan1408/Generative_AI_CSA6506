"""Experiment 16: Compare tokenization output from BERT and GPT-2."""
from transformers import AutoTokenizer

bert = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
sentence = "I love artificial intelligence."

for name, tokenizer in [("BERT", bert), ("GPT-2", gpt2)]:
    print(f"\n{name}")
    print("Tokens:", tokenizer.tokenize(sentence))
    print("Token IDs:", tokenizer.encode(sentence, add_special_tokens=True))
