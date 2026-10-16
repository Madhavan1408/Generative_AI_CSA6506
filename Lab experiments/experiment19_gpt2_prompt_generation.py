"""Experiment 19: Generate text from a prompt using GPT-2."""
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Artificial intelligence can help society by"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=35,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
)
print("Prompt:", prompt)
print("Generated text:", tokenizer.decode(output[0], skip_special_tokens=True))
