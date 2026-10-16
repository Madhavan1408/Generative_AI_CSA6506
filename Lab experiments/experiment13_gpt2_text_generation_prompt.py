"""Experiment 13: Generate a text continuation using GPT-2."""
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The future of technology"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
print("Generated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
