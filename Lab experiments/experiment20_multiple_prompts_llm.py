"""Experiment 20: Generate and compare GPT-2 responses for multiple prompts."""
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompts = [
    "Artificial intelligence is",
    "Machine learning helps",
    "The future of robotics",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nPrompt:", prompt)
    print("Response:", response)
