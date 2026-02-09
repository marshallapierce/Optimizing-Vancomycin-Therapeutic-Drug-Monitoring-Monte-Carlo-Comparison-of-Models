from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to the downloaded model
model_path = "D:/Dolphin3.0-Llama3.1-8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model... (This may take a while and use significant memory)")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,  # Use float16 for memory efficiency if GPU available
    device_map="auto"  # Automatically map to available devices
)

print("Model loaded successfully!")

# Example inference
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")