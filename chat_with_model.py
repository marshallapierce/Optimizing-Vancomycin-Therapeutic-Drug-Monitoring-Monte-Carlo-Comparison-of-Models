from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to the downloaded model
model_path = "D:/Dolphin3.0-Llama3.1-8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model... (This may take a while)")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto"
)

print("Model loaded! You can now ask questions. Type 'quit' to exit.")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break

    # Format as ChatML
    chat_input = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # Limit for CPU speed
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    assistant_start = response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    assistant_response = response[assistant_start:].split("<|im_end|>")[0].strip()

    print(f"AI: {assistant_response}")