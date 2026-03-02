from huggingface_hub import snapshot_download

# Download the Dolphin3.0-Llama3.1-8B model to D: drive
repo_id = "dphn/Dolphin3.0-Llama3.1-8B"
local_dir = "D:/Dolphin3.0-Llama3.1-8B"

print(f"Starting download of {repo_id} to {local_dir}...")
snapshot_download(repo_id=repo_id, local_dir=local_dir)
print("Download completed!")