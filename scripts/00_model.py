"""Download the embedding model (all-MiniLM-L6-v2, ONNX export)."""
from huggingface_hub import snapshot_download

p = snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    allow_patterns=["onnx/model.onnx", "*.json", "*.txt"],
    local_dir="models/all-MiniLM-L6-v2",
)
print("model at:", p)
