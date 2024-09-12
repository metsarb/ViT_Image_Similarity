# Image Similarity Comparison Using ViT Model

This project uses the Vision Transformer (ViT) model to compare the similarity between two images. It computes the similarity score based on the cosine similarity of their embeddings.

## Requirements

The following Python libraries are used in this project:

- `torch` (PyTorch)
- `Pillow` (PIL for image processing)
- `transformers` (Hugging Face Transformers for ViT model)
- `sklearn` (for cosine similarity)
- `certifi` and `ssl` (for SSL certificate handling on macOS)

You can install the required libraries by running:
```bash
pip install torch pillow transformers scikit-learn certifi
