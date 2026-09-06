from pathlib import Path

import torch
import open_clip
from PIL import Image, ImageFilter

def apply_blur(image, severity="medium"):
    blur_radius = {
        "mild": 1,
        "medium": 2,
        "severe": 4
    }

    return image.filter(
        ImageFilter.GaussianBlur(radius=blur_radius[severity])
    )


device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

#load CLIP ViT-B/32
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

#load 300-image gallery
image_paths = sorted(Path("images").glob("*.jpg"))

print(f"Encoding {len(image_paths)} images...")

image_features = []

with torch.no_grad():
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")

        blur_mode = "severe"

        if blur_mode != "clean":
            image = apply_blur(image, blur_mode)

        image_tensor = preprocess(image).unsqueeze(0).to(device)

        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

        image_features.append(features.cpu())

image_features = torch.cat(image_features)

#user can type any query
query = input("\nEnter a text query: ")

text = tokenizer([query]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#compare text against all 300 images
similarities = text_features.cpu() @ image_features.T

#top 5 matches
top_scores, top_indices = similarities[0].topk(5)

print("\nTop 5 retrieved images:\n")

for rank, (score, index) in enumerate(
    zip(top_scores, top_indices), start=1
):
    print(
        f"{rank}. {image_paths[index.item()].name} "
        f"| similarity = {score.item():.4f}"
    )
