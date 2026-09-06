from pathlib import Path
import gc

import numpy as np
import torch
import open_clip
from PIL import Image, ImageFilter, ImageDraw


#configuration

IMAGE_DIR = Path("images")
OUTPUT_DIR = Path("embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16

MODELS = {
    "CLIP ViT-B/32": {
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "filename": "clip_vit_b32",
    },
    "CLIP ViT-L/14": {
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "filename": "clip_vit_l14",
    },
    "SigLIP ViT-B/16": {
        "model_name": "hf-hub:timm/ViT-B-16-SigLIP",
        "pretrained": None,
        "filename": "siglip_vit_b16",
    },
}

SEVERITY_PARAMS = {
    "mild": {
        "blur": 1,
        "noise": 0.05,
        "occlusion": 0.10,
    },
    "medium": {
        "blur": 2,
        "noise": 0.10,
        "occlusion": 0.25,
    },
    "severe": {
        "blur": 4,
        "noise": 0.20,
        "occlusion": 0.40,
    },
}

DEGRADATIONS = [
    "Clean",
    "Blur",
    "Noise",
    "Occlusion",
    "Blur + Noise",
    "Blur + Occlusion",
    "Noise + Occlusion",
    "Blur + Noise + Occlusion",
]

SEVERITIES = [
    "mild",
    "medium",
    "severe",
]


#degradations

def apply_blur(image, severity):
    radius = SEVERITY_PARAMS[severity]["blur"]
    return image.filter(
        ImageFilter.GaussianBlur(radius=radius)
    )


def apply_noise(image, severity, seed=42):
    std = SEVERITY_PARAMS[severity]["noise"]

    array = (
        np.asarray(image)
        .astype(np.float32)
        / 255.0
    )

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=std,
        size=array.shape,
    )

    noisy = np.clip(
        array + noise,
        0.0,
        1.0,
    )

    noisy = (
        noisy * 255.0
    ).astype(np.uint8)

    return Image.fromarray(noisy)


def apply_occlusion(image, severity):
    fraction = SEVERITY_PARAMS[severity]["occlusion"]

    image = image.copy()
    width, height = image.size

    side = int(
        np.sqrt(
            fraction * width * height
        )
    )

    side = min(
        side,
        width,
        height,
    )

    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [left, top, right, bottom],
        fill=(0, 0, 0),
    )

    return image


def apply_degradation(
    image,
    degradation,
    severity,
    seed=42,
):
    if degradation == "Clean":
        return image.copy()

    output = image.copy()

    if "Blur" in degradation:
        output = apply_blur(
            output,
            severity,
        )

    if "Noise" in degradation:
        output = apply_noise(
            output,
            severity,
            seed=seed,
        )

    if "Occlusion" in degradation:
        output = apply_occlusion(
            output,
            severity,
        )

    return output


#load model

def load_model(model_choice):
    config = MODELS[model_choice]

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"\nLoading {model_choice}")
    print(f"Device: {device}")

    if config["pretrained"] is None:
        model, _, preprocess = (
            open_clip.create_model_and_transforms(
                config["model_name"]
            )
        )
    else:
        model, _, preprocess = (
            open_clip.create_model_and_transforms(
                config["model_name"],
                pretrained=config["pretrained"],
            )
        )

    model = model.to(device)
    model.eval()

    return model, preprocess, device


#gallery encoding

def encode_condition(
    model,
    preprocess,
    device,
    image_paths,
    degradation,
    severity,
):
    all_features = []

    with torch.no_grad():

        for batch_start in range(
            0,
            len(image_paths),
            BATCH_SIZE,
        ):
            batch_paths = image_paths[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            batch_tensors = []

            for local_index, image_path in enumerate(
                batch_paths
            ):
                gallery_index = (
                    batch_start + local_index
                )

                with Image.open(image_path) as img:
                    image = img.convert("RGB")

                image = apply_degradation(
                    image=image,
                    degradation=degradation,
                    severity=severity,
                    seed=42 + gallery_index,
                )

                image_tensor = preprocess(
                    image
                )

                batch_tensors.append(
                    image_tensor
                )

            image_batch = torch.stack(
                batch_tensors
            ).to(device)

            features = model.encode_image(
                image_batch
            )

            features = features / (
                features.norm(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-12
            )

            all_features.append(
                features.cpu().float()
            )

            del image_batch
            del features

    return torch.cat(
        all_features,
        dim=0,
    ).numpy()



def main():

    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    if not image_paths:
        raise RuntimeError(
            "No JPG images found in images/."
        )

    print(
        f"Found {len(image_paths)} gallery images."
    )

    image_names = np.array(
        [path.name for path in image_paths]
    )

    for model_choice, config in MODELS.items():

        model, preprocess, device = load_model(
            model_choice
        )

        conditions = [
            ("Clean", "mild")
        ]

        for degradation in DEGRADATIONS:

            if degradation == "Clean":
                continue

            for severity in SEVERITIES:
                conditions.append(
                    (degradation, severity)
                )

        print(
            f"{model_choice}: "
            f"{len(conditions)} gallery conditions"
        )

        for condition_number, (
            degradation,
            severity,
        ) in enumerate(
            conditions,
            start=1,
        ):

            if degradation == "Clean":
                condition_name = "clean"
            else:
                degradation_slug = (
                    degradation
                    .lower()
                    .replace(" + ", "_")
                    .replace(" ", "_")
                )

                condition_name = (
                    f"{degradation_slug}_{severity}"
                )

            output_path = (
                OUTPUT_DIR
                / (
                    f"{config['filename']}"
                    f"__{condition_name}.npz"
                )
            )

            if output_path.exists():
                print(
                    f"[{condition_number:02d}/22] "
                    f"Skipping existing: "
                    f"{output_path.name}"
                )
                continue

            print(
                f"[{condition_number:02d}/22] "
                f"{degradation} / {severity}"
            )

            features = encode_condition(
                model=model,
                preprocess=preprocess,
                device=device,
                image_paths=image_paths,
                degradation=degradation,
                severity=severity,
            )

            np.savez_compressed(
                output_path,
                features=features,
                image_names=image_names,
            )

            print(
                f"    saved {features.shape} "
                f"-> {output_path}"
            )

            del features

        del model
        del preprocess

        gc.collect()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("Precomputation complete.")

    files = list(
        OUTPUT_DIR.glob("*.npz")
    )

    print(
        f"Created {len(files)} embedding files."
    )


if __name__ == "__main__":
    main()
