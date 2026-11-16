from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def analyze_image(image_path, output_dir):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load image
    with Image.open(image_path) as img:
        rgb_image = img.convert("RGB")
        image = np.asarray(rgb_image)

    # 2. Basic information
    height, width, channels = image.shape

    print("\n=== Image Information ===")
    print(f"File: {image_path.name}")
    print(f"Dimensions: {height} x {width}")
    print(f"Channels: {channels}")
    print(f"Data type: {image.dtype}")
    print(f"Total pixels: {height * width:,}")
    print(f"Missing values: {np.isnan(image).sum()}")
    print(f"Minimum pixel value: {image.min()}")
    print(f"Maximum pixel value: {image.max()}")
    print(f"Mean pixel value: {image.mean():.2f}")
    print(f"Standard deviation: {image.std():.2f}")

    # 3. Channel statistics
    print("\n=== RGB Channel Statistics ===")

    for i, channel in enumerate(["Red", "Green", "Blue"]):
        values = image[:, :, i]

        print(
            f"{channel}: "
            f"Min={values.min()}, "
            f"Max={values.max()}, "
            f"Mean={values.mean():.2f}, "
            f"Std={values.std():.2f}"
        )

    # 4. Figure 1: Original image
    plt.figure(figsize=(7, 5))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_dir / "original_image.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 5. Figure 2: RGB intensity distribution
    plt.figure(figsize=(8, 5))

    colors = ["red", "green", "blue"]
    labels = ["Red", "Green", "Blue"]

    for i in range(3):
        plt.hist(
            image[:, :, i].ravel(),
            bins=np.arange(257),
            color=colors[i],
            alpha=0.4,
            label=labels[i]
        )

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("RGB Pixel Intensity Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "rgb_histogram.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\n=== Figures Saved ===")
    print(output_dir / "original_image.png")
    print(output_dir / "rgb_histogram.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze an image for the SVD project."
    )

    parser.add_argument(
        "image_path",
        help="Path to the input image"
    )

    parser.add_argument(
        "--output_dir",
        default="figures",
        help="Directory for output figures"
    )

    args = parser.parse_args()

    analyze_image(args.image_path, args.output_dir)
