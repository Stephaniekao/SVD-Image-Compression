from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_image(image_path):
    """Load an image as the data matrix A for the SVD (used by run_real.py).

    Cleaning steps (record these in the report):
      1. Convert to grayscale with PIL's convert("L"):
         L = 0.299 R + 0.587 G + 0.114 B  (ITU-R 601-2 luma).
      2. Convert to float and divide by 255, so every entry A_ij is in [0, 1].
    One entry A_ij is the brightness of pixel (i, j).
    """
    with Image.open(image_path) as img:
        gray = img.convert("L")
        A = np.asarray(gray, dtype=float) / 255.0
    return A


def analyze_image(image_path, output_dir):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load image (RGB for exploration, grayscale A for the method)
    with Image.open(image_path) as img:
        image = np.asarray(img.convert("RGB"))
    A = load_image(image_path)

    # 2. Basic information
    height, width, channels = image.shape

    print("\n=== Image Information ===")
    print(f"File: {image_path.name}")
    print(f"Dimensions: {height} x {width}")
    print(f"Channels: {channels}")
    print(f"Data type: {image.dtype}")
    print(f"Total pixels: {height * width:,}")
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

    # 4. Grayscale data matrix A (what the SVD actually uses)
    print("\n=== Grayscale Matrix A (used for the SVD) ===")
    print(f"Shape (m x n): {A.shape[0]} x {A.shape[1]}")
    print(f"Data type: {A.dtype}")
    print(f"Missing values (NaN): {np.isnan(A).sum()}")   # checked on float data
    print(f"Min={A.min():.4f}, Max={A.max():.4f}, "
          f"Mean={A.mean():.4f}, Std={A.std():.4f}")
    print(f"||A||_F^2 = {np.sum(A**2):.4f}  (= sum of sigma_i^2, Iteration 2 Q4)")

    # 5. Figure 1: Original image
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

    # 6. Figure 2: RGB intensity distribution
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

    # 7. Figure 3: Grayscale image A
    plt.figure(figsize=(7, 5))
    plt.imshow(A, cmap="gray", vmin=0, vmax=1)
    plt.title("Grayscale Image A (scaled to [0, 1])")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_dir / "grayscale_image.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\n=== Figures Saved ===")
    print(output_dir / "original_image.png")
    print(output_dir / "rgb_histogram.png")
    print(output_dir / "grayscale_image.png")


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
