import os
import subprocess
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# PNG Save Function (Lossless, default compression)
# --------------------------------------------------
def save_png(image: Image.Image, save_path: Path):
    """
    Save an image in PNG format using PIL.

    PNG is a lossless format. This function uses the default compression settings,
    which are sufficient for most use cases in digital pathology.

    Args:
        image (PIL.Image): RGB image object
        save_path (Path): output path ending with .png
    """
    image.save(save_path)
    print(f"✅ Saved PNG: {save_path}")


# --------------------------------------------------
# TIFF Save Function (LZW lossless compression)
# --------------------------------------------------
def save_tiff_lzw(image: Image.Image, save_path: Path):
    """
    Save an image in TIFF format using LZW lossless compression.

    This is commonly used for efficient storage with no information loss.
    Compatible with many digital pathology viewers.

    Args:
        image (PIL.Image): RGB image object
        save_path (Path): output path ending with .tiff
    """
    image.save(save_path, compression="tiff_lzw")
    print(f"✅ Saved TIFF_LZW: {save_path}")


# --------------------------------------------------
# JPEG2000 Save Function (via OpenJPEG CLI tool)
# --------------------------------------------------
def convert_to_jp2(input_path: Path, output_path: Path, rate: int = 10):
    """
    Convert an image file to JPEG2000 format using OpenJPEG's opj_compress tool.

    JPEG2000 supports lossy compression with adjustable rates.
    A compression rate of 10 corresponds to ~10:1 compression (original bpp / 10).
    Requires `opj_compress` to be installed on the system.

    Args:
        input_path (Path): input image path (e.g., PNG or TIFF)
        output_path (Path): output .jp2 path
        rate (int): compression ratio (default: 10)

    Example:
        convert_to_jp2("tile.png", "tile.jp2", rate=10)
    """
    cmd = [
        "opj_compress",
        "-i", str(input_path),
        "-o", str(output_path),
        "-r", str(rate)
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Converted to JP2 (rate={rate}): {output_path}")
