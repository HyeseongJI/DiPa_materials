import numpy as np
from PIL import Image

def is_background_tile(rgb_tile, threshold=0.60):
    """
    Determines whether a tile contains mostly background based on white pixel ratio.
    This is the primary filter used during tile extraction (e.g., Level 1, 512×512 patches).

    Parameters:
        rgb_tile (np.ndarray): RGB tile image (already cropped from the WSI)
        threshold (float): Proportion of white pixels above which the tile is discarded

    Returns:
        bool: True if the tile is mostly background (white), False otherwise

    Background:
        As described in the manuscript, tiles in which more than 60% of pixels have
        grayscale intensity ≥220 are considered background and excluded during preprocessing.
    """
    gray = np.mean(rgb_tile, axis=2)
    white_ratio = (gray > 220).sum() / gray.size
    return white_ratio > threshold


def is_background(image_path, threshold_white=230, threshold_black=25, ratio_thresh=0.85):
    """
    Determines whether an image tile is background based on the presence of extreme pixel values.
    This function serves as a secondary filter during dataset preparation (e.g., prior to K-Fold split).

    Parameters:
        image_path (str or Path): Path to the tile image file (e.g., PNG)
        threshold_white (int): Intensity threshold above which pixels are considered bright white
        threshold_black (int): Intensity threshold below which pixels are considered dark black
        ratio_thresh (float): If either white or black pixels exceed this ratio, the tile is excluded

    Returns:
        bool: True if the tile is mostly white or black (background or artifact), False otherwise

    Background:
        This step complements the primary filter by removing tiles that are either
        completely blank (white) or contain scanning artifacts (black),
        as described in the preprocessing section of the paper.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    white_mask = np.all(arr > threshold_white, axis=2)
    black_mask = np.all(arr < threshold_black, axis=2)

    white_ratio = np.mean(white_mask)
    black_ratio = np.mean(black_mask)

    return white_ratio > ratio_thresh or black_ratio > ratio_thresh
