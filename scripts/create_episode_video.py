#!/usr/bin/env python3
"""
Script to create a 2x2 grid video from episode evaluation images.

Usage:
    python scripts/create_episode_video.py <episode_folder_path> [--output <output_path>]

Example:
    python scripts/create_episode_video.py data/debug-logs/ppo_20251218_014810/eval_data/episode_0002
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np

# ============================================================================
# CONFIGURATION - Adjust these variables to change the grid layout
# ============================================================================

# Each position in the 2x2 grid is defined by (view, type)
# Available views: "birdview", "frontview", "sideview", "robot0_eye_in_hand"
# Available types: "depth", "gt", "image", "reconstruction"

TOP_LEFT = ("robot0_eye_in_hand", "depth")
TOP_RIGHT = ("birdview", "reconstruction")
BOTTOM_LEFT = ("frontview", "image")
BOTTOM_RIGHT = ("birdview", "gt")

# Common size for all images (width, height)
COMMON_SIZE = (128, 128)

# Video settings
FPS = 3

# ============================================================================


def get_image_pattern(view: str, img_type: str) -> str:
    """Generate the image filename pattern for a given view and type."""
    return f"step_*_{view}_{img_type}.png"


def extract_step_number(filename: str) -> int:
    """Extract the step number from a filename like 'step_001_birdview_depth.png'."""
    match = re.search(r"step_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return 0


def load_and_resize_image(path: str, target_size: tuple) -> np.ndarray:
    """Load an image and resize it to the target size."""
    img = cv2.imread(path)
    if img is None:
        # Create a black placeholder image if the file doesn't exist
        print(f"Warning: Could not load image {path}")
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Resize to common size
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def get_sorted_image_files(folder: str, view: str, img_type: str) -> list:
    """Get sorted list of image files for a given view and type."""
    pattern = os.path.join(folder, get_image_pattern(view, img_type))
    files = glob.glob(pattern)
    files.sort(key=lambda x: extract_step_number(os.path.basename(x)))
    return files


def create_grid_frame(
    folder: str,
    step: int,
    top_left: tuple,
    top_right: tuple,
    bottom_left: tuple,
    bottom_right: tuple,
    common_size: tuple,
) -> np.ndarray:
    """Create a single 2x2 grid frame for a given step."""

    positions = [top_left, top_right, bottom_left, bottom_right]

    images = []
    for view, img_type in positions:
        filename = f"step_{step:03d}_{view}_{img_type}.png"
        filepath = os.path.join(folder, filename)
        img = load_and_resize_image(filepath, common_size)
        images.append(img)

    # Create 2x2 grid
    top_row = np.hstack([images[0], images[1]])
    bottom_row = np.hstack([images[2], images[3]])
    grid = np.vstack([top_row, bottom_row])

    return grid


def create_video(
    episode_folder: str,
    output_path: str = None,
) -> str:
    """
    Create a video from episode images.

    Args:
        episode_folder: Path to the episode folder containing images
        output_path: Path for the output video (default: episode_folder/episode_video.mp4)

    Returns:
        Path to the created video
    """
    episode_folder = os.path.abspath(episode_folder)

    if output_path is None:
        output_path = os.path.join(episode_folder, "episode_video.mp4")

    # Find all steps by looking at one of the image types
    reference_files = get_sorted_image_files(episode_folder, TOP_LEFT[0], TOP_LEFT[1])

    if not reference_files:
        raise ValueError(
            f"No images found in {episode_folder} for {TOP_LEFT[0]}_{TOP_LEFT[1]}"
        )

    # Extract step numbers
    steps = [extract_step_number(os.path.basename(f)) for f in reference_files]

    print(f"Found {len(steps)} steps")
    print(f"Grid layout:")
    print(f"  Top Left:     {TOP_LEFT[0]}/{TOP_LEFT[1]}")
    print(f"  Top Right:    {TOP_RIGHT[0]}/{TOP_RIGHT[1]}")
    print(f"  Bottom Left:  {BOTTOM_LEFT[0]}/{BOTTOM_LEFT[1]}")
    print(f"  Bottom Right: {BOTTOM_RIGHT[0]}/{BOTTOM_RIGHT[1]}")
    print(f"  Common Size:  {COMMON_SIZE}")

    # Create first frame to get dimensions
    first_frame = create_grid_frame(
        episode_folder,
        steps[0],
        TOP_LEFT,
        TOP_RIGHT,
        BOTTOM_LEFT,
        BOTTOM_RIGHT,
        COMMON_SIZE,
    )

    height, width = first_frame.shape[:2]
    print(f"  Video Size:   {width}x{height}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

    # Write frames
    for i, step in enumerate(steps):
        frame = create_grid_frame(
            episode_folder,
            step,
            TOP_LEFT,
            TOP_RIGHT,
            BOTTOM_LEFT,
            BOTTOM_RIGHT,
            COMMON_SIZE,
        )
        out.write(frame)

        if (i + 1) % 10 == 0 or (i + 1) == len(steps):
            print(f"  Processed {i + 1}/{len(steps)} frames")

    out.release()
    print(f"\nVideo saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a 2x2 grid video from episode evaluation images."
    )
    parser.add_argument(
        "episode_folder",
        type=str,
        help="Path to the episode folder containing images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output video path (default: episode_folder/episode_video.mp4)",
    )

    args = parser.parse_args()

    create_video(
        args.episode_folder,
        args.output,
    )


if __name__ == "__main__":
    main()
