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

# Each position in the 2x2 grid is defined by a lambda that takes a step number
# and returns the filename stem (without .png extension).
#
# For dynamic images that change per step:
#   lambda step: f"step_{step:03d}_robot0_eye_in_hand_depth"
#
# For static images that stay constant throughout the video:
#   lambda step: "birdview_mesh_gt"
#

TOP_LEFT = lambda step: f"step_{step:03d}_robot0_eye_in_hand_depth"
TOP_RIGHT = lambda step: f"step_{step:03d}_birdview_mesh_reconstruction"
BOTTOM_LEFT = lambda step: f"step_{step:03d}_frontview_image"
BOTTOM_RIGHT = lambda step: "birdview_mesh_gt"

# Common size for all images (width, height)
COMMON_SIZE = (128, 128)

# Video settings
FPS = 4

# ============================================================================


def get_dynamic_image_pattern(position_fn) -> str | None:
    """
    Generate the image filename pattern for a position function.
    Returns None if the position is static (doesn't depend on step).
    """
    # Test with two different step numbers to detect if it's dynamic
    test_step_1 = position_fn(1)
    test_step_2 = position_fn(2)

    if test_step_1 == test_step_2:
        # Static image - doesn't change with step
        return None

    # Dynamic image - replace the step number with wildcard
    # Find where the step number is in the filename and replace with *
    import re

    pattern = re.sub(r"step_\d+", "step_*", test_step_1)
    return f"{pattern}.png"


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


def get_sorted_image_files(folder: str, position_fn) -> list:
    """Get sorted list of image files for a given position function."""
    pattern_suffix = get_dynamic_image_pattern(position_fn)
    if pattern_suffix is None:
        # Static image - return empty list (we'll need to find steps another way)
        return []

    pattern = os.path.join(folder, pattern_suffix)
    files = glob.glob(pattern)
    files.sort(key=lambda x: extract_step_number(os.path.basename(x)))
    return files


def find_all_steps(folder: str, positions: list) -> list:
    """Find all step numbers by checking all dynamic positions."""
    for position_fn in positions:
        files = get_sorted_image_files(folder, position_fn)
        if files:
            return [extract_step_number(os.path.basename(f)) for f in files]

    raise ValueError(
        f"No dynamic images found in {folder}. At least one position must be dynamic."
    )


def create_grid_frame(
    folder: str,
    step: int,
    positions: list,
    common_size: tuple,
) -> np.ndarray:
    """Create a single 2x2 grid frame for a given step."""

    images = []
    for position_fn in positions:
        filename = f"{position_fn(step)}.png"
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

    positions = [TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT]

    # Find all steps by looking at dynamic image types
    steps = find_all_steps(episode_folder, positions)

    if not steps:
        raise ValueError(f"No images found in {episode_folder}")

    # Determine which positions are dynamic vs static
    def get_position_label(pos_fn):
        test_name = pos_fn(0)
        pattern = get_dynamic_image_pattern(pos_fn)
        if pattern is None:
            return f"{test_name} (static)"
        return test_name.replace("step_000_", "")

    print(f"Found {len(steps)} steps")
    print(f"Grid layout:")
    print(f"  Top Left:     {get_position_label(TOP_LEFT)}")
    print(f"  Top Right:    {get_position_label(TOP_RIGHT)}")
    print(f"  Bottom Left:  {get_position_label(BOTTOM_LEFT)}")
    print(f"  Bottom Right: {get_position_label(BOTTOM_RIGHT)}")
    print(f"  Common Size:  {COMMON_SIZE}")

    # Create first frame to get dimensions
    first_frame = create_grid_frame(
        episode_folder,
        steps[0],
        positions,
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
            positions,
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
