#!/usr/bin/env python3
"""
Script to check overlap between JSON level files in two folders.

Compares levels based on their core structure (polygons, circles, joints, thrusters)
ignoring runtime state like accumulated manifolds.

Usage:
    python scripts/check_level_overlap.py <folder1> <folder2>

Example:
    python scripts/check_level_overlap.py outputs/eval_variants outputs/hard_level_variants
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


def extract_level_signature(level_data: dict) -> str:
    """
    Extract a signature from the level that captures its core structure.
    Ignores runtime state like manifolds and accumulated impulses.
    """
    env_state = level_data.get("env_state", level_data)

    # Extract the core structural elements
    signature_parts = []

    # Polygon data (positions, vertices, roles, etc.)
    if "polygon" in env_state:
        for poly in env_state["polygon"]:
            if poly.get("active", True):
                sig = {
                    "position": poly.get("position"),
                    "rotation": round(poly.get("rotation", 0), 4),
                    "vertices": poly.get("vertices"),
                    "n_vertices": poly.get("n_vertices"),
                    "role": poly.get("role"),
                    "inverse_mass": poly.get("inverse_mass"),
                    "collision_mode": poly.get("collision_mode"),
                }
                signature_parts.append(("polygon", sig))

    # Circle data
    if "circle" in env_state:
        for circle in env_state["circle"]:
            if circle.get("active", True):
                sig = {
                    "position": circle.get("position"),
                    "radius": circle.get("radius"),
                    "role": circle.get("role"),
                    "inverse_mass": circle.get("inverse_mass"),
                }
                signature_parts.append(("circle", sig))

    # Joint data
    if "joint" in env_state:
        for joint in env_state["joint"]:
            if joint.get("active", True):
                sig = {
                    "a_index": joint.get("a_index"),
                    "b_index": joint.get("b_index"),
                    "a_relative_pos": joint.get("a_relative_pos"),
                    "b_relative_pos": joint.get("b_relative_pos"),
                    "motor_on": joint.get("motor_on"),
                    "motor_binding": joint.get("motor_binding"),
                    "is_fixed_joint": joint.get("is_fixed_joint"),
                }
                signature_parts.append(("joint", sig))

    # Thruster data
    if "thruster" in env_state:
        for thruster in env_state["thruster"]:
            if thruster.get("active", True):
                sig = {
                    "object_index": thruster.get("object_index"),
                    "relative_position": thruster.get("relative_position"),
                    "rotation": round(thruster.get("rotation", 0), 4),
                    "power": thruster.get("power"),
                    "thruster_binding": thruster.get("thruster_binding"),
                }
                signature_parts.append(("thruster", sig))

    # Gravity
    if "gravity" in env_state:
        signature_parts.append(("gravity", env_state["gravity"]))

    # Create a deterministic string representation
    sig_str = json.dumps(signature_parts, sort_keys=True)
    return hashlib.md5(sig_str.encode()).hexdigest()


def load_levels_from_folder(folder_path: Path) -> Dict[str, Tuple[str, dict]]:
    """
    Load all JSON level files from a folder.
    Returns dict mapping filename -> (signature, full_data)
    """
    levels = {}
    json_files = list(folder_path.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            signature = extract_level_signature(data)
            levels[json_file.name] = (signature, data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")

    return levels


def find_overlaps(folder1_levels: Dict, folder2_levels: Dict) -> List[Tuple[str, str]]:
    """
    Find levels that have matching signatures between two folders.
    Returns list of (folder1_filename, folder2_filename) pairs.
    """
    overlaps = []

    # Build reverse lookup for folder2: signature -> list of filenames
    sig_to_files2 = {}
    for filename, (sig, _) in folder2_levels.items():
        if sig not in sig_to_files2:
            sig_to_files2[sig] = []
        sig_to_files2[sig].append(filename)

    # Check each folder1 level against folder2
    for filename1, (sig1, _) in folder1_levels.items():
        if sig1 in sig_to_files2:
            for filename2 in sig_to_files2[sig1]:
                overlaps.append((filename1, filename2))

    return overlaps


def main():
    parser = argparse.ArgumentParser(description="Check for overlapping JSON levels between two folders")
    parser.add_argument("folder1", type=str, help="First folder path (e.g., outputs/eval_variants)")
    parser.add_argument(
        "folder2",
        type=str,
        help="Second folder path (e.g., outputs/hard_level_variants)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed comparison info")

    args = parser.parse_args()

    folder1 = Path(args.folder1)
    folder2 = Path(args.folder2)

    if not folder1.exists():
        print(f"Error: Folder '{folder1}' does not exist")
        return 1
    if not folder2.exists():
        print(f"Error: Folder '{folder2}' does not exist")
        return 1

    print(f"\nLoading levels from: {folder1}")
    levels1 = load_levels_from_folder(folder1)
    print(f"  Found {len(levels1)} JSON files")

    print(f"\nLoading levels from: {folder2}")
    levels2 = load_levels_from_folder(folder2)
    print(f"  Found {len(levels2)} JSON files")

    print("\nChecking for overlaps...")
    overlaps = find_overlaps(levels1, levels2)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Folder 1: {folder1} ({len(levels1)} levels)")
    print(f"Folder 2: {folder2} ({len(levels2)} levels)")
    print(f"\nOverlapping levels: {len(overlaps)}")

    if overlaps:
        print("\nMatches found:")
        for f1, f2 in overlaps:
            print(f"  {f1} <-> {f2}")

        # Calculate overlap percentages
        unique_f1 = len(set(f1 for f1, _ in overlaps))
        unique_f2 = len(set(f2 for _, f2 in overlaps))
        print(f"\n{unique_f1}/{len(levels1)} levels from folder1 have matches ({100*unique_f1/len(levels1):.1f}%)")
        print(f"{unique_f2}/{len(levels2)} levels from folder2 have matches ({100*unique_f2/len(levels2):.1f}%)")
    else:
        print("\nNo overlapping levels found! The two folders contain completely different levels.")

    if args.verbose and overlaps:
        print(f"\n{'='*60}")
        print("DETAILED SIGNATURES")
        print(f"{'='*60}")
        print("\nFolder 1 signatures:")
        for filename, (sig, _) in sorted(levels1.items()):
            print(f"  {filename}: {sig[:16]}...")
        print("\nFolder 2 signatures:")
        for filename, (sig, _) in sorted(levels2.items()):
            print(f"  {filename}: {sig[:16]}...")

    return 0


if __name__ == "__main__":
    exit(main())
