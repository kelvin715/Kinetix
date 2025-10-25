#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from typing import Any


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def is_int(x: Any) -> bool:
    # In Python, bool is subclass of int; ensure bools are not treated as ints here
    return isinstance(x, int) and not isinstance(x, bool)


def is_float(x: Any) -> bool:
    return isinstance(x, float)


def mix_numbers(a: float, b: float, t: float) -> float:
    return (1.0 - t) * float(a) + t * float(b)


def recursive_mix(a: Any, b: Any, t: float) -> Any:
    """Recursively interpolate structures between a (easy) and b (hard).

    Rules:
    - floats: linear interpolation
    - ints (including enum-like fields): choose from a if t < 0.5 else from b
    - bools, strings, None: prefer a
    - lists: elementwise mix if same length, else prefer a
    - dicts: mix by keys, prefer a for keys only present in one side
    """
    # Numbers
    if is_float(a) and (is_float(b) or is_int(b)):
        return mix_numbers(a, float(b), t)
    if is_float(b) and (is_float(a) or is_int(a)):
        return mix_numbers(float(a), b, t)
    if is_int(a) and is_int(b):
        return a if t < 0.5 else b
    # Bools/strings/None: keep a
    if is_bool(a) and is_bool(b):
        return a
    if isinstance(a, str) and isinstance(b, str):
        return a
    if a is None or b is None:
        return a

    # Lists
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return a
        return [recursive_mix(x, y, t) for x, y in zip(a, b)]

    # Dicts
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        # prefer keys from a; if also in b, mix; if only in a, keep a
        for k in a.keys():
            if k in b:
                out[k] = recursive_mix(a[k], b[k], t)
            else:
                out[k] = a[k]
        return out

    # Fallback: keep a
    return a


def add_noise(value: Any, rng: random.Random, scale: float) -> Any:
    """Add small noise to floats only. Keep ints/bools/others intact."""
    if is_bool(value) or is_int(value):
        return value
    if is_float(value):
        # multiplicative noise around 1.0
        eps = rng.normalvariate(0.0, scale)
        return value * (1.0 + eps)
    if isinstance(value, list):
        return [add_noise(v, rng, scale) for v in value]
    if isinstance(value, dict):
        return {k: add_noise(v, rng, scale) for k, v in value.items()}
    return value


def generate_levels(
    easy_path: str, hard_path: str, out_dir: str, num_levels: int, variants_per_level: int, task_name: str = "grasp"
) -> None:
    easy = load_json(easy_path)
    hard = load_json(hard_path)

    os.makedirs(out_dir, exist_ok=True)
    # Place outputs under a subfolder for clarity
    base_out = os.path.join(out_dir, f"{task_name}_levels")
    os.makedirs(base_out, exist_ok=True)

    for level_idx in range(1, num_levels + 1):
        # t in [0, 1]
        t = 0.0 if num_levels == 1 else (level_idx - 1) / (num_levels - 1)
        mixed = recursive_mix(easy, hard, t)

        level_dir = os.path.join(base_out, f"level_{level_idx}")
        os.makedirs(level_dir, exist_ok=True)

        # noise scale from 1% up to ~3%
        noise_scale = 0.01 + 0.005 * (level_idx - 1)
        for variant_idx in range(1, variants_per_level + 1):
            rng = random.Random(2025 * 100 + level_idx * 1000 + variant_idx)
            noised = add_noise(mixed, rng, noise_scale)

            fname = f"{task_name}_level{level_idx}_v{variant_idx:02d}.json"
            fpath = os.path.join(level_dir, fname)
            with open(fpath, "w") as f:
                # compact JSON similar to provided files
                json.dump(noised, f, separators=(",", ":"))
            print(f"[gen] saved: {fpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate graded env configs for grasper task.")
    parser.add_argument(
        "--easy",
        required=False,
        default="/proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/kinetix/levels/l/grasp_easy.json",
        help="Path to easy base JSON",
    )
    parser.add_argument(
        "--hard",
        required=False,
        default="/proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/kinetix/levels/l/grasp_hard.json",
        help="Path to hard base JSON",
    )
    parser.add_argument(
        "--out-dir",
        required=False,
        default="/proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/kinetix/levels/l",
        help="Directory under which to write generated levels",
    )
    parser.add_argument("--levels", type=int, default=5, help="Number of difficulty levels")
    parser.add_argument("--variants", type=int, default=10, help="Variants per level")
    parser.add_argument("--task-name", type=str, default="grasp", help="Task name prefix for files")
    args = parser.parse_args()

    generate_levels(
        easy_path=args.easy,
        hard_path=args.hard,
        out_dir=args.out_dir,
        num_levels=args.levels,
        variants_per_level=args.variants,
        task_name=args.task_name,
    )


if __name__ == "__main__":
    main()
