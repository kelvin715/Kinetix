#!/usr/bin/env python3
import argparse
import os
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from kinetix.environment import ActionType, ObservationType, make_kinetix_env
from kinetix.render import make_render_pixels
from kinetix.util.saving import load_from_json_file


def list_json_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def render_level(json_path: str, out_png: str) -> None:
    env_state, static_env_params, env_params = load_from_json_file(json_path)

    env = make_kinetix_env(
        action_type=ActionType.CONTINUOUS,
        observation_type=ObservationType.PIXELS,
        reset_fn=lambda rng: env_state,
        env_params=env_params,
        static_env_params=static_env_params,
    )

    renderer = make_render_pixels(env_params, static_env_params)
    pixels = renderer(env_state)
    # Match example: transpose and vertical flip for correct orientation
    img = np.array(pixels).astype(np.uint8)
    img = img.transpose(1, 0, 2)[::-1]

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.imsave(out_png, img)


def main():
    parser = argparse.ArgumentParser(description="Batch render PNG previews for grasp levels")
    parser.add_argument(
        "--in-dir",
        type=str,
        default="/proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/kinetix/levels/l/grasp_levels",
        help="Directory containing JSON level files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/proj-vertical-llms-pvc/users/zhihan/robots_proj/Kinetix/kinetix/levels/l/grasp_levels_previews",
        help="Directory to write PNG previews",
    )
    args = parser.parse_args()

    json_files = list_json_files(args.in_dir)
    if not json_files:
        print(f"No JSON files found under {args.in_dir}")
        return

    for jp in json_files:
        rel = os.path.relpath(jp, args.in_dir)
        out_png = os.path.join(args.out_dir, os.path.splitext(rel)[0] + ".png")
        print(f"[render] {jp} -> {out_png}")
        render_level(jp, out_png)


if __name__ == "__main__":
    main()
