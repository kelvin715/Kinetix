import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# from kinetix.render import make_render_pixels
# from kinetix.util import load_from_json_file
from kinetix.environment import (  # noqa
    ActionType,
    ObservationType,
    make_kinetix_env,  # noqa
)
from kinetix.render import make_render_pixels
from kinetix.util import load_from_json_file


def render_json_to_png(json_path: Path, png_path: Path, downscale_override: int | None = None) -> None:
    env_state, static_env_params, env_params = load_from_json_file(str(json_path))
    if downscale_override is not None:
        static_env_params = static_env_params.replace(downscale=int(downscale_override))

    renderer = make_render_pixels(env_params, static_env_params)
    pixels = renderer(env_state)
    img = np.asarray(pixels).astype(np.uint8).transpose(1, 0, 2)[::-1]
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(png_path), img)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render all JSON variants to PNGs.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/level_variants",
        help="Directory containing JSON level variants.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/level_variants/images",
        help="Directory to save rendered PNGs.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern to match JSON files.",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=None,
        help="Optional override for static_env_params.downscale during rendering.",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = (repo_root / in_dir).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()

    json_files = sorted(in_dir.glob(args.pattern))
    if not json_files:
        print(f"No JSON files found in {in_dir} matching {args.pattern}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    for jp in json_files:
        png_name = jp.stem + ".png"
        png_path = out_dir / png_name
        try:
            render_json_to_png(jp, png_path, args.downscale)
            print(f"Saved {png_path}")
        except Exception as e:
            print(f"Failed to render {jp}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
