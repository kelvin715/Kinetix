from pathlib import Path
from typing import Optional, Tuple

import hydra
import jax.numpy as jnp
import numpy as np
from jax2d.engine import calculate_collision_matrix, get_empty_collision_manifolds
from omegaconf import DictConfig, OmegaConf

from kinetix.util.saving import (
    export_env_state_to_json,
    get_correct_path_of_json_level,
    load_from_json_file,
)


def _find_role_one_shape(env_state) -> Tuple[str, int]:
    """Return (shape_type, index) where shape_type in {"polygon","circle"} for role == 1.
    Prefer polygon if both exist; otherwise circle. Raises if not found.
    """
    poly_idxs = np.where(np.asarray(env_state.polygon_shape_roles) == 1)[0]
    poly_idxs = [i for i in poly_idxs if bool(env_state.polygon.active[i])]
    if len(poly_idxs) > 0:
        return "polygon", int(poly_idxs[0])
    circ_idxs = np.where(np.asarray(env_state.circle_shape_roles) == 1)[0]
    circ_idxs = [i for i in circ_idxs if bool(env_state.circle.active[i])]
    if len(circ_idxs) > 0:
        return "circle", int(circ_idxs[0])
    raise ValueError("No active shape with role==1 (ball/green) found in level.")


def _select_target_shape(env_state, shape_type: str, index: Optional[int]) -> Tuple[str, int]:
    if shape_type == "auto" or shape_type is None:
        return _find_role_one_shape(env_state)
    if shape_type not in ("polygon", "circle"):
        raise ValueError("override.shape_type must be one of {'auto','polygon','circle'}")
    if index is None:
        # If type provided but index not given, try to auto-pick role 1 of that type
        if shape_type == "polygon":
            idxs = np.where(np.asarray(env_state.polygon_shape_roles) == 1)[0]
            idxs = [i for i in idxs if bool(env_state.polygon.active[i])]
        else:
            idxs = np.where(np.asarray(env_state.circle_shape_roles) == 1)[0]
            idxs = [i for i in idxs if bool(env_state.circle.active[i])]
        if len(idxs) == 0:
            raise ValueError(f"No active role==1 shape found for type {shape_type}.")
        return shape_type, int(idxs[0])
    return shape_type, int(index)


def _sample_position(rng: np.random.Generator, base_xy: np.ndarray, cfg) -> np.ndarray:
    rx = cfg.position.range_x
    ry = cfg.position.range_y
    dx = rng.uniform(low=rx[0], high=rx[1]) if cfg.vary.position else 0.0
    dy = rng.uniform(low=ry[0], high=ry[1]) if cfg.vary.position else 0.0
    return base_xy + np.array([dx, dy], dtype=np.float32)


def _sample_rotation(rng: np.random.Generator, base_rot: float, cfg) -> float:
    if not cfg.vary.rotation:
        return base_rot
    rmin, rmax = cfg.rotation.range
    return float(rng.uniform(low=rmin, high=rmax))


def _apply_variant(
    env_state,
    static_env_params,
    shape_type: str,
    shape_idx: int,
    new_xy: np.ndarray,
    new_rot: float,
):
    if shape_type == "polygon":
        env_state = env_state.replace(
            polygon=env_state.polygon.replace(
                position=env_state.polygon.position.at[shape_idx].set(jnp.array(new_xy, dtype=jnp.float32)),
                rotation=env_state.polygon.rotation.at[shape_idx].set(jnp.array(new_rot, dtype=jnp.float32)),
            )
        )
    else:
        env_state = env_state.replace(
            circle=env_state.circle.replace(
                position=env_state.circle.position.at[shape_idx].set(jnp.array(new_xy, dtype=jnp.float32)),
                rotation=env_state.circle.rotation.at[shape_idx].set(jnp.array(new_rot, dtype=jnp.float32)),
            )
        )

    # Reset collision-related buffers and recalc collision matrix
    acc_rr, acc_cr, acc_cc = get_empty_collision_manifolds(static_env_params)
    env_state = env_state.replace(
        acc_rr_manifolds=acc_rr,
        acc_cr_manifolds=acc_cr,
        acc_cc_manifolds=acc_cc,
        collision_matrix=calculate_collision_matrix(static_env_params, env_state.joint),
    )
    return env_state


def _compose_filename(stem: str, i: int) -> str:
    return f"{stem}_v{str(i).zfill(4)}.json"


@hydra.main(version_base=None, config_path=".", config_name="level_variations")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if not (cfg.vary.position or cfg.vary.rotation):
        raise ValueError("At least one of vary.position or vary.rotation must be true.")

    src_path = get_correct_path_of_json_level(cfg.input_json)
    env_state, static_env_params, env_params = load_from_json_file(src_path)

    shape_type, shape_idx = _select_target_shape(env_state, cfg.override.shape_type, cfg.override.index)

    # Base values
    if shape_type == "polygon":
        base_xy = np.array(env_state.polygon.position)
        base_xy = np.array(base_xy[shape_idx])
        base_rot = float(np.array(env_state.polygon.rotation)[shape_idx])
    else:
        base_xy = np.array(env_state.circle.position)
        base_xy = np.array(base_xy[shape_idx])
        base_rot = float(np.array(env_state.circle.rotation)[shape_idx])

    # Defaults: if user wants deltas around current pos, cfg ranges can be small
    rng = np.random.default_rng(int(cfg.seed))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename stem from input
    in_stem = Path(src_path).stem
    stem_suffix = []
    if cfg.vary.position:
        stem_suffix.append("pos")
    if cfg.vary.rotation:
        stem_suffix.append("rot")
    stem = f"{in_stem}_{'-'.join(stem_suffix) if stem_suffix else 'copy'}"

    for i in range(int(cfg.num_variants)):
        xy = _sample_position(rng, base_xy, cfg)
        rot = _sample_rotation(rng, base_rot, cfg)

        variant_state = _apply_variant(env_state, static_env_params, shape_type, shape_idx, xy, rot)

        filename = _compose_filename(stem, i)
        out_path = str(out_dir / filename)
        export_env_state_to_json(out_path, variant_state, static_env_params, env_params)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
