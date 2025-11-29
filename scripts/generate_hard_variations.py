"""Generate variations of grasp_hard.json by varying positions of blue circle and green square + podium."""

from pathlib import Path

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


def _find_blue_circle(env_state) -> int:
    """Find the blue circle (role == 2) index."""
    circ_roles = np.array(env_state.circle_shape_roles)
    circ_active = np.array(env_state.circle.active).astype(bool)
    for i, (role, active) in enumerate(zip(circ_roles, circ_active)):
        if role == 2 and active:
            return i
    raise ValueError("No active circle with role==2 (blue/agent) found.")


def _find_green_square(env_state) -> int:
    """Find the green square (polygon with role == 1) index."""
    poly_roles = np.array(env_state.polygon_shape_roles)
    poly_active = np.array(env_state.polygon.active).astype(bool)
    for i, (role, active) in enumerate(zip(poly_roles, poly_active)):
        if role == 1 and active:
            return i
    raise ValueError("No active polygon with role==1 (green/goal) found.")


def _find_podium(env_state, green_idx: int) -> int:
    """Find the podium polygon - the static shape directly below the green square.

    We identify it as the closest active polygon (role 0) that is below the green square.
    """
    green_pos = np.array(env_state.polygon.position)[green_idx]
    green_x, green_y = float(green_pos[0]), float(green_pos[1])

    poly_pos = np.array(env_state.polygon.position)
    poly_roles = np.array(env_state.polygon_shape_roles)
    poly_active = np.array(env_state.polygon.active).astype(bool)
    poly_inv_mass = np.array(env_state.polygon.inverse_mass)

    best_idx = -1
    best_dist = float("inf")

    for i in range(len(poly_pos)):
        if not poly_active[i]:
            continue
        if i == green_idx:
            continue
        # Looking for static (inverse_mass == 0) role 0 shapes
        if poly_roles[i] != 0:
            continue
        if poly_inv_mass[i] != 0:  # Not static
            continue

        px, py = float(poly_pos[i, 0]), float(poly_pos[i, 1])

        # Must be below or at same level as green square
        if py > green_y + 0.5:
            continue

        # Check x proximity (should be very close)
        dist = abs(px - green_x) + abs(py - green_y)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx == -1:
        raise ValueError("Could not find podium polygon below the green square.")

    return best_idx


def _get_shape_radius(env_state, shape_type: str, idx: int) -> float:
    """Get conservative bounding radius for a shape."""
    if shape_type == "circle":
        return float(np.array(env_state.circle.radius)[idx])
    else:
        verts = np.array(env_state.polygon.vertices)[idx]
        n = int(np.array(env_state.polygon.n_vertices)[idx])
        verts = verts[:n]
        circ = float(np.linalg.norm(verts, axis=1).max()) if n > 0 else 0.0
        poly_r = float(np.array(env_state.polygon.radius)[idx])
        return max(circ, poly_r)


def _sample_x_position(
    rng: np.random.Generator,
    base_x: float,
    shape_radius: float,
    cfg,
    static_env_params,
    env_params,
) -> float:
    """Sample a new x position within screen bounds."""
    width = float(static_env_params.screen_dim[0] / env_params.pixels_per_unit)

    x_low = shape_radius + cfg.x_margin
    x_high = max(x_low, width - shape_radius - cfg.x_margin)

    return float(rng.uniform(low=x_low, high=x_high))


def _sample_green_y_position(
    rng: np.random.Generator,
    podium_top_y: float,
    green_radius: float,
    cfg,
    static_env_params,
    env_params,
) -> float:
    """Sample a new y position for the green box above the podium."""
    height = float(static_env_params.screen_dim[1] / env_params.pixels_per_unit)

    # Green box must be at least on top of podium, up to some max height
    y_low = podium_top_y + green_radius + cfg.get("green_y_min_above_podium", 0.1)

    # Constrain y_high: min of (screen top margin, offset from y_low, hard cap at y=2.5 to stay visible)
    max_visible_y = cfg.get("green_y_max_visible", 2.5)
    y_high = min(
        height - green_radius - cfg.get("y_margin", 0.3),
        y_low + cfg.get("green_y_max_offset", 0.8),
        max_visible_y,
    )
    y_high = max(y_low, y_high)

    return float(rng.uniform(low=y_low, high=y_high))


def _apply_variant(
    env_state,
    static_env_params,
    blue_circle_idx: int,
    green_square_idx: int,
    podium_idx: int,
    new_blue_x: float,
    new_green_x: float,
    new_green_y: float = None,
):
    """Apply position changes to the blue circle and green square + podium.

    Args:
        new_green_y: If provided, sets new y position for green box only (podium y unchanged).
    """
    # Update blue circle x position (keep y the same)
    blue_pos = np.array(env_state.circle.position)[blue_circle_idx]
    new_blue_pos = jnp.array([new_blue_x, float(blue_pos[1])], dtype=jnp.float32)
    env_state = env_state.replace(
        circle=env_state.circle.replace(
            position=env_state.circle.position.at[blue_circle_idx].set(new_blue_pos),
        )
    )

    # Calculate x delta for green square and podium
    green_pos = np.array(env_state.polygon.position)[green_square_idx]
    x_delta = new_green_x - float(green_pos[0])

    # Update green square position (x always, y if provided)
    green_y = new_green_y if new_green_y is not None else float(green_pos[1])
    new_green_pos = jnp.array([new_green_x, green_y], dtype=jnp.float32)
    env_state = env_state.replace(
        polygon=env_state.polygon.replace(
            position=env_state.polygon.position.at[green_square_idx].set(new_green_pos),
        )
    )

    # Update podium x position (same delta as green square, y unchanged)
    podium_pos = np.array(env_state.polygon.position)[podium_idx]
    new_podium_pos = jnp.array([float(podium_pos[0]) + x_delta, float(podium_pos[1])], dtype=jnp.float32)
    env_state = env_state.replace(
        polygon=env_state.polygon.replace(
            position=env_state.polygon.position.at[podium_idx].set(new_podium_pos),
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


@hydra.main(version_base=None, config_path=".", config_name="hard_variations")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    src_path = get_correct_path_of_json_level(cfg.input_json)
    env_state, static_env_params, env_params = load_from_json_file(src_path)

    # Find the relevant shapes
    blue_circle_idx = _find_blue_circle(env_state)
    green_square_idx = _find_green_square(env_state)
    podium_idx = _find_podium(env_state, green_square_idx)

    print(f"Found blue circle at index: {blue_circle_idx}")
    print(f"Found green square at index: {green_square_idx}")
    print(f"Found podium at index: {podium_idx}")

    # Get base positions and radii
    blue_base_x = float(np.array(env_state.circle.position)[blue_circle_idx, 0])
    green_base_pos = np.array(env_state.polygon.position)[green_square_idx]
    green_base_x = float(green_base_pos[0])
    podium_pos = np.array(env_state.polygon.position)[podium_idx]

    blue_radius = _get_shape_radius(env_state, "circle", blue_circle_idx)
    # Use podium radius for the green+podium group x (it's wider)
    podium_radius = _get_shape_radius(env_state, "polygon", podium_idx)
    green_radius = _get_shape_radius(env_state, "polygon", green_square_idx)
    green_x_radius = max(green_radius, podium_radius)

    # Calculate podium top y (podium center + its half-height)
    podium_verts = np.array(env_state.polygon.vertices)[podium_idx]
    podium_n = int(np.array(env_state.polygon.n_vertices)[podium_idx])
    podium_half_height = float(np.abs(podium_verts[:podium_n, 1]).max()) if podium_n > 0 else 0.0
    podium_top_y = float(podium_pos[1]) + podium_half_height

    rng = np.random.default_rng(int(cfg.seed))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename stem from input
    in_stem = Path(src_path).stem
    vary_green_y = cfg.get("vary_green_y", True)
    stem = f"{in_stem}_{'xyvar' if vary_green_y else 'xvar'}"

    for i in range(int(cfg.num_variants)):
        # Sample new x positions
        new_blue_x = _sample_x_position(rng, blue_base_x, blue_radius, cfg, static_env_params, env_params)
        new_green_x = _sample_x_position(rng, green_base_x, green_x_radius, cfg, static_env_params, env_params)

        # Ensure minimum separation between blue circle and green+podium
        min_sep = cfg.get("min_separation", 1.0)
        attempts = 0
        while abs(new_blue_x - new_green_x) < min_sep + blue_radius + green_x_radius and attempts < 100:
            new_green_x = _sample_x_position(rng, green_base_x, green_x_radius, cfg, static_env_params, env_params)
            attempts += 1

        # Sample green y position if enabled
        new_green_y = None
        if vary_green_y:
            new_green_y = _sample_green_y_position(rng, podium_top_y, green_radius, cfg, static_env_params, env_params)

        variant_state = _apply_variant(
            env_state,
            static_env_params,
            blue_circle_idx,
            green_square_idx,
            podium_idx,
            new_blue_x,
            new_green_x,
            new_green_y,
        )

        filename = _compose_filename(stem, i)
        out_path = str(out_dir / filename)
        export_env_state_to_json(out_path, variant_state, static_env_params, env_params)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
