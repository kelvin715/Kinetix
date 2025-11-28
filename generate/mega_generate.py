from pathlib import Path
from typing import List, Optional, Tuple

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


def _select_shape_by_role(
    env_state,
    role_id: int,
    prefer_polygon: bool = True,
) -> Tuple[str, int]:
    """Return (shape_type, index) with a matching role_id. Prefer given type if available."""
    poly_roles = np.asarray(env_state.polygon_shape_roles)
    circ_roles = np.asarray(env_state.circle_shape_roles)
    poly_active = np.asarray(env_state.polygon.active).astype(bool)
    circ_active = np.asarray(env_state.circle.active).astype(bool)

    poly_idxs = [i for i in np.where(poly_roles == role_id)[0] if poly_active[i]]
    circ_idxs = [i for i in np.where(circ_roles == role_id)[0] if circ_active[i]]

    if prefer_polygon:
        if len(poly_idxs) > 0:
            return "polygon", int(poly_idxs[0])
        if len(circ_idxs) > 0:
            return "circle", int(circ_idxs[0])
    else:
        if len(circ_idxs) > 0:
            return "circle", int(circ_idxs[0])
        if len(poly_idxs) > 0:
            return "polygon", int(poly_idxs[0])
    raise ValueError(f"No active shape found with role == {role_id}.")


def _polygon_radius(env_state, idx: int) -> float:
    try:
        verts = np.array(env_state.polygon.vertices)[idx]
        n = int(np.array(env_state.polygon.n_vertices)[idx])
        verts = verts[:n]
        circ = float(np.linalg.norm(verts, axis=1).max()) if n > 0 else 0.0
        poly_r = float(np.array(env_state.polygon.radius)[idx])
        return max(circ, poly_r)
    except Exception:
        return 0.0


def _poly_world_aabb(env_state, idx: int) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned bounding box of polygon j in world coordinates."""
    try:
        pos = np.array(env_state.polygon.position)[idx]
        rot = float(np.array(env_state.polygon.rotation)[idx])
        verts = np.array(env_state.polygon.vertices)[idx]
        n = int(np.array(env_state.polygon.n_vertices)[idx])
        verts = verts[:n]
        if n == 0:
            return None
    except Exception:
        return None
    c = np.cos(rot)
    s = np.sin(rot)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    world_verts = verts @ R.T + pos[None, :]
    xmin = float(world_verts[:, 0].min())
    xmax = float(world_verts[:, 0].max())
    ymin = float(world_verts[:, 1].min())
    ymax = float(world_verts[:, 1].max())
    return (xmin, xmax, ymin, ymax)


def _poly_world_aabb_with_rotation(env_state, idx: int, new_rot: float) -> Optional[Tuple[float, float, float, float]]:
    """AABB if we rotate polygon idx to new_rot at its current position."""
    try:
        pos = np.array(env_state.polygon.position)[idx]
        verts = np.array(env_state.polygon.vertices)[idx]
        n = int(np.array(env_state.polygon.n_vertices)[idx])
        verts = verts[:n]
        if n == 0:
            return None
    except Exception:
        return None
    c = np.cos(new_rot)
    s = np.sin(new_rot)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    world_verts = verts @ R.T + pos[None, :]
    xmin = float(world_verts[:, 0].min())
    xmax = float(world_verts[:, 0].max())
    ymin = float(world_verts[:, 1].min())
    ymax = float(world_verts[:, 1].max())
    return (xmin, xmax, ymin, ymax)


def _collect_other_aabbs(
    env_state,
    static_env_params,
    exclude: List[Tuple[str, int]],
    ignore_static_fixated_polys: bool = False,
) -> np.ndarray:
    """Collect AABBs for all active shapes except those in exclude list.
    Optionally ignore the first N static-fixated polygons (e.g., floor/walls) to avoid AABB false positives.
    """
    excluded = {(t, int(i)) for t, i in exclude}
    other_aabbs = []
    # Polygons
    poly_active = np.array(env_state.polygon.active).astype(bool)
    n_polys = poly_active.shape[0] if poly_active.ndim > 0 else 0
    for j in range(n_polys):
        if not poly_active[j]:
            continue
        if (
            ignore_static_fixated_polys
            and static_env_params is not None
            and j < int(getattr(static_env_params, "num_static_fixated_polys", 0))
        ):
            # Skip static/fixated polygons (like floor/walls)
            continue
        if ("polygon", j) in excluded:
            continue
        aabb = _poly_world_aabb(env_state, j)
        if aabb is not None:
            other_aabbs.append(aabb)
    # Circles
    circ_active = np.array(env_state.circle.active).astype(bool)
    circ_pos = np.array(env_state.circle.position)
    circ_radius = np.array(env_state.circle.radius)
    n_circs = circ_active.shape[0] if circ_active.ndim > 0 else 0
    for j in range(n_circs):
        if not circ_active[j]:
            continue
        if ("circle", j) in excluded:
            continue
        rj = float(circ_radius[j])
        cx = float(circ_pos[j, 0])
        cy = float(circ_pos[j, 1])
        other_aabbs.append((cx - rj, cx + rj, cy - rj, cy + rj))
    return np.array(other_aabbs, dtype=np.float32) if len(other_aabbs) > 0 else np.zeros((0, 4), dtype=np.float32)


def _sample_x_only_for_shape(
    rng: np.random.Generator,
    env_state,
    static_env_params,
    env_params,
    target_type: str,
    target_idx: int,
    x_min: Optional[float],
    x_max: Optional[float],
    max_tries: int,
    eps: float,
) -> float:
    """Sample a new x position for the target shape (y is fixed), ensuring no overlap."""
    width = float(static_env_params.screen_dim[0] / env_params.pixels_per_unit)

    # Compute conservative radius for target
    if target_type == "circle":
        target_r = float(np.array(env_state.circle.radius)[target_idx])
        y_fixed = float(np.array(env_state.circle.position)[target_idx, 1])
    else:
        target_r = _polygon_radius(env_state, target_idx)
        y_fixed = float(np.array(env_state.polygon.position)[target_idx, 1])

    # Bounds
    if x_min is None or x_max is None:
        x_low = target_r
        x_high = max(x_low, width - target_r)
    else:
        x_low = float(x_min)
        x_high = float(x_max)

    # Ignore static fixated polygons to avoid AABB false positives with floor/walls
    other_aabbs = _collect_other_aabbs(
        env_state,
        static_env_params,
        exclude=[(target_type, target_idx)],
        ignore_static_fixated_polys=True,
    )

    for _ in range(int(max_tries)):
        x = float(rng.uniform(low=x_low, high=x_high))
        # target AABB as circle around (x, y_fixed)
        txmin, txmax = x - target_r, x + target_r
        tymin, tymax = y_fixed - target_r, y_fixed + target_r
        if other_aabbs.shape[0] == 0:
            return x
        oxmin = other_aabbs[:, 0]
        oxmax = other_aabbs[:, 1]
        oymin = other_aabbs[:, 2]
        oymax = other_aabbs[:, 3]
        overlap_x = np.logical_not((txmax < oxmin - eps) | (txmin > oxmax + eps))
        overlap_y = np.logical_not((tymax < oymin - eps) | (tymin > oymax + eps))
        if not np.any(overlap_x & overlap_y):
            return x
    # Fallback: keep base x
    if target_type == "circle":
        return float(np.array(env_state.circle.position)[target_idx, 0])
    return float(np.array(env_state.polygon.position)[target_idx, 0])


def _sample_green_rotation(
    rng: np.random.Generator,
    env_state,
    target_idx: int,
    rot_min: float,
    rot_max: float,
    max_tries: int,
    eps: float,
) -> float:
    """Sample green rotation (polygon) ensuring no overlap after rotation."""
    # For rotation, keep all shapes in the check to avoid interpenetration with dynamic parts
    other_aabbs = _collect_other_aabbs(
        env_state,
        static_env_params=None,
        exclude=[("polygon", target_idx)],
        ignore_static_fixated_polys=False,
    )
    for _ in range(int(max_tries)):
        new_rot = float(rng.uniform(low=rot_min, high=rot_max))
        aabb = _poly_world_aabb_with_rotation(env_state, target_idx, new_rot)
        if aabb is None:
            continue
        txmin, txmax, tymin, tymax = aabb
        if other_aabbs.shape[0] == 0:
            return new_rot
        oxmin = other_aabbs[:, 0]
        oxmax = other_aabbs[:, 1]
        oymin = other_aabbs[:, 2]
        oymax = other_aabbs[:, 3]
        overlap_x = np.logical_not((txmax < oxmin - eps) | (txmin > oxmax + eps))
        overlap_y = np.logical_not((tymax < oymin - eps) | (tymin > oymax + eps))
        if not np.any(overlap_x & overlap_y):
            return new_rot
    # Fallback: keep base rotation
    return float(np.array(env_state.polygon.rotation)[target_idx])


def _sample_green_position_xy(
    rng: np.random.Generator,
    env_state,
    static_env_params,
    env_params,
    target_idx: int,
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
    max_tries: int,
    eps: float,
) -> Tuple[float, float]:
    """Sample (x, y) for green polygon ensuring no overlap."""
    width = float(static_env_params.screen_dim[0] / env_params.pixels_per_unit)
    height = float(static_env_params.screen_dim[1] / env_params.pixels_per_unit)
    target_r = _polygon_radius(env_state, target_idx)

    # Bounds default: keep polygon's bounding circle inside screen
    if x_min is None or x_max is None:
        x_low = target_r
        x_high = max(x_low, width - target_r)
    else:
        x_low = float(x_min)
        x_high = float(x_max)
    if y_min is None or y_max is None:
        y_low = target_r
        y_high = max(y_low, height - target_r)
    else:
        y_low = float(y_min)
        y_high = float(y_max)

    other_aabbs = _collect_other_aabbs(
        env_state,
        static_env_params,
        exclude=[("polygon", target_idx)],
        ignore_static_fixated_polys=False,
    )
    base_xy = np.array(env_state.polygon.position)[target_idx]

    for _ in range(int(max_tries)):
        x = float(rng.uniform(low=x_low, high=x_high))
        y = float(rng.uniform(low=y_low, high=y_high))
        # Use circle AABB at proposed center as conservative bound
        txmin, txmax = x - target_r, x + target_r
        tymin, tymax = y - target_r, y + target_r
        if other_aabbs.shape[0] == 0:
            return x, y
        oxmin = other_aabbs[:, 0]
        oxmax = other_aabbs[:, 1]
        oymin = other_aabbs[:, 2]
        oymax = other_aabbs[:, 3]
        overlap_x = np.logical_not((txmax < oxmin - eps) | (txmin > oxmax + eps))
        overlap_y = np.logical_not((tymax < oymin - eps) | (tymin > oymax + eps))
        if not np.any(overlap_x & overlap_y):
            return x, y
    return float(base_xy[0]), float(base_xy[1])


def _apply_shape_updates(
    env_state,
    static_env_params,
    updates: List[Tuple[str, int, Optional[float], Optional[float]]],
):
    """Apply position.x or rotation updates to shapes.
    updates: list of (type, idx, new_x or None, new_rot or None)
    """
    for t, idx, new_x, new_rot in updates:
        if t == "polygon":
            if new_x is not None:
                cur = np.array(env_state.polygon.position)
                cur[idx, 0] = new_x
                env_state = env_state.replace(
                    polygon=env_state.polygon.replace(position=jnp.array(cur, dtype=jnp.float32))
                )
            if new_rot is not None:
                cur = np.array(env_state.polygon.rotation)
                cur[idx] = new_rot
                env_state = env_state.replace(
                    polygon=env_state.polygon.replace(rotation=jnp.array(cur, dtype=jnp.float32))
                )
        else:
            if new_x is not None:
                cur = np.array(env_state.circle.position)
                cur[idx, 0] = new_x
                env_state = env_state.replace(
                    circle=env_state.circle.replace(position=jnp.array(cur, dtype=jnp.float32))
                )
            if new_rot is not None:
                cur = np.array(env_state.circle.rotation)
                cur[idx] = new_rot
                env_state = env_state.replace(
                    circle=env_state.circle.replace(rotation=jnp.array(cur, dtype=jnp.float32))
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


def _apply_joint_angle_deltas_to_bodies(env_state, static_env_params, deltas_by_joint_idx: List[Tuple[int, float]]):
    """Approximate FK with subtree propagation to make pose changes visible.
    Rotate the joint's B body and all descendants (following a_index -> b_index edges)
    around the parent's joint global position by the same delta. Update both polygon and
    circle positions/rotations.
    """
    num_polys = int(static_env_params.num_polygons)
    joint = env_state.joint
    poly_pos = np.array(env_state.polygon.position)
    poly_rot = np.array(env_state.polygon.rotation)
    poly_active = np.array(env_state.polygon.active).astype(bool)
    circ_pos = np.array(env_state.circle.position)
    circ_rot = np.array(env_state.circle.rotation)
    circ_active = np.array(env_state.circle.active).astype(bool)

    a_index = np.array(joint.a_index)
    b_index = np.array(joint.b_index)

    # Build adjacency: parent body id -> list of child joint indices
    parent_to_child_joints: dict[int, list[int]] = {}
    for j_idx_iter in range(len(a_index)):
        parent_to_child_joints.setdefault(int(a_index[j_idx_iter]), []).append(j_idx_iter)

    def rotate_body_around_pivot(global_body_id: int, pivot: np.ndarray, delta: float):
        nonlocal poly_pos, poly_rot, circ_pos, circ_rot
        if global_body_id < num_polys:
            idx = global_body_id
            if not poly_active[idx]:
                return
            cur_pos = poly_pos[idx]
            v = cur_pos - pivot
            c = np.cos(delta)
            s = np.sin(delta)
            v_rot = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)
            new_pos = pivot + v_rot
            poly_pos[idx] = new_pos
            poly_rot[idx] = float(poly_rot[idx]) + float(delta)
        else:
            idx = global_body_id - num_polys
            if not circ_active[idx]:
                return
            cur_pos = circ_pos[idx]
            v = cur_pos - pivot
            c = np.cos(delta)
            s = np.sin(delta)
            v_rot = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)
            new_pos = pivot + v_rot
            circ_pos[idx] = new_pos
            circ_rot[idx] = float(circ_rot[idx]) + float(delta)

    for j_idx, delta in deltas_by_joint_idx:
        pivot = np.array(joint.global_position)[j_idx]
        root_body = int(b_index[j_idx])
        # BFS over subtree starting at root_body
        queue = [root_body]
        visited: set[int] = set()
        while queue:
            body_id = queue.pop(0)
            if body_id in visited:
                continue
            visited.add(body_id)
            rotate_body_around_pivot(body_id, pivot, delta)
            for cj in parent_to_child_joints.get(body_id, []):
                queue.append(int(b_index[cj]))

    env_state = env_state.replace(
        polygon=env_state.polygon.replace(
            position=jnp.array(poly_pos, dtype=jnp.float32),
            rotation=jnp.array(poly_rot, dtype=jnp.float32),
        ),
        circle=env_state.circle.replace(
            position=jnp.array(circ_pos, dtype=jnp.float32),
            rotation=jnp.array(circ_rot, dtype=jnp.float32),
        ),
    )
    # Reset collision buffers
    acc_rr, acc_cr, acc_cc = get_empty_collision_manifolds(static_env_params)
    env_state = env_state.replace(
        acc_rr_manifolds=acc_rr,
        acc_cr_manifolds=acc_cr,
        acc_cc_manifolds=acc_cc,
        collision_matrix=calculate_collision_matrix(static_env_params, env_state.joint),
    )
    return env_state


def _vary_joint_angles(rng: np.random.Generator, env_state, static_env_params, cfg):
    """Vary joint target angles and apply deltas to bodies so the pose is visible in the initial state."""
    if not cfg.vary.joints:
        return env_state
    mode = str(cfg.joints.mode)
    r0, r1 = float(cfg.joints.range[0]), float(cfg.joints.range[1])
    respect_limits = bool(cfg.joints.respect_limits)

    cur_rot = np.array(env_state.joint.rotation, dtype=np.float32)
    min_rot = np.array(env_state.joint.min_rotation, dtype=np.float32)
    max_rot = np.array(env_state.joint.max_rotation, dtype=np.float32)

    n = cur_rot.shape[0] if cur_rot.ndim > 0 else 0
    if cfg.joints.selection == "indices" and len(cfg.joints.indices) > 0:
        target_idxs = [int(i) for i in list(cfg.joints.indices)]
    else:
        target_idxs = list(range(n))

    deltas: List[Tuple[int, float]] = []
    for j in target_idxs:
        base = float(cur_rot[j])
        if mode == "absolute":
            new_val = float(rng.uniform(low=r0, high=r1))
        else:
            delta = float(rng.uniform(low=r0, high=r1))
            new_val = base + delta
        if respect_limits:
            lo = float(min_rot[j])
            hi = float(max_rot[j])
            # If limit bounds are equal (no limits), skip clamping
            if not (np.isclose(lo, hi)):
                new_val = float(np.clip(new_val, lo, hi))
        cur_rot[j] = new_val
        delta_applied = new_val - base
        deltas.append((j, delta_applied))

    env_state = env_state.replace(joint=env_state.joint.replace(rotation=jnp.array(cur_rot, dtype=jnp.float32)))
    # Apply deltas to connected bodies so they visibly move
    env_state = _apply_joint_angle_deltas_to_bodies(env_state, static_env_params, deltas)
    return env_state


def _compose_filename(stem: str, i: int) -> str:
    return f"{stem}_v{str(i).zfill(4)}.json"


@hydra.main(version_base=None, config_path=".", config_name="mega_level_variations")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    src_path = get_correct_path_of_json_level(cfg.input_json)
    env_state, static_env_params, env_params = load_from_json_file(src_path)

    rng = np.random.default_rng(int(cfg.seed))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve shapes by role
    prefer_polygon = bool(cfg.roles.prefer_polygon)
    green_t, green_i = _select_shape_by_role(env_state, int(cfg.roles.green_role), prefer_polygon)
    blue_t, blue_i = _select_shape_by_role(env_state, int(cfg.roles.blue_role), prefer_polygon)
    grey_t, grey_i = _select_shape_by_role(env_state, int(cfg.roles.grey_role), prefer_polygon)

    # Base values
    base_blue_x = float(np.array(getattr(env_state, blue_t).position)[blue_i, 0])
    base_grey_x = float(np.array(getattr(env_state, grey_t).position)[grey_i, 0])
    if green_t != "polygon":
        raise ValueError("Green object expected to be a polygon for rotation sampling.")
    base_green_rot = float(np.array(env_state.polygon.rotation)[green_i])

    # Filename stem
    in_stem = Path(src_path).stem
    suffixes = []
    if cfg.vary.joints:
        suffixes.append("joints")
    if cfg.vary.blue_x:
        suffixes.append("bluex")
    if cfg.vary.grey_x:
        suffixes.append("greyx")
    if cfg.vary.green_rot:
        suffixes.append("greenrot")
    stem = f"{in_stem}_{'-'.join(suffixes) if suffixes else 'copy'}"

    # Sampling params
    eps = float(cfg.sampling.eps)
    do_fallback = bool(cfg.sampling.fallback_to_base)

    blue_x_min = None if cfg.blue_x.min is None else float(cfg.blue_x.min)
    blue_x_max = None if cfg.blue_x.max is None else float(cfg.blue_x.max)
    blue_max_tries = int(cfg.blue_x.max_tries)

    grey_x_min = None if cfg.grey_x.min is None else float(cfg.grey_x.min)
    grey_x_max = None if cfg.grey_x.max is None else float(cfg.grey_x.max)
    grey_max_tries = int(cfg.grey_x.max_tries)

    grot_min, grot_max = float(cfg.green_rot.range[0]), float(cfg.green_rot.range[1])
    grot_max_tries = int(cfg.green_rot.max_tries)

    gpos_xmin = None if cfg.green_pos.x_min is None else float(cfg.green_pos.x_min)
    gpos_xmax = None if cfg.green_pos.x_max is None else float(cfg.green_pos.x_max)
    gpos_ymin = None if cfg.green_pos.y_min is None else float(cfg.green_pos.y_min)
    gpos_ymax = None if cfg.green_pos.y_max is None else float(cfg.green_pos.y_max)
    gpos_max_tries = int(cfg.green_pos.max_tries)

    for i in range(int(cfg.num_variants)):
        cur_state = env_state
        updates: List[Tuple[str, int, Optional[float], Optional[float]]] = []

        # 1) Joints variation (does not affect overlap of bodies directly)
        if cfg.vary.joints:
            cur_state = _vary_joint_angles(rng, cur_state, static_env_params, cfg)

        # 2) Green position first (affects non-overlap for others)
        if getattr(cfg.vary, "green_pos", False):
            new_gx, new_gy = _sample_green_position_xy(
                rng=rng,
                env_state=cur_state,
                static_env_params=static_env_params,
                env_params=env_params,
                target_idx=green_i,
                x_min=gpos_xmin,
                x_max=gpos_xmax,
                y_min=gpos_ymin,
                y_max=gpos_ymax,
                max_tries=gpos_max_tries,
                eps=eps,
            )
            updates.append(("polygon", green_i, None, None))  # placeholder to keep order
            # apply position immediately
            cur_state = cur_state.replace(  # type: ignore[attr-defined]
                polygon=cur_state.polygon.replace(  # type: ignore[attr-defined]
                    position=cur_state.polygon.position.at[green_i].set(jnp.array([new_gx, new_gy], dtype=jnp.float32))
                )
            )
            # reset collisions after move
            acc_rr, acc_cr, acc_cc = get_empty_collision_manifolds(static_env_params)
            cur_state = cur_state.replace(
                acc_rr_manifolds=acc_rr,
                acc_cr_manifolds=acc_cr,
                acc_cc_manifolds=acc_cc,
                collision_matrix=calculate_collision_matrix(static_env_params, cur_state.joint),
            )

        # 3) Blue x-only
        if cfg.vary.blue_x:
            new_blue_x = _sample_x_only_for_shape(
                rng=rng,
                env_state=cur_state,
                static_env_params=static_env_params,
                env_params=env_params,
                target_type=blue_t,
                target_idx=blue_i,
                x_min=blue_x_min,
                x_max=blue_x_max,
                max_tries=blue_max_tries,
                eps=eps,
            )
            if not do_fallback and np.isclose(new_blue_x, base_blue_x):
                # If not allowed to fallback and sampling failed, skip saving this variant
                pass
            updates.append((blue_t, blue_i, new_blue_x, None))
            cur_state = _apply_shape_updates(cur_state, static_env_params, [updates[-1]])

        # 4) Grey x-only; consider updated 'cur_state' so we avoid overlapping updated blue
        if cfg.vary.grey_x:
            new_grey_x = _sample_x_only_for_shape(
                rng=rng,
                env_state=cur_state,
                static_env_params=static_env_params,
                env_params=env_params,
                target_type=grey_t,
                target_idx=grey_i,
                x_min=grey_x_min,
                x_max=grey_x_max,
                max_tries=grey_max_tries,
                eps=eps,
            )
            if not do_fallback and np.isclose(new_grey_x, base_grey_x):
                pass
            updates.append((grey_t, grey_i, new_grey_x, None))
            cur_state = _apply_shape_updates(cur_state, static_env_params, [updates[-1]])

        # 5) Green rotation with overlap rejection
        if cfg.vary.green_rot:
            new_green_rot = _sample_green_rotation(
                rng=rng,
                env_state=cur_state,
                target_idx=green_i,
                rot_min=grot_min,
                rot_max=grot_max,
                max_tries=grot_max_tries,
                eps=eps,
            )
            if not do_fallback and np.isclose(new_green_rot, base_green_rot):
                pass
            updates.append(("polygon", green_i, None, new_green_rot))
            cur_state = _apply_shape_updates(cur_state, static_env_params, [updates[-1]])

        # Save
        out_path = str(out_dir / _compose_filename(stem, i))
        export_env_state_to_json(out_path, cur_state, static_env_params, env_params)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
