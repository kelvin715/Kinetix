"""Generate variations of grasp_hard.json by varying arm joint angles and positions."""

from pathlib import Path
from typing import List, Tuple

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


# Arm joint indices in the grasp_hard level
# Joint 4: connects polygon 4 to polygon 5 (arm base to arm segment)
# Joint 5: connects polygon 5 to circle 0 (arm segment to end ball)
ARM_JOINT_INDICES = [4, 5]

# Hand joint indices (we don't vary these)
HAND_JOINT_INDICES = [0, 1, 2, 3]


# ============================================================================
# Position variation helpers (from generate_hard_variations.py)
# ============================================================================


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
    """Find the podium polygon - the static shape directly below the green square."""
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
        if poly_roles[i] != 0:
            continue
        if poly_inv_mass[i] != 0:
            continue

        px, py = float(poly_pos[i, 0]), float(poly_pos[i, 1])
        if py > green_y + 0.5:
            continue

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
    shape_radius: float,
    cfg,
    static_env_params,
    env_params,
) -> float:
    """Sample a new x position within screen bounds."""
    width = float(static_env_params.screen_dim[0] / env_params.pixels_per_unit)
    x_margin = cfg.get("x_margin", 0.3)
    x_low = shape_radius + x_margin
    x_high = max(x_low, width - shape_radius - x_margin)
    return float(rng.uniform(low=x_low, high=x_high))


def _apply_position_changes(
    env_state,
    blue_circle_idx: int,
    green_square_idx: int,
    podium_idx: int,
    new_blue_x: float,
    new_green_podium_x: float,
):
    """Apply x position changes to blue circle and green square + podium."""
    # Update blue circle x position (keep y the same)
    blue_pos = np.array(env_state.circle.position)[blue_circle_idx]
    new_blue_pos = jnp.array([new_blue_x, float(blue_pos[1])], dtype=jnp.float32)
    env_state = env_state.replace(
        circle=env_state.circle.replace(
            position=env_state.circle.position.at[blue_circle_idx].set(new_blue_pos),
        )
    )

    # Update green square x position (keep y the same - stays on podium)
    green_pos = np.array(env_state.polygon.position)[green_square_idx]
    new_green_pos = jnp.array([new_green_podium_x, float(green_pos[1])], dtype=jnp.float32)
    env_state = env_state.replace(
        polygon=env_state.polygon.replace(
            position=env_state.polygon.position.at[green_square_idx].set(new_green_pos),
        )
    )

    # Update podium x position (keep y the same)
    podium_pos = np.array(env_state.polygon.position)[podium_idx]
    new_podium_pos = jnp.array([new_green_podium_x, float(podium_pos[1])], dtype=jnp.float32)
    env_state = env_state.replace(
        polygon=env_state.polygon.replace(
            position=env_state.polygon.position.at[podium_idx].set(new_podium_pos),
        )
    )

    return env_state


# ============================================================================
# Joint variation helpers
# ============================================================================


def _rotate_point(point: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D point by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ point


def _get_joint_info(env_state, joint_idx: int) -> dict:
    """Extract joint information from env_state."""
    joint = env_state.joint
    return {
        "a_index": int(np.array(joint.a_index)[joint_idx]),
        "b_index": int(np.array(joint.b_index)[joint_idx]),
        "a_relative_pos": np.array(
            [
                float(np.array(joint.a_relative_pos)[joint_idx, 0]),
                float(np.array(joint.a_relative_pos)[joint_idx, 1]),
            ]
        ),
        "b_relative_pos": np.array(
            [
                float(np.array(joint.b_relative_pos)[joint_idx, 0]),
                float(np.array(joint.b_relative_pos)[joint_idx, 1]),
            ]
        ),
        "rotation": float(np.array(joint.rotation)[joint_idx]),
        "global_position": np.array(
            [
                float(np.array(joint.global_position)[joint_idx, 0]),
                float(np.array(joint.global_position)[joint_idx, 1]),
            ]
        ),
    }


def _get_polygon_pose(env_state, poly_idx: int) -> Tuple[np.ndarray, float]:
    """Get position and rotation of a polygon."""
    pos = np.array(
        [
            float(np.array(env_state.polygon.position)[poly_idx, 0]),
            float(np.array(env_state.polygon.position)[poly_idx, 1]),
        ]
    )
    rot = float(np.array(env_state.polygon.rotation)[poly_idx])
    return pos, rot


def _get_circle_pose(env_state, circ_idx: int) -> Tuple[np.ndarray, float]:
    """Get position and rotation of a circle."""
    pos = np.array(
        [
            float(np.array(env_state.circle.position)[circ_idx, 0]),
            float(np.array(env_state.circle.position)[circ_idx, 1]),
        ]
    )
    rot = float(np.array(env_state.circle.rotation)[circ_idx])
    return pos, rot


def _compute_child_pose_from_parent(
    parent_pos: np.ndarray,
    parent_rot: float,
    parent_relative_pos: np.ndarray,
    child_relative_pos: np.ndarray,
    new_joint_rotation: float,
    parent_is_body_a: bool,
) -> Tuple[np.ndarray, float]:
    """Compute new position and rotation of child body given parent and joint angle.

    Joint rotation convention: joint_rotation = B_rot - A_rot

    Args:
        parent_pos: Position of the fixed/parent body
        parent_rot: Rotation of the fixed/parent body
        parent_relative_pos: Joint position in parent's local frame
        child_relative_pos: Joint position in child's local frame
        new_joint_rotation: New joint rotation value
        parent_is_body_a: If True, parent is body A; if False, parent is body B
    """
    # Compute joint global position (determined by parent)
    global_joint_pos = parent_pos + _rotate_point(parent_relative_pos, parent_rot)

    # Compute child's new rotation based on joint convention: joint_rot = B_rot - A_rot
    if parent_is_body_a:
        # Parent is A, child is B: B_rot = A_rot + joint_rot
        child_rot = parent_rot + new_joint_rotation
    else:
        # Parent is B, child is A: A_rot = B_rot - joint_rot
        child_rot = parent_rot - new_joint_rotation

    # Compute child's new position
    child_pos = global_joint_pos - _rotate_point(child_relative_pos, child_rot)

    return child_pos, child_rot


def _update_polygon_pose(env_state, poly_idx: int, new_pos: np.ndarray, new_rot: float):
    """Update a polygon's position and rotation."""
    return env_state.replace(
        polygon=env_state.polygon.replace(
            position=env_state.polygon.position.at[poly_idx].set(jnp.array(new_pos, dtype=jnp.float32)),
            rotation=env_state.polygon.rotation.at[poly_idx].set(jnp.array(new_rot, dtype=jnp.float32)),
        )
    )


def _apply_arm_joint_angles(
    env_state,
    static_env_params,
    new_joint_4_angle: float,
    new_joint_5_angle: float,
    num_polygons: int,
):
    """Apply new arm joint angles, keeping gray circle fixed.

    Kinematic chain: Circle 0 (fixed) → Joint 5 → Polygon 5 → Joint 4 → Polygon 4

    Joint 5: a_index=5 (polygon 5), b_index=12 (circle 0)
        - Circle 0 is the fixed parent (body B)
        - Polygon 5 is the child (body A) that moves

    Joint 4: a_index=4 (polygon 4), b_index=5 (polygon 5)
        - Polygon 5 is the parent (body B, now in its new position)
        - Polygon 4 is the child (body A) that moves
    """
    # === Step 1: Apply Joint 5 (Circle 0 → Polygon 5) ===
    joint_5_info = _get_joint_info(env_state, 5)

    # Circle 0 is body B and is fixed (it's the anchor)
    circle_0_idx = joint_5_info["b_index"] - num_polygons  # Convert to circle index
    circle_0_pos, circle_0_rot = _get_circle_pose(env_state, circle_0_idx)

    # Compute polygon 5's new pose (polygon 5 is body A, circle 0 is body B/parent)
    new_poly_5_pos, new_poly_5_rot = _compute_child_pose_from_parent(
        parent_pos=circle_0_pos,
        parent_rot=circle_0_rot,
        parent_relative_pos=joint_5_info["b_relative_pos"],  # Parent is B
        child_relative_pos=joint_5_info["a_relative_pos"],  # Child is A
        new_joint_rotation=new_joint_5_angle,
        parent_is_body_a=False,  # Parent is body B
    )

    # Update polygon 5
    env_state = _update_polygon_pose(env_state, 5, new_poly_5_pos, new_poly_5_rot)

    # Update joint 5 rotation and global position
    global_joint_5_pos = circle_0_pos + _rotate_point(joint_5_info["b_relative_pos"], circle_0_rot)
    env_state = env_state.replace(
        joint=env_state.joint.replace(
            rotation=env_state.joint.rotation.at[5].set(jnp.array(new_joint_5_angle, dtype=jnp.float32)),
            global_position=env_state.joint.global_position.at[5].set(jnp.array(global_joint_5_pos, dtype=jnp.float32)),
        )
    )

    # === Step 2: Apply Joint 4 (Polygon 5 → Polygon 4) ===
    joint_4_info = _get_joint_info(env_state, 4)

    # Polygon 5 is body B (parent, now at new position)
    # Polygon 4 is body A (child that moves)
    new_poly_4_pos, new_poly_4_rot = _compute_child_pose_from_parent(
        parent_pos=new_poly_5_pos,
        parent_rot=new_poly_5_rot,
        parent_relative_pos=joint_4_info["b_relative_pos"],  # Parent is B (polygon 5)
        child_relative_pos=joint_4_info["a_relative_pos"],  # Child is A (polygon 4)
        new_joint_rotation=new_joint_4_angle,
        parent_is_body_a=False,  # Parent is body B
    )

    # Update polygon 4
    env_state = _update_polygon_pose(env_state, 4, new_poly_4_pos, new_poly_4_rot)

    # Update joint 4 rotation and global position
    global_joint_4_pos = new_poly_5_pos + _rotate_point(joint_4_info["b_relative_pos"], new_poly_5_rot)
    env_state = env_state.replace(
        joint=env_state.joint.replace(
            rotation=env_state.joint.rotation.at[4].set(jnp.array(new_joint_4_angle, dtype=jnp.float32)),
            global_position=env_state.joint.global_position.at[4].set(jnp.array(global_joint_4_pos, dtype=jnp.float32)),
        )
    )

    # === Step 3: Update hand parts (polygons 6-9) that are attached to polygon 4 ===
    # We need to move the hand parts to follow polygon 4
    # Hand joints: 0, 1, 2, 3
    # Joint 1: a_index=4, b_index=7 (poly 4 → poly 7)
    # Joint 2: a_index=4, b_index=6 (poly 4 → poly 6)
    # Joint 0: a_index=7, b_index=9 (poly 7 → poly 9)
    # Joint 3: a_index=6, b_index=8 (poly 6 → poly 8)

    # Update polygons connected directly to polygon 4 (joints 1 and 2)
    for joint_idx in [1, 2]:
        joint_info = _get_joint_info(env_state, joint_idx)
        # Polygon 4 is body A (parent), the finger base is body B (child)
        child_idx = joint_info["b_index"]
        joint_rot = float(np.array(env_state.joint.rotation)[joint_idx])

        new_child_pos, new_child_rot = _compute_child_pose_from_parent(
            parent_pos=new_poly_4_pos,
            parent_rot=new_poly_4_rot,
            parent_relative_pos=joint_info["a_relative_pos"],
            child_relative_pos=joint_info["b_relative_pos"],
            new_joint_rotation=joint_rot,  # Keep original joint angle
            parent_is_body_a=True,
        )
        env_state = _update_polygon_pose(env_state, child_idx, new_child_pos, new_child_rot)

        # Update joint global position
        global_joint_pos = new_poly_4_pos + _rotate_point(joint_info["a_relative_pos"], new_poly_4_rot)
        env_state = env_state.replace(
            joint=env_state.joint.replace(
                global_position=env_state.joint.global_position.at[joint_idx].set(
                    jnp.array(global_joint_pos, dtype=jnp.float32)
                ),
            )
        )

    # Update finger tips (joints 0 and 3)
    for joint_idx, parent_poly_idx in [(0, 7), (3, 6)]:
        joint_info = _get_joint_info(env_state, joint_idx)
        parent_pos, parent_rot = _get_polygon_pose(env_state, parent_poly_idx)
        child_idx = joint_info["b_index"]
        joint_rot = float(np.array(env_state.joint.rotation)[joint_idx])

        new_child_pos, new_child_rot = _compute_child_pose_from_parent(
            parent_pos=parent_pos,
            parent_rot=parent_rot,
            parent_relative_pos=joint_info["a_relative_pos"],
            child_relative_pos=joint_info["b_relative_pos"],
            new_joint_rotation=joint_rot,
            parent_is_body_a=True,
        )
        env_state = _update_polygon_pose(env_state, child_idx, new_child_pos, new_child_rot)

        # Update joint global position
        global_joint_pos = parent_pos + _rotate_point(joint_info["a_relative_pos"], parent_rot)
        env_state = env_state.replace(
            joint=env_state.joint.replace(
                global_position=env_state.joint.global_position.at[joint_idx].set(
                    jnp.array(global_joint_pos, dtype=jnp.float32)
                ),
            )
        )

    return env_state


def _sample_joint_angle(
    rng: np.random.Generator,
    base_angle: float,
    cfg,
    joint_name: str,
) -> float:
    """Sample a new joint angle within configured range."""
    # Get range from config, defaulting to small variations
    angle_range = cfg.get(f"{joint_name}_range", [-0.5, 0.5])
    min_delta, max_delta = angle_range

    delta = float(rng.uniform(low=min_delta, high=max_delta))
    return base_angle + delta


def _compose_filename(stem: str, i: int) -> str:
    return f"{stem}_v{str(i).zfill(4)}.json"


@hydra.main(version_base=None, config_path=".", config_name="joint_variations")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    src_path = get_correct_path_of_json_level(cfg.input_json)
    env_state, static_env_params, env_params = load_from_json_file(src_path)

    num_polygons = static_env_params.num_polygons

    # Get base joint angles
    joint_4_base = float(np.array(env_state.joint.rotation)[4])
    joint_5_base = float(np.array(env_state.joint.rotation)[5])

    print(f"Base joint 4 angle: {joint_4_base:.4f} rad ({np.degrees(joint_4_base):.2f} deg)")
    print(f"Base joint 5 angle: {joint_5_base:.4f} rad ({np.degrees(joint_5_base):.2f} deg)")

    # Find shapes for position variation
    vary_positions = cfg.get("vary_positions", False)
    if vary_positions:
        blue_circle_idx = _find_blue_circle(env_state)
        green_square_idx = _find_green_square(env_state)
        podium_idx = _find_podium(env_state, green_square_idx)

        blue_radius = _get_shape_radius(env_state, "circle", blue_circle_idx)
        podium_radius = _get_shape_radius(env_state, "polygon", podium_idx)
        green_radius = _get_shape_radius(env_state, "polygon", green_square_idx)
        # Use larger radius for green+podium group
        green_podium_radius = max(green_radius, podium_radius)

        print(f"Found blue circle at index: {blue_circle_idx}")
        print(f"Found green square at index: {green_square_idx}")
        print(f"Found podium at index: {podium_idx}")

    rng = np.random.default_rng(int(cfg.seed))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename stem
    in_stem = Path(src_path).stem
    stem = f"{in_stem}_jointvar"
    if vary_positions:
        stem = f"{in_stem}_joint_pos_var"

    for i in range(int(cfg.num_variants)):
        # Sample new joint angles
        if cfg.get("vary_joint_4", True):
            new_joint_4 = _sample_joint_angle(rng, joint_4_base, cfg, "joint_4")
        else:
            new_joint_4 = joint_4_base

        if cfg.get("vary_joint_5", True):
            new_joint_5 = _sample_joint_angle(rng, joint_5_base, cfg, "joint_5")
        else:
            new_joint_5 = joint_5_base

        # Apply joint angle changes (gray circle stays fixed, arm moves)
        variant_state = _apply_arm_joint_angles(
            env_state,
            static_env_params,
            new_joint_4,
            new_joint_5,
            num_polygons,
        )

        # Apply position changes if enabled
        if vary_positions:
            new_blue_x = _sample_x_position(rng, blue_radius, cfg, static_env_params, env_params)
            new_green_podium_x = _sample_x_position(rng, green_podium_radius, cfg, static_env_params, env_params)

            # Ensure minimum separation between blue circle and green+podium
            min_sep = cfg.get("min_separation", 0.5)
            attempts = 0
            while abs(new_blue_x - new_green_podium_x) < min_sep + blue_radius + green_podium_radius and attempts < 100:
                new_green_podium_x = _sample_x_position(rng, green_podium_radius, cfg, static_env_params, env_params)
                attempts += 1

            variant_state = _apply_position_changes(
                variant_state,
                blue_circle_idx,
                green_square_idx,
                podium_idx,
                new_blue_x,
                new_green_podium_x,
            )

        # Reset collision manifolds
        acc_rr, acc_cr, acc_cc = get_empty_collision_manifolds(static_env_params)
        variant_state = variant_state.replace(
            acc_rr_manifolds=acc_rr,
            acc_cr_manifolds=acc_cr,
            acc_cc_manifolds=acc_cc,
            collision_matrix=calculate_collision_matrix(static_env_params, variant_state.joint),
        )

        filename = _compose_filename(stem, i)
        out_path = str(out_dir / filename)
        export_env_state_to_json(out_path, variant_state, static_env_params, env_params)
        if vary_positions:
            print(
                f"Saved {out_path} (j4={np.degrees(new_joint_4):.1f}°, j5={np.degrees(new_joint_5):.1f}°, blue_x={new_blue_x:.2f}, green_x={new_green_podium_x:.2f})"
            )
        else:
            print(f"Saved {out_path} (j4={np.degrees(new_joint_4):.1f}°, j5={np.degrees(new_joint_5):.1f}°)")


if __name__ == "__main__":
    main()
