# controllers/rl_train/rl_train.py

from controller import Supervisor
import math
import os
import time
import csv
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# ---------------------------
# Importar otros files
# ---------------------------
import sys
sys.path.append(os.path.dirname(__file__))

from mapping.occupancy_grid import (
    build_occupancy_grid_from_defs,
    world_to_grid,
    save_occupancy_png,
    reachable_mask_from_start,
    save_goals_accessibility_plot,
    save_occupancy_png_with_all_values,
    collect_defs_by_prefix
)
# ----------------------------
# GPU (Apple Metal / MPS)
# ----------------------------
import torch
DEVICE = "cpu"
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# -----------------------------
# Helpers
# -----------------------------
def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_rotation(rot) -> float:
    """
    rot is axis-angle: [ax, ay, az, angle]
    Your world uses Z as vertical axis => yaw is rotation around Z.
    """
    ax, ay, az, angle = rot
    if abs(az) > 0.9:
        return angle if az >= 0 else -angle
    return 0.0


def diffdrive_wheel_speeds(v: float, w: float, wheel_radius: float, wheel_base: float):
    """
    Differential drive:
      v_l = v - w*(L/2)
      v_r = v + w*(L/2)
    Convert to wheel angular velocities (rad/s) by dividing by R.
    """
    v_l = v - w * (wheel_base / 2.0)
    v_r = v + w * (wheel_base / 2.0)
    wl = v_l / wheel_radius
    wr = v_r / wheel_radius
    return wl, wr


# -----------------------------
# Gym Env inside Webots
# -----------------------------
class WarehouseNavEnv(gym.Env):
    """
    Goal-conditioned navigation with LiDAR + relative goal direction.
    - Start pose fixed (ZE).
    - Goal sampled from TRAIN set each episode.
    - Action: (v, w) scaled from [-1,1] to ranges.
    - Observation: normalized LiDAR + [dist_norm, sin(theta), cos(theta)].

    Changes included:
    - STUCK based on "no movement" (not "no progress to goal")
    - Allow reverse (v can be negative)
    - More time pressure + idle penalty
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # --- Webots init ---
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        # --- Nodes/devices ---
        self.mir = self.robot.getFromDef("MIR")
        if self.mir is None:
            raise RuntimeError("No node with DEF 'MIR'. Set DEF MIR on the MiR100.")

        self.mir_trans = self.mir.getField("translation")
        self.mir_rot = self.mir.getField("rotation")

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)

        self.left_motor = self.robot.getDevice("middle_left_wheel_joint")
        self.right_motor = self.robot.getDevice("middle_right_wheel_joint")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # --- Approx MiR params ---
        self.WHEEL_RADIUS = 0.10
        self.WHEEL_BASE = 0.50

        # --- Episode config ---
        self.max_episode_seconds = 90.0
        self.max_steps = int(self.max_episode_seconds / self.dt)

        # --- Success / safety thresholds ---
        self.goal_radius = 0.30
        self.collision_lidar_min = 0.20
        self.near_obstacle_dist = 0.40

        # --- STUCK (NO-MOVEMENT) detection ---
        self.prev_xy = None
        self.no_move_steps = 0
        self.no_move_limit = int(6.0 / self.dt)  # 6 seconds worth of steps
        self.no_move_min = 0.005                 # 5 mm per step threshold

        # --- Goal sets ---
        self.train_goals = [
            "goal_01", "goal_03",
            "goal_06", "goal_07", "goal_10",
            "goal_11", "goal_14", "goal_15",
            "goal_17", "goal_18", "goal_21",
            "goal_23", "goal_25", "goal_26"
        ]
        self.test_goals = [
            "goal_02", "goal_04",
            "goal_05", "goal_08", "goal_09",
            "goal_12", "goal_13", "goal_16",
            "goal_19", "goal_20", "goal_22",
            "goal_24", "goal_27", "goal_28"
        ]

        # Load all goal nodes once (DEF must match)
        self.goal_nodes = {}
        for g in self.train_goals + self.test_goals:
            node = self.robot.getFromDef(g)
            if node is None:
                raise RuntimeError(f"Goal DEF '{g}' not found. Ensure DEF is set exactly to {g}.")
            self.goal_nodes[g] = node.getField("translation")

        self.active_goal_name = None
        self.active_goal_field = None

        # --- Reset pose (ZE) ---
        self.reset_translation = [-6.6, -8.5, 0.0]
        self.reset_rotation = [0.0, 0.0, 1.0, 0.0]

        # --- Action/Observation spaces ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.n_lidar = len(self.lidar.getRangeImage())  # expected ~180
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_lidar + 3,), dtype=np.float32
        )

        # --- Internal episode state ---
        self.step_count = 0
        self.prev_dist = None

        # --- Scaling for actions ---
        self.v_max = 0.75          # forward max
        self.v_back = 0.15         # backward max
        self.w_max = 1.10

        # --- Distance normalization ---
        self.dist_norm_max = 20.0

        # -----------------------------
        # Occupancy Grid (map discretization)
        # -----------------------------
        self.bounds = (-13.0, 10.0, -10.0, 12.0)
        self.cell_size = 0.25
        self.r_infl_m = 0.20  # baja a 0.35 si ves que "se come" pasillos

        # Defs EXACTOS como en Webots
        self.obstacle_defs = collect_defs_by_prefix(self.robot, prefixes=("SHELF_", "wall"))
        from mapping.occupancy_grid import aabb2d_from_solid_box_bounding_object

        for d in self.obstacle_defs:
            node = self.robot.getFromDef(d)
            aabb = aabb2d_from_solid_box_bounding_object(node)
            if aabb is None:
                continue
            x0, y0, x1, y1 = aabb
            dx = x1 - x0
            dy = y1 - y0
            # Busca rectángulos muy altos y estrechos (franja)
            if dy > 10.0 and dx < 1.0:
                print("[AABB SUSPECT]", d, "aabb=", aabb, "dx=", dx, "dy=", dy, "pos=", node.getPosition())
        print("[MAP] detected obstacle_defs =", len(self.obstacle_defs))
        print("[MAP] first 10:", self.obstacle_defs[:10])

        # --- construir occupancy grid ---
        result = build_occupancy_grid_from_defs(
            supervisor=self.robot,
            bounds=self.bounds,
            cell_size=self.cell_size,
            obstacle_defs=self.obstacle_defs,
            r_infl_m=self.r_infl_m
        )

        self.occ_grid = result[0]
        self.grid_origin = result[1]
        painted = result[2]
        missing = result[3]
        skipped = result[4]

        print(
            f"[MAP] grid={self.occ_grid.shape} cell={self.cell_size} | "
            f"painted={painted}/{len(self.obstacle_defs)} | "
            f"occupied={np.mean(self.occ_grid):.2%} | "
            f"missing={len(missing)} skipped={len(skipped)}"
        )

        if missing:
            print("[MAP] Missing DEFs (first 10):", missing[:10])
        if skipped:
            print("[MAP] Skipped (first 10):", skipped[:10])

        save_occupancy_png(
            self.occ_grid, 
            self.bounds, 
            out_path="occ_grid.png")
        print("[MAP] Saved occ grid to:", os.path.abspath("occ_grid.png"))

        save_occupancy_png_with_all_values(
            self.occ_grid,
            self.bounds,
            self.cell_size,
            out_path="occ_grid_all_values.png",
            fontsize=2
        )
        print("[MAP] Saved occ grid ALL values to:", os.path.abspath("occ_grid_all_values.png"))

        # -----------------------------
        # comprobar si los goals son alcanzables
        # -----------------------------
        start_x, start_y = self.reset_translation[0], self.reset_translation[1]
        start_cell = world_to_grid(start_x, start_y, self.grid_origin, self.cell_size)

        reach = reachable_mask_from_start(self.occ_grid, start_cell)

        goals_report = []
        blocked = []

        # Recorremos todos los goals (train + test)
        for gname, gfield in self.goal_nodes.items():
            gx, gy, _ = gfield.getSFVec3f()
            cgx, cgy = world_to_grid(gx, gy, self.grid_origin, self.cell_size)

            H, W = self.occ_grid.shape
            in_bounds = (0 <= cgx < W and 0 <= cgy < H)
            free = in_bounds and (self.occ_grid[cgy, cgx] == 0)
            reachable = free and bool(reach[cgy, cgx])

            goals_report.append((gname, gx, gy, cgx, cgy, free, reachable))

            if not reachable:
                blocked.append(gname)

        print(f"[GOALS] reachable={len(goals_report) - len(blocked)}/{len(goals_report)} | blocked={len(blocked)}")
        if blocked:
            print("[GOALS] blocked (first 15):", blocked[:15])

        # Guardar gráfica (verde=reachable, rojo=no)
        goals_xy_for_plot = [(x, y, ok) for (_, x, y, _, _, _, ok) in goals_report]
        save_goals_accessibility_plot(
            occ=self.occ_grid,
            bounds=self.bounds,
            start_xy=(start_x, start_y),
            goals_xy=goals_xy_for_plot,
            reachable_mask=reach,
            out_path="goals_accessibility.png"
        )
        print("[GOALS] Saved plot to:", os.path.abspath("goals_accessibility.png"))


    def _apply_cmd_vel(self, v: float, w: float):
        wl, wr = diffdrive_wheel_speeds(v, w, self.WHEEL_RADIUS, self.WHEEL_BASE)
        self.left_motor.setVelocity(wl)
        self.right_motor.setVelocity(wr)

    def _get_pose_xy_yaw(self):
        p = self.mir_trans.getSFVec3f()
        r = self.mir_rot.getSFRotation()
        yaw = yaw_from_rotation(r)
        return p[0], p[1], yaw

    def _get_goal_xy(self):
        g = self.active_goal_field.getSFVec3f()
        return g[0], g[1]

    def _lidar_obs(self):
        ranges = self.lidar.getRangeImage()
        max_range = float(self.lidar.getMaxRange())
        arr = np.asarray(ranges, dtype=np.float32)
        arr = np.where(np.isinf(arr), max_range, arr)
        arr = np.clip(arr, 0.0, max_range)

        arr01 = arr / max_range
        arr_m11 = (arr01 * 2.0) - 1.0
        return arr_m11, float(np.min(arr))

    def _goal_features(self, x: float, y: float, yaw: float):
        gx, gy = self._get_goal_xy()
        dx = gx - x
        dy = gy - y
        dist = math.sqrt(dx * dx + dy * dy)
        angle_to_goal = math.atan2(dy, dx)
        rel = wrap_to_pi(angle_to_goal - yaw)

        dist_norm = np.clip(dist / self.dist_norm_max, 0.0, 1.0)
        dist_m11 = (dist_norm * 2.0) - 1.0
        return dist, rel, float(dist_m11), float(math.sin(rel)), float(math.cos(rel))

    def _get_obs(self):
        x, y, yaw = self._get_pose_xy_yaw()
        lidar_m11, lidar_min = self._lidar_obs()
        dist, rel, dist_m11, s, c = self._goal_features(x, y, yaw)

        obs = np.concatenate([lidar_m11, np.array([dist_m11, s, c], dtype=np.float32)], axis=0)
        return obs.astype(np.float32), dist, rel, lidar_min

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Stop robot
        self._apply_cmd_vel(0.0, 0.0)

        # Teleport to ZE
        self.mir_trans.setSFVec3f(self.reset_translation)
        self.mir_rot.setSFRotation(self.reset_rotation)

        # Reset physics
        self.robot.simulationResetPhysics()

        # Choose a random TRAIN goal
        self.active_goal_name = self.np_random.choice(self.train_goals)
        self.active_goal_field = self.goal_nodes[self.active_goal_name]

        # Print goal for visibility
        gx0, gy0, gz0 = self.active_goal_field.getSFVec3f()
        print(f"[RESET] New episode | goal={self.active_goal_name} at ({gx0:.2f}, {gy0:.2f}, {gz0:.2f})")

        # DEBUG: comprobar celdas robot/goal (solo para validar el grid)
        rx, ry, _ = self._get_pose_xy_yaw()
        rgx, rgy = world_to_grid(rx, ry, self.grid_origin, self.cell_size)

        gx, gy = self._get_goal_xy()
        ggx, ggy = world_to_grid(gx, gy, self.grid_origin, self.cell_size)

        H, W = self.occ_grid.shape

        def cell(cx, cy):
            if 0 <= cx < W and 0 <= cy < H:
                return int(self.occ_grid[cy, cx])
            return None

        print(f"[MAP] robot_cell=({rgx},{rgy}) occ={cell(rgx,rgy)} | goal_cell=({ggx},{ggy}) occ={cell(ggx,ggy)}")

        # Let simulation settle a few steps so LiDAR updates
        for _ in range(5):
            if self.robot.step(self.timestep) == -1:
                break

        # Reset counters
        self.step_count = 0
        self.no_move_steps = 0

        x, y, _ = self._get_pose_xy_yaw()
        self.prev_xy = (x, y)

        obs, dist, rel, lidar_min = self._get_obs()
        self.prev_dist = dist

        info = {
            "goal": self.active_goal_name,
            "dist": dist,
            "rel_angle": rel,
            "lidar_min": lidar_min,
        }
        return obs, info

    def step(self, action):
        self.step_count += 1

        # Scale action [-1,1] -> (v,w)
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)

        # v in [-v_back, v_max]
        v = ((a[0] + 1.0) / 2.0) * (self.v_max + self.v_back) - self.v_back
        w = a[1] * self.w_max

        self._apply_cmd_vel(float(v), float(w))

        if self.robot.step(self.timestep) == -1:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {"terminated_reason": "simulation_end"}

        obs, dist, rel, lidar_min = self._get_obs()

        # Termination checks
        success = dist <= self.goal_radius
        collision = lidar_min < self.collision_lidar_min
        timeout = self.step_count >= self.max_steps

        # STUCK based on real movement
        x, y, _ = self._get_pose_xy_yaw()
        dx = x - self.prev_xy[0]
        dy = y - self.prev_xy[1]
        moved = math.sqrt(dx * dx + dy * dy)

        if moved < self.no_move_min:
            self.no_move_steps += 1
        else:
            self.no_move_steps = 0

        stuck = self.no_move_steps >= self.no_move_limit
        self.prev_xy = (x, y)

        terminated = success or collision or stuck
        truncated = timeout and not terminated

        # Reward shaping (goal-directed)
        progress = (self.prev_dist - dist)
        r_progress = 25.0 * progress

        # stronger time penalty + idle penalty
        r_time = -0.02
        r_idle = -0.02 if abs(v) < 0.05 else 0.0

        r_near = 0.0
        if lidar_min < self.near_obstacle_dist:
            r_near = -0.2 * (self.near_obstacle_dist - lidar_min)

        r_heading = 0.5 * math.cos(rel)
        r_ctrl = -0.001 * abs(w)

        reward = r_progress + r_time + r_idle + r_near + r_heading + r_ctrl

        if success:
            reward += 200.0
        if collision:
            reward -= 100.0
        if stuck:
            reward -= 20.0

        self.prev_dist = dist

        # Print episode result (only when done)
        if terminated or truncated:
            if success:
                print(f"[DONE] SUCCESS | goal={self.active_goal_name} | dist={dist:.2f} | steps={self.step_count}")
            elif collision:
                print(f"[DONE] COLLISION | goal={self.active_goal_name} | lidar_min={lidar_min:.2f} | dist={dist:.2f} | steps={self.step_count}")
            elif stuck:
                print(f"[DONE] STUCK | goal={self.active_goal_name} | dist={dist:.2f} | steps={self.step_count}")
            elif truncated:
                print(f"[DONE] TIMEOUT | goal={self.active_goal_name} | dist={dist:.2f} | steps={self.step_count}")

        info = {
            "goal": self.active_goal_name,
            "dist": dist,
            "rel_angle": rel,
            "lidar_min": lidar_min,
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "stuck": stuck,
            "steps": int(self.step_count),
            "episode_time_s": float(self.step_count * self.dt),
        }

        return obs, float(reward), bool(terminated), bool(truncated), info


# -----------------------------
# Metrics + CSV (single file)
# -----------------------------
class EpisodeStatsCSVCallback(BaseCallback):
    """
    - Logs custom/* to TensorBoard
    - Prints cumulative counts every print_every episodes
    - Appends one row per finished episode to a single CSV file

    CSV columns:
    timestamp, run_name, episode_idx, goal, result, dist, lidar_min, steps, episode_time_s
    """

    def __init__(self, run_name: str, csv_path: str, print_every: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.run_name = run_name
        self.csv_path = csv_path
        self.print_every = print_every

        self.episode_idx = 0
        self.count_success = 0
        self.count_collision = 0
        self.count_timeout = 0
        self.count_stuck = 0

    def _append_csv_row(self, row: dict):
        fieldnames = [
            "timestamp", "run_name", "episode_idx", "goal", "result",
            "dist", "lidar_min", "steps", "episode_time_s"
        ]

        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if not infos or not dones:
            return True

        for info, done in zip(infos, dones):
            if not done or not isinstance(info, dict):
                continue

            self.episode_idx += 1

            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            timeout = bool(info.get("timeout", False))
            stuck = bool(info.get("stuck", False))

            if success:
                self.count_success += 1
                result = "SUCCESS"
            elif collision:
                self.count_collision += 1
                result = "COLLISION"
            elif timeout:
                self.count_timeout += 1
                result = "TIMEOUT"
            elif stuck:
                self.count_stuck += 1
                result = "STUCK"
            else:
                result = "DONE"

            # TensorBoard custom scalars
            total = max(1, self.episode_idx)
            self.logger.record("custom/success_rate", self.count_success / total)
            self.logger.record("custom/collision_rate", self.count_collision / total)
            self.logger.record("custom/timeout_rate", self.count_timeout / total)
            self.logger.record("custom/stuck_rate", self.count_stuck / total)
            self.logger.record("custom/final_dist", float(info.get("dist", 0.0)))
            self.logger.record("custom/min_lidar", float(info.get("lidar_min", 0.0)))
            self.logger.record("custom/episode_time_s", float(info.get("episode_time_s", 0.0)))

            # CSV row
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_name": self.run_name,
                "episode_idx": self.episode_idx,
                "goal": info.get("goal", ""),
                "result": result,
                "dist": float(info.get("dist", 0.0)),
                "lidar_min": float(info.get("lidar_min", 0.0)),
                "steps": int(info.get("steps", 0)),
                "episode_time_s": float(info.get("episode_time_s", 0.0)),
            }
            self._append_csv_row(row)

            if self.episode_idx % self.print_every == 0:
                print(
                    f"[STATS] episodes={self.episode_idx} | "
                    f"SUCCESS={self.count_success} | COLLISION={self.count_collision} | "
                    f"TIMEOUT={self.count_timeout} | STUCK={self.count_stuck}"
                )

        return True


# -----------------------------
# Training entrypoint
# -----------------------------
def main():
    env = WarehouseNavEnv()

    run_name = time.strftime("ppo_warehouse_%Y%m%d_%H%M%S")
    print("CWD (working dir):", os.getcwd())

    tb_dir = os.path.join("runs", run_name)
    os.makedirs(tb_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join("checkpoints", run_name),
        name_prefix="ppo_mir"
    )

    # One single CSV for all runs (accumulated)
    csv_path = os.path.join(os.getcwd(), "training_history.csv")
    stats_csv_callback = EpisodeStatsCSVCallback(
        run_name=run_name,
        csv_path=csv_path,
        print_every=50
    )

    model = PPO(
        policy="MlpPolicy",
        device=DEVICE,
        env=env,
        verbose=1,
        tensorboard_log=tb_dir,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0
    )

    print("Using device:", DEVICE)
    print("CSV path:", csv_path)

    total_timesteps = 1_000_000

    print(f"\nTraining PPO... TensorBoard logs at: {tb_dir}")
    print("To watch live:")
    print("  tensorboard --logdir runs\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, stats_csv_callback],
        tb_log_name="PPO"
    )

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{run_name}.zip")
    model.save(model_path)
    print(f"\nSaved model to: {model_path}")

    env._apply_cmd_vel(0.0, 0.0)


if __name__ == "__main__":
    main()