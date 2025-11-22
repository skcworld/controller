"""
MAP (Model-and Acceleration-based Pursuit) Controller implementation.
논문(알고리즘 자체의 비교)를 위해서, 제어 성능을 높이기 위해 휴리스틱하게 추가된 알고리즘들을 전부 주석처리함. 알고리즘 자체의 비교만 가능하게 함.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from controller.utils.steering_lookup import SteeringLookup


@dataclass
class MAPResult:
    """Result structure for MAP controller output."""
    speed: float
    acceleration: float
    jerk: float
    steering_angle: float
    L1_point: np.ndarray
    L1_distance: float
    idx_nearest_waypoint: int


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))


class MAP_Controller:
    """
    Model-Adaptive Pursuit controller with L1 adaptive lookahead distance.
    Uses steering lookup table, acceleration scaling, and lateral error compensation.
    """

    def __init__(
        self,
        t_clip_min: float,
        t_clip_max: float,
        m_l1: float,
        q_l1: float,
        speed_lookahead: float,
        lat_err_coeff: float,
        acc_scaler_for_steer: float,
        dec_scaler_for_steer: float,
        start_scale_speed: float,
        end_scale_speed: float,
        downscale_factor: float,
        speed_lookahead_for_steer: float,
        diff_threshold: float,
        deacc_gain: float,
        LUT_path: str,
        logger_info: Optional[Callable[[str], None]] = None,
        logger_warn: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize MAP controller.
        
        Args:
            t_clip_min: Minimum L1 lookahead distance [m]
            t_clip_max: Maximum L1 lookahead distance [m]
            m_l1: L1 distance velocity multiplier [s]
            q_l1: L1 distance base offset [m]
            speed_lookahead: Lookahead time for speed command [s]
            lat_err_coeff: Lateral error speed reduction coefficient
            acc_scaler_for_steer: Steering scale factor during acceleration
            dec_scaler_for_steer: Steering scale factor during deceleration
            start_scale_speed: Speed at which steering downscaling begins [m/s]
            end_scale_speed: Speed at which steering downscaling reaches maximum [m/s]
            downscale_factor: Maximum steering downscale factor
            speed_lookahead_for_steer: Lookahead time for steering lookup [s]
            diff_threshold: Speed difference threshold for startup blending [m/s]
            deacc_gain: Blending gain for startup deacceleration
            LUT_path: Path to steering lookup table CSV
            logger_info: Optional info logging callback
            logger_warn: Optional warning logging callback
        """
        # Parameter storage
        self.t_clip_min = t_clip_min
        self.t_clip_max = t_clip_max
        self.m_l1 = m_l1
        self.q_l1 = q_l1
        self.speed_lookahead = speed_lookahead
        self.lat_err_coeff = lat_err_coeff
        self.acc_scaler_for_steer = acc_scaler_for_steer
        self.dec_scaler_for_steer = dec_scaler_for_steer
        self.start_scale_speed = start_scale_speed
        self.end_scale_speed = end_scale_speed
        self.downscale_factor = downscale_factor
        self.speed_lookahead_for_steer = speed_lookahead_for_steer
        self.diff_threshold = diff_threshold
        self.deacc_gain = deacc_gain
        self.LUT_path = LUT_path

        # Logging callbacks
        self.logger_info = logger_info
        self.logger_warn = logger_warn

        # State variables
        self.position_in_map: np.ndarray = np.zeros(3)
        self.waypoint_array_in_map: np.ndarray = np.zeros((0, 8))
        self.speed_now: float = 0.0
        self.position_in_map_frenet: np.ndarray = np.zeros(2)
        self.acc_now: np.ndarray = np.zeros(10)
        self.speed_command: Optional[float] = None
        self.idx_nearest_waypoint: Optional[int] = None
        self.curr_steering_angle: float = 0.0
        self.curvature_waypoints: float = 0.0

        # Startup blending flag
        self.first_steering_calculation = True

        # Initialize steering lookup table
        self.steering_lookup = SteeringLookup(LUT_path)

    # Parameter setters for dynamic reconfiguration
    def set_t_clip_min(self, value: float) -> None:
        self.t_clip_min = value

    def set_t_clip_max(self, value: float) -> None:
        self.t_clip_max = value

    def set_m_l1(self, value: float) -> None:
        self.m_l1 = value

    def set_q_l1(self, value: float) -> None:
        self.q_l1 = value

    def set_speed_lookahead(self, value: float) -> None:
        self.speed_lookahead = value

    def set_lat_err_coeff(self, value: float) -> None:
        self.lat_err_coeff = value

    def set_acc_scaler_for_steer(self, value: float) -> None:
        self.acc_scaler_for_steer = value

    def set_dec_scaler_for_steer(self, value: float) -> None:
        self.dec_scaler_for_steer = value

    def set_start_scale_speed(self, value: float) -> None:
        self.start_scale_speed = value

    def set_end_scale_speed(self, value: float) -> None:
        self.end_scale_speed = value

    def set_downscale_factor(self, value: float) -> None:
        self.downscale_factor = value

    def set_speed_lookahead_for_steer(self, value: float) -> None:
        self.speed_lookahead_for_steer = value

    def set_diff_threshold(self, value: float) -> None:
        self.diff_threshold = value

    def set_deacc_gain(self, value: float) -> None:
        self.deacc_gain = value

    def main_loop(
        self,
        position_in_map: np.ndarray,
        waypoint_array_in_map: np.ndarray,
        speed_now: float,
        position_in_map_frenet: np.ndarray,
        acc_now: np.ndarray
    ) -> MAPResult:
        """
        Main control loop - computes steering and speed commands.
        
        Args:
            position_in_map: Vehicle pose [x, y, yaw] in map frame
            waypoint_array_in_map: Waypoint matrix [N x 8]: [x, y, speed, ratio, s, kappa, psi, ax]
            speed_now: Current vehicle speed [m/s]
            position_in_map_frenet: Frenet coordinates [s, d]
            acc_now: Acceleration buffer (10 samples) [m/s^2]
        
        Returns:
            MAPResult with speed, steering, and debug info
        """
        # Update state
        self.position_in_map = position_in_map
        self.waypoint_array_in_map = waypoint_array_in_map
        self.speed_now = speed_now
        self.position_in_map_frenet = position_in_map_frenet
        self.acc_now = acc_now

        # Extract yaw and compute velocity vector
        yaw = position_in_map[2]
        v = np.array([np.cos(yaw) * speed_now, np.sin(yaw) * speed_now])

        # Calculate lateral error normalization
        lat_e_norm, lateral_error = self.calc_lateral_error_norm()

        # Calculate speed command (BEFORE L1 point calculation - matches C++ order)
        self.speed_command = self.calc_speed_command(v, lat_e_norm)

        speed = 0.0
        acceleration = 0.0
        jerk = 0.0

        if self.speed_command is not None:
            speed = max(0.0, self.speed_command)
        else:
            speed = 0.0
            acceleration = 0.0
            jerk = 0.0
            if self.logger_warn:
                self.logger_warn("[MAP Controller] speed was none")

        # Calculate L1 point
        L1_point, L1_distance = self.calc_L1_point(lateral_error)

        # Check L1 point validity
        if not np.isfinite(L1_point).all():
            raise RuntimeError("L1_point is invalid")

        # Startup blending: if the difference between current speed and
        # the nearest waypoint's speed profile is >= diff_threshold, treat as initial rollout
        # and blend the final commanded speed with current speed to avoid aggressive jumps.
        # Matches C++ lines 352-370
        # if self.idx_nearest_waypoint is not None:
        #     nearest_idx = self.idx_nearest_waypoint
        #     if 0 <= nearest_idx < self.waypoint_array_in_map.shape[0]:
        #         profile_speed = self.waypoint_array_in_map[nearest_idx, 2]
        #         diff = abs(profile_speed - self.speed_now)
        #         if diff >= self.diff_threshold:
        #             prev_speed = speed
        #             # C++ formula: speed = deacc_gain * (speed + speed_now)
        #             speed = self.deacc_gain * (speed + self.speed_now)
        #             if self.logger_info:
        #                 self.logger_info(
        #                     f"[MAP Controller] Startup blend active: |profile - v| = {diff:.2f} m/s "
        #                     f"(threshold={self.diff_threshold:.2f}), gain={self.deacc_gain:.2f}, "
        #                     f"speed {prev_speed:.2f} -> {speed:.2f}"
        #                 )

        # Calculate steering angle (AFTER speed calculation and blending - matches C++ order)
        steering_angle = self.calc_steering_angle(L1_point, L1_distance, yaw, lat_e_norm, v)

        # Build result
        result = MAPResult(
            speed=speed,
            acceleration=acceleration,
            jerk=jerk,
            steering_angle=steering_angle,
            L1_point=L1_point,
            L1_distance=L1_distance,
            idx_nearest_waypoint=self.idx_nearest_waypoint if self.idx_nearest_waypoint is not None else -1
        )

        return result

    def calc_steering_angle(
        self,
        L1_point: np.ndarray,
        L1_distance: float,
        yaw: float,
        lat_e_norm: float,
        v: np.ndarray
    ) -> float:
        """
        Calculate steering angle using lookup table and multi-stage scaling.
        
        Pipeline: lookup(lat_acc, speed) → speed_steer_scaling → acc_scaling
                 → speed multiplier (1.0-1.25) → rate limiting (0.4) → clamp(±0.45)
        
        Args:
            L1_point: Target L1 point [x, y] in map frame
            L1_distance: Distance to L1 point [m]
            yaw: Vehicle yaw angle [rad]
            lat_e_norm: Normalized lateral error [0-0.5]
            v: Velocity vector [vx, vy] [m/s]
        
        Returns:
            Steering angle [rad]
        """
        # Calculate lookahead position for steering lookup
        adv_ts_st = self.speed_lookahead_for_steer
        la_position = np.array([
            self.position_in_map[0] + v[0] * adv_ts_st,
            self.position_in_map[1] + v[1] * adv_ts_st
        ])

        # Find nearest waypoint to lookahead position
        idx_la_steer = self.nearest_waypoint(la_position, self.waypoint_array_in_map[:, :2])

        # Get speed at lookahead position for lookup table
        speed_la_for_lu = self.waypoint_array_in_map[idx_la_steer, 2]
        # speed_for_lu = self.speed_adjust_lat_err(speed_la_for_lu, lat_e_norm)

        # Calculate L1 geometry
        L1_vector = L1_point - self.position_in_map[:2]
        
        if np.linalg.norm(L1_vector) == 0.0:
            if self.logger_warn:
                self.logger_warn("[MAP Controller] norm of L1 vector was 0, lateral_acc set to 0")
            lateral_acc = 0.0
        else:
            nvec = np.array([-np.sin(yaw), np.cos(yaw)])
            sin_eta = np.dot(nvec, L1_vector) / np.linalg.norm(L1_vector)
            sin_eta = clamp(sin_eta, -1.0, 1.0)
            eta = np.arcsin(sin_eta)
            # CRITICAL: Factor of 2 is required for correct lateral acceleration
            # C++ 수정 1: speed_for_lu를 speed_now로 변경 (2025.10.31 새벽에 수정함)
            lateral_acc = 2.0 * (self.speed_now ** 2) * np.sin(eta) / L1_distance

        # Lookup steering angle from table
        # C++ 수정 2: speed_for_lu를 speed_now로 변경 (2025.10.31 새벽에 수정함)
        steering_angle = self.steering_lookup.get_steering_angle(self.speed_now, lateral_acc)

        # C++ 수정 4: speed_steer_scaling에 speed_now 전달
        # steering_angle = self.speed_steer_scaling(steering_angle, self.speed_now)

        # Apply acceleration-based scaling
        # steering_angle = self.acc_scaling(steering_angle)

        # Apply speed multiplier (1.0 to 1.25 based on speed)
        # steering_angle *= clamp(1.0 + (self.speed_now / 10.0), 1.0, 1.25)

        # Apply rate limiting (0.4 rad/step) - skip on first calculation
        threshold = 0.4
        if self.first_steering_calculation:
            self.first_steering_calculation = False
            if self.logger_info:
                self.logger_info("[MAP Controller] First steering calculation, skipping rate limiting")
        elif abs(steering_angle - self.curr_steering_angle) > threshold:
            if self.logger_info:
                clamped_angle = clamp(
                    steering_angle,
                    self.curr_steering_angle - threshold,
                    self.curr_steering_angle + threshold
                )
                self.logger_info(
                    f"[MAP Controller] steering angle clipped: {steering_angle} -> {clamped_angle}"
                )
            steering_angle = clamp(
                steering_angle,
                self.curr_steering_angle - threshold,
                self.curr_steering_angle + threshold
            )

        # Apply hard limits
        max_steering_angle = 0.45
        steering_angle = clamp(steering_angle, -max_steering_angle, max_steering_angle)

        self.curr_steering_angle = steering_angle
        return steering_angle

    def calc_L1_point(self, lateral_error: float) -> Tuple[np.ndarray, float]:
        """
        Calculate L1 target point with adaptive lookahead distance.
        
        L1_distance = q_l1 + speed_now * m_l1
        Bounded by: max(t_clip_min, lateral_multiplier * lateral_error) to t_clip_max
        
        Args:
            lateral_error: Absolute lateral error [m]
        
        Returns:
            Tuple of (L1_point [x,y], L1_distance)
        """
        # Find nearest waypoint
        self.idx_nearest_waypoint = self.nearest_waypoint(
            self.position_in_map[:2],
            self.waypoint_array_in_map[:, :2]
        )

        if self.idx_nearest_waypoint is None:
            self.idx_nearest_waypoint = 0

        # Calculate mean curvature from nearest waypoint forward
        if (self.waypoint_array_in_map.shape[0] - self.idx_nearest_waypoint) > 2:
            lookahead_idx = int(np.floor(self.speed_now * self.speed_lookahead * 1.0 * 10.0))
            end_idx = min(
                self.idx_nearest_waypoint + lookahead_idx,
                self.waypoint_array_in_map.shape[0]
            )
            
            self.curvature_waypoints = np.mean(
                np.abs(self.waypoint_array_in_map[self.idx_nearest_waypoint:end_idx, 5])
            )

        # Calculate adaptive L1 distance
        L1_distance = self.q_l1 + self.speed_now * self.m_l1

        # Apply lateral error-based lower bound
        # lateral_multiplier = 2.0 if lateral_error > 1.0 else np.sqrt(2.0)
        # lower_bound = max(self.t_clip_min, lateral_multiplier * lateral_error)
        # L1_distance = clamp(L1_distance, lower_bound, self.t_clip_max)

        # if self.logger_info and lateral_error > 1.0:
        #     self.logger_info(
        #         f"[MAP Controller] Large lateral error: {lateral_error}m, L1_distance: {L1_distance}m"
        #     )

        # Get waypoint at L1 distance ahead
        L1_point = self.waypoint_at_distance_before_car(
            L1_distance,
            self.waypoint_array_in_map[:, :2],
            self.idx_nearest_waypoint
        )

        return L1_point, L1_distance

    def calc_speed_command(self, v: np.ndarray, lat_e_norm: float) -> Optional[float]:
        """
        Calculate speed command with lateral error adjustment.
        Matches C++ lines 518-534.
        
        Args:
            v: Velocity vector [vx, vy] [m/s]
            lat_e_norm: Normalized lateral error [0-0.5]
        
        Returns:
            Speed command [m/s] or None
        """
        # Calculate lookahead position
        adv_ts_sp = self.speed_lookahead
        la_position = np.array([
            self.position_in_map[0] + v[0] * adv_ts_sp,
            self.position_in_map[1] + v[1] * adv_ts_sp
        ])

        # Find nearest waypoint to lookahead position
        idx_la_position = self.nearest_waypoint(la_position, self.waypoint_array_in_map[:, :2])

        # Get global speed from waypoint
        global_speed = self.waypoint_array_in_map[idx_la_position, 2]
        
        # Adjust speed based on lateral error
        # global_speed = self.speed_adjust_lat_err(global_speed, lat_e_norm)

        return global_speed

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(p2 - p1)

    # def acc_scaling(self, steer: float) -> float:
    #     """
    #     Scale steering based on mean acceleration state.
    #     Matches C++ line 541-545.
        
    #     Accelerating (mean_acc >= 0.8): multiply by acc_scaler_for_steer
    #     Decelerating (mean_acc <= -0.8): multiply by dec_scaler_for_steer
        
    #     Args:
    #         steer: Input steering angle [rad]
        
    #     Returns:
    #         Scaled steering angle [rad]
    #     """
    #     mean_acc = np.mean(self.acc_now) if len(self.acc_now) > 0 else 0.0
        
    #     if mean_acc >= 0.8:
    #         return steer * self.acc_scaler_for_steer
    #     elif mean_acc <= -0.8:
    #         return steer * self.dec_scaler_for_steer
        
    #     return steer

    # def speed_steer_scaling(self, steer: float, speed: float) -> float:
    #     """
    #     Apply speed-based downscaling to steering angle.
        
    #     Linear interpolation from 1.0 (at start_scale_speed) to 
    #     (1.0 - downscale_factor) at end_scale_speed.
        
    #     Args:
    #         steer: Input steering angle [rad]
    #         speed: Current speed [m/s]
        
    #     Returns:
    #         Scaled steering angle [rad]
    #     """
    #     speed_diff = max(0.1, self.end_scale_speed - self.start_scale_speed)
    #     t = clamp((speed - self.start_scale_speed) / speed_diff, 0.0, 1.0)
    #     factor = 1.0 - t * self.downscale_factor
    #     return steer * factor

    def calc_lateral_error_norm(self) -> Tuple[float, float]:
        """
        Calculate normalized lateral error.
        
        Maps lateral error [0-2m] to normalized range [0-0.5].
        
        Returns:
            Tuple of (normalized_error, absolute_error)
        """
        lateral_error = abs(self.position_in_map_frenet[1])

        # Clip to [0, 2] meters and normalize to [0, 0.5]
        max_lat_e = 2.0
        min_lat_e = 0.0
        lat_e_clip = clamp(lateral_error, min_lat_e, max_lat_e)
        lat_e_norm = 0.5 * ((lat_e_clip - min_lat_e) / (max_lat_e - min_lat_e))

        return lat_e_norm, lateral_error

    # def speed_adjust_lat_err(self, global_speed: float, lat_e_norm: float) -> float:
    #     """
    #     Adjust speed based on lateral error and curvature.
        
    #     Speed reduction: speed *= (1 - lat_err_coeff + lat_err_coeff * exp(-lat_e_norm * curv))
        
    #     Args:
    #         global_speed: Nominal speed [m/s]
    #         lat_e_norm: Normalized lateral error [0-0.5]
        
    #     Returns:
    #         Adjusted speed [m/s]
    #     """
    #     lat_e_coeff = self.lat_err_coeff
    #     lat_e_norm *= 2.0  # Scale to [0-1]

    #     # Normalize curvature to [0, 1]
    #     curv = clamp(2.0 * (self.curvature_waypoints / 0.8) - 2.0, 0.0, 1.0)
        
    #     # Exponential speed reduction
    #     global_speed *= (1.0 - lat_e_coeff + lat_e_coeff * np.exp(-lat_e_norm * curv))

    #     return global_speed

    def nearest_waypoint(self, position: np.ndarray, waypoints_xy: np.ndarray) -> int:
        """
        Find index of nearest waypoint to position.
        
        Args:
            position: Query position [x, y]
            waypoints_xy: Waypoint array [N x 2]
        
        Returns:
            Index of nearest waypoint
        """
        N = waypoints_xy.shape[0]
        if N <= 0:
            return 0

        # Calculate distances to all waypoints
        distances = np.linalg.norm(waypoints_xy - position, axis=1)
        best_idx = np.argmin(distances)

        return int(best_idx)

    def waypoint_at_distance_before_car(
        self,
        distance: float,
        waypoints_xy: np.ndarray,
        idx_waypoint_behind_car: int
    ) -> np.ndarray:
        """
        Get waypoint at specified distance ahead of car.
        
        Assumes waypoints are spaced at 0.1m intervals.
        
        Args:
            distance: Desired lookahead distance [m]
            waypoints_xy: Waypoint array [N x 2]
            idx_waypoint_behind_car: Index of nearest waypoint behind car
        
        Returns:
            Waypoint position [x, y]
        """
        if not np.isfinite(distance):
            distance = self.t_clip_min

        d_distance = distance
        waypoints_distance = 0.1
        d_index = int(d_distance / waypoints_distance + 0.5)

        idx = min(waypoints_xy.shape[0] - 1, idx_waypoint_behind_car + d_index)
        return waypoints_xy[idx]
