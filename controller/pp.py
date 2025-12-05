"""
Pure Pursuit (PP) Controller implementation.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class PPResult:
    """Result structure for PP controller output."""
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


class PP_Controller:
    """
    Pure Pursuit controller with adaptive lookahead distance.
    Does not use steering lookup table - uses geometric Pure Pursuit formula.
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
        Initialize PP controller.
        
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
            speed_lookahead_for_steer: Lookahead time for steering calculation [s]
            LUT_path: Path to steering lookup table CSV (not used in PP)
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

        # Startup flag
        self.first_steering_calculation = True

        # Note: Pure Pursuit does not use steering lookup table
        # LUT_path is accepted for interface compatibility but not used

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

    def main_loop(
        self,
        position_in_map: np.ndarray,
        waypoint_array_in_map: np.ndarray,
        speed_now: float,
        position_in_map_frenet: np.ndarray,
        acc_now: np.ndarray
    ) -> PPResult:

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

        # Calculate speed command
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
                self.logger_warn("[PP Controller] speed was none")

        # Calculate L1 point
        L1_point, L1_distance = self.calc_L1_point(lateral_error)

        # Check L1 point validity
        if not np.isfinite(L1_point).all():
            raise RuntimeError("L1_point is invalid")

        # Calculate steering angle
        steering_angle = self.calc_steering_angle(L1_point, L1_distance, yaw, lat_e_norm, v)

        # Build result
        result = PPResult(
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
        # Calculate lookahead position for speed adjustment
        adv_ts_st = self.speed_lookahead_for_steer
        la_position = np.array([
            self.position_in_map[0] + v[0] * adv_ts_st,
            self.position_in_map[1] + v[1] * adv_ts_st
        ])

        # Find nearest waypoint to lookahead position
        idx_la_steer = self.nearest_waypoint(la_position, self.waypoint_array_in_map[:, :2])

        # Calculate L1 vector
        L1_vector = L1_point - self.position_in_map[:2]
        
        # Calculate eta (angle between vehicle heading and L1 vector)
        eta = 0.0
        if np.linalg.norm(L1_vector) == 0.0:
            if self.logger_warn:
                self.logger_warn("[PP Controller] norm of L1 vector was 0, eta is set to 0")
            eta = 0.0
        else:
            nvec = np.array([-np.sin(yaw), np.cos(yaw)])
            sin_eta = np.dot(nvec, L1_vector) / np.linalg.norm(L1_vector)
            eta = np.arcsin(sin_eta)

        # Pure Pursuit steering angle formula
        steering_angle = 0.0
        if L1_distance == 0.0:
            if self.logger_warn:
                self.logger_warn("[PP Controller] L1_distance is 0, steering_angle is set to 0")
            steering_angle = 0.0
        else:
            wheelbase = 0.3302  # meters
            steering_angle = np.arctan((2.0 * wheelbase * np.sin(eta)) / L1_distance)

        # Apply rate limiting (0.4 rad/step) - skip on first calculation
        threshold = 0.4
        if self.first_steering_calculation:
            self.first_steering_calculation = False
            if self.logger_info:
                self.logger_info("[PP Controller] First steering calculation, skipping rate limiting")
        elif abs(steering_angle - self.curr_steering_angle) > threshold:
            if self.logger_info:
                clamped_angle = clamp(
                    steering_angle,
                    self.curr_steering_angle - threshold,
                    self.curr_steering_angle + threshold
                )
                self.logger_info(
                    f"[PP Controller] steering angle clipped: {steering_angle} -> {clamped_angle}"
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
        # Find nearest waypoint
        self.idx_nearest_waypoint = self.nearest_waypoint(
            self.position_in_map[:2],
            self.waypoint_array_in_map[:, :2]
        )

        if self.idx_nearest_waypoint is None:
            self.idx_nearest_waypoint = 0

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

        # Get waypoint at L1 distance ahead
        L1_point = self.waypoint_at_distance_before_car(
            L1_distance,
            self.waypoint_array_in_map[:, :2],
            self.idx_nearest_waypoint
        )

        return L1_point, L1_distance

    def calc_speed_command(self, v: np.ndarray, lat_e_norm: float) -> Optional[float]:

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

        return global_speed

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(p2 - p1)

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
