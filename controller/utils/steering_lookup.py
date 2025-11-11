"""
Steering lookup table implementation using CSV-based 2D interpolation.
Translates velocity and acceleration to steering angle.
"""

import numpy as np
from typing import Tuple, Optional


class SteeringLookup:
    """
    2D interpolation-based steering angle lookup from CSV table.
    CSV format: First row is velocity values, first column is acceleration values,
    remaining cells are steering angle values.
    """

    def __init__(self, csv_path: str):
        """
        Load CSV lookup table and prepare for interpolation.
        
        Args:
            csv_path: Path to CSV file containing steering lookup table
        """
        self.csv_path = csv_path
        self.velocity_values: np.ndarray = np.array([])
        self.acceleration_values: np.ndarray = np.array([])
        self.steering_matrix: np.ndarray = np.array([])
        
        self._load_csv()

    def _load_csv(self) -> None:
        """Load and parse CSV lookup table."""
        try:
            # Load full CSV file
            data = np.loadtxt(self.csv_path, delimiter=',')
            
            # Extract velocity values from first row (excluding first cell)
            self.velocity_values = data[0, 1:]
            
            # Extract acceleration values from first column (excluding first cell)
            self.acceleration_values = data[1:, 0]
            
            # Extract steering angle matrix (excluding header row/column)
            self.steering_matrix = data[1:, 1:]
            
        except Exception as e:
            raise RuntimeError(f"Failed to load steering lookup table from {self.csv_path}: {e}")

    def get_steering_angle(self, velocity: float, acceleration: float) -> float:
        """
        Get steering angle for given velocity and acceleration using 2D linear interpolation.
        Matches C++ crazy_controller implementation exactly (lines 118-170).
        
        Args:
            velocity: Current velocity [m/s]
            acceleration: Desired lateral acceleration [m/s^2]
        
        Returns:
            Steering angle [rad]
        """
        # Preserve sign of acceleration (C++ line 120)
        sign_accel = 1.0 if acceleration > 0.0 else -1.0
        
        # Lookup only for positive accelerations (C++ line 121)
        accel_abs = abs(acceleration)
        
        # Find nearest velocity column index (C++ lines 129-137)
        vel_idx = self._find_nearest(self.velocity_values, velocity)
        
        # Get acceleration column for this velocity (C++ lines 139-143)
        acc_column = self.steering_matrix[:, vel_idx]
        
        # Find closest neighbors in acceleration axis (C++ line 145)
        # This returns (closest_idx, second_idx)
        idx0, idx1 = self._find_closest_neighbors(acc_column, accel_abs)
        
        # Interpolate or use direct value (C++ lines 147-168)
        if idx0 == idx1:
            # No interpolation needed
            steer_angle = self.acceleration_values[idx0]
        else:
            # Linear interpolation
            x0 = acc_column[idx0]
            x1 = acc_column[idx1]
            y0 = self.acceleration_values[idx0]
            y1 = self.acceleration_values[idx1]
            
            if abs(x1 - x0) < 1e-12:
                steer_angle = y0
            else:
                t = (accel_abs - x0) / (x1 - x0)
                steer_angle = y0 + (y1 - y0) * t
        
        # Apply sign back to steering angle (C++ line 170)
        return steer_angle * sign_accel

    def _find_nearest(self, array: np.ndarray, value: float) -> int:
        """
        Find index of nearest value in sorted array.
        
        Args:
            array: Sorted 1D numpy array
            value: Value to find
        
        Returns:
            Index of nearest element
        """
        if len(array) == 0:
            return 0
        
        # Find insertion point
        idx = np.searchsorted(array, value, side='left')
        
        # Clamp to valid range
        if idx == 0:
            return 0
        if idx >= len(array):
            return len(array) - 1
        
        # Choose closest of two neighbors
        if abs(value - array[idx - 1]) < abs(value - array[idx]):
            return idx - 1
        else:
            return idx

    def _find_closest_neighbors(self, array: np.ndarray, value: float) -> Tuple[int, int]:
        """
        Find indices of two closest neighbors for interpolation.
        Handles NaN values by truncating array at first NaN.
        Matches C++ crazy_controller implementation exactly.
        
        Args:
            array: Sorted 1D numpy array (may contain NaN)
            value: Value to bracket
        
        Returns:
            Tuple of (lower_index, upper_index)
        """
        if len(array) == 0:
            raise ValueError("find_closest_neighbors: array is empty")
        
        # Remove NaN values by truncating at first NaN (C++ lines 74-80)
        is_nan_array = np.argwhere(np.isnan(array))
        if len(is_nan_array) > 0:
            first_nan = is_nan_array[0][0]
            array = array[0:first_nan]
        
        if len(array) == 0:
            raise ValueError("find_closest_neighbors: array has only NaNs")
        
        # Find nearest value (C++ find_nearest)
        distances = np.abs(array - value)
        closest_idx = int(np.argmin(distances))
        
        # Handle boundaries (C++ lines 87-108)
        if closest_idx == 0:
            # Lower boundary: clamp to minimum value
            return (0, 0)
        elif closest_idx == len(array) - 1:
            # Upper boundary: use last two points for extrapolation
            if len(array) >= 2:
                return (len(array) - 1, len(array) - 2)
            else:
                return (closest_idx, closest_idx)
        else:
            # Middle: choose closest neighbor from left or right
            left = closest_idx - 1
            right = closest_idx + 1
            
            dl = abs(array[left] - value)
            dr = abs(array[right] - value)
            
            second_idx = right if dr < dl else left
            
            return (closest_idx, second_idx)
