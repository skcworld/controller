"""
Unified Controller Manager for MAP, PP, and AUG controllers.
ROS2 node that manages all control algorithms with mode parameter selection.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import numpy as np
import yaml
import signal
import sys
from typing import Optional, Tuple
from threading import Lock
import traceback

# ROS2 messages
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from ae_hyu_msgs.msg import WpntArray
from rcl_interfaces.msg import ParameterEvent, ParameterDescriptor, FloatingPointRange, ParameterType

# TF2
import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped

# Controller implementations
from controller.map import MAP_Controller
from controller.pp import PP_Controller
from controller.aug import AUG_Controller
from controller.utils.parameter_handler import ParameterEventHandler


class ControllerManager(Node):
    """
    Unified controller manager node supporting MAP, PP, and AUG control modes.
    Handles all ROS2 subscriptions, TF transforms, and control loop orchestration.
    """

    def __init__(self):
        super().__init__(
            'controller_manager',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        # Shutdown handling
        self.shutdown_requested = False

        # Initialize node
        if not self.initialize():
            self.get_logger().error("Failed to initialize controller")
            return

        self.get_logger().info("Controller initialized successfully")

    def initialize(self) -> bool:
        """Initialize controller node and all components."""
        try:
            # Get essential parameters
            # rclpy Parameter API exposes the value via the 'value' attribute
            self.LUT_path = self.get_parameter('lookup_table_path').value
            self.mode = self.get_parameter('mode').value

            # Get TF frame parameters
            self.map_frame = self.get_parameter('map_frame').value if self.has_parameter('map_frame') else 'map'
            self.base_link_frame = self.get_parameter('base_link_frame').value if self.has_parameter('base_link_frame') else 'base_link'

            self.get_logger().info(f"Using lookup table: {self.LUT_path}")
            self.get_logger().info(f"Using TF: {self.map_frame} -> {self.base_link_frame}")
            self.get_logger().info(f"Controller mode: {self.mode}")

            # Initialize TF
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

            # Initialize publishers
            self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

            # Initialize state variables
            self.waypoint_array_in_map: np.ndarray = np.zeros((0, 8))
            self.speed_now: Optional[float] = None
            self.position_in_map: Optional[np.ndarray] = None
            self.position_in_map_frenet: Optional[np.ndarray] = None
            self.acc_now: np.ndarray = np.zeros(10)
            self.waypoint_safety_counter: int = 0
            self.rate: int = 40  # Hz

            # State lock
            self.state_lock = Lock()

            # Timing measurement for control loop (counter-based)
            self.timing_log_counter = 0

            # Initialize controller based on mode
            if self.mode == "MAP":
                self.get_logger().info("Initializing MAP controller")
                self.init_map_controller()
                self.controller = self.map_controller
            elif self.mode == "PP":
                self.get_logger().info("Initializing PP controller")
                self.init_pp_controller()
                self.controller = self.pp_controller
            elif self.mode == "AUG":
                self.get_logger().info("Initializing AUG controller")
                self.init_aug_controller()
                self.controller = self.aug_controller
            else:
                self.get_logger().error(f"Invalid mode: {self.mode}")
                return False

            # Initialize subscribers
            self.sub_local_waypoints = self.create_subscription(
                WpntArray,
                '/local_waypoints',
                self.local_waypoint_cb,
                10
            )
            self.sub_car_state = self.create_subscription(
                Odometry,
                '/odom',
                self.car_state_cb,
                10
            )
            self.sub_car_state_frenet = self.create_subscription(
                Odometry,
                '/frenet/odom',
                self.car_state_frenet_cb,
                10
            )
            self.sub_imu = self.create_subscription(
                Imu,
                '/sensors/imu/raw',
                self.imu_cb,
                10
            )

            # Initialize parameter event handler with delay
            self.param_handler: Optional[ParameterEventHandler] = None
            self.param_init_timer = self.create_timer(
                0.01,  # 10ms delay
                self._init_parameter_handler
            )

            # Initialize main control timer
            period = 1.0 / float(self.rate)
            self.timer = self.create_timer(period, self.control_loop)

            self.get_logger().info("Controller ready")
            return True

        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
            traceback.print_exc()
            return False

    def _init_parameter_handler(self):
        """Initialize parameter event handler (called once with delay)."""
        self.param_init_timer.cancel()
        
        qos = QoSProfile(depth=10)
        self.param_handler = ParameterEventHandler(self, qos)
        self.param_handler.add_parameter_event_callback(self.on_parameter_event)
        
        self.get_logger().info("✓ ParameterEventHandler initialized")

    def init_map_controller(self):
        """Initialize MAP controller with parameters from YAML."""
        self.get_logger().info("Initializing MAP controller...")

        # Load L1 parameters from YAML
        l1_params_path = self.get_parameter('l1_params_path').value
        self.get_logger().info(f"Loading L1 parameters from: {l1_params_path}")
        self.declare_l1_dynamic_parameters_from_yaml(l1_params_path)

        # Initialize acceleration rolling buffer
        self.acc_now = np.zeros(10)
        self.get_logger().debug("Acceleration buffer initialized")

        # Create MAP controller with logging callbacks
        def log_info(msg: str):
            self.get_logger().info(f"[MAP] {msg}")

        def log_warn(msg: str):
            self.get_logger().warn(f"[MAP] {msg}")

        self.map_controller = MAP_Controller(
            self.get_parameter('t_clip_min').value,
            self.get_parameter('t_clip_max').value,
            self.get_parameter('m_l1').value,
            self.get_parameter('q_l1').value,
            self.get_parameter('speed_lookahead').value,
            self.get_parameter('lat_err_coeff').value,
            self.get_parameter('acc_scaler_for_steer').value,
            self.get_parameter('dec_scaler_for_steer').value,
            self.get_parameter('start_scale_speed').value,
            self.get_parameter('end_scale_speed').value,
            self.get_parameter('downscale_factor').value,
            self.get_parameter('speed_lookahead_for_steer').value,
            self.get_parameter('diff_threshold').value,
            self.get_parameter('deacc_gain').value,
            self.LUT_path,
            log_info,
            log_warn
        )

        self.get_logger().info("✓ MAP controller initialized successfully")

    def init_pp_controller(self):
        """Initialize PP controller with parameters from YAML."""
        self.get_logger().info("Initializing PP controller...")

        # Load PP parameters from YAML
        pp_params_path = self.get_parameter('pp_params_path').value
        self.get_logger().info(f"Loading PP parameters from: {pp_params_path}")
        self.declare_pp_dynamic_parameters_from_yaml(pp_params_path)

        # Initialize acceleration rolling buffer
        self.acc_now = np.zeros(10)
        self.get_logger().debug("Acceleration buffer initialized")

        # Create PP controller with logging callbacks
        def log_info(msg: str):
            self.get_logger().info(f"[PP] {msg}")

        def log_warn(msg: str):
            self.get_logger().warn(f"[PP] {msg}")

        self.pp_controller = PP_Controller(
            self.get_parameter('t_clip_min').value,
            self.get_parameter('t_clip_max').value,
            self.get_parameter('m_l1').value,
            self.get_parameter('q_l1').value,
            self.get_parameter('speed_lookahead').value,
            self.get_parameter('lat_err_coeff').value,
            self.get_parameter('acc_scaler_for_steer').value,
            self.get_parameter('dec_scaler_for_steer').value,
            self.get_parameter('start_scale_speed').value,
            self.get_parameter('end_scale_speed').value,
            self.get_parameter('downscale_factor').value,
            self.get_parameter('speed_lookahead_for_steer').value,
            self.get_parameter('diff_threshold').value,
            self.get_parameter('deacc_gain').value,
            self.LUT_path,
            log_info,
            log_warn
        )

        self.get_logger().info("✓ PP controller initialized successfully")

    def init_aug_controller(self):
        """Initialize AUG controller with parameters from YAML."""
        self.get_logger().info("Initializing AUG controller...")

        # Load AUG parameters from YAML
        aug_params_path = self.get_parameter('aug_params_path').value
        self.get_logger().info(f"Loading AUG parameters from: {aug_params_path}")
        self.declare_aug_dynamic_parameters_from_yaml(aug_params_path)

        # Initialize acceleration rolling buffer
        self.acc_now = np.zeros(10)
        self.get_logger().debug("Acceleration buffer initialized")

        # Create AUG controller with logging callbacks
        def log_info(msg: str):
            self.get_logger().info(f"[AUG] {msg}")

        def log_warn(msg: str):
            self.get_logger().warn(f"[AUG] {msg}")

        self.aug_controller = AUG_Controller(
            self.get_parameter('t_clip_min').value,
            self.get_parameter('t_clip_max').value,
            self.get_parameter('m_l1').value,
            self.get_parameter('q_l1').value,
            self.get_parameter('speed_lookahead').value,
            self.get_parameter('lat_err_coeff').value,
            self.get_parameter('acc_scaler_for_steer').value,
            self.get_parameter('dec_scaler_for_steer').value,
            self.get_parameter('start_scale_speed').value,
            self.get_parameter('end_scale_speed').value,
            self.get_parameter('downscale_factor').value,
            self.get_parameter('speed_lookahead_for_steer').value,
            self.get_parameter('diff_threshold').value,
            self.get_parameter('deacc_gain').value,
            self.LUT_path,
            self.get_parameter('Cf').value,
            self.get_parameter('Cr').value,
            self.get_parameter('L').value,
            self.get_parameter('lf').value,
            self.get_parameter('lr').value,
            self.get_parameter('m').value,
            self.get_parameter('kf').value,
            self.get_parameter('kr').value,
            log_info,
            log_warn
        )

        self.get_logger().info("✓ AUG controller initialized successfully")

    def declare_l1_dynamic_parameters_from_yaml(self, yaml_path: str):
        """Declare dynamic parameters from L1 YAML configuration."""
        with open(yaml_path, 'r') as f:
            root = yaml.safe_load(f)

        if 'crazy_controller' not in root or 'ros__parameters' not in root['crazy_controller']:
            raise RuntimeError("Invalid l1_params YAML: missing crazy_controller.ros__parameters")

        params = root['crazy_controller']['ros__parameters']

        def declare_double(name: str, default: float, range_from: float, range_to: float, step: float):
            descriptor = ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(
                    from_value=range_from,
                    to_value=range_to,
                    step=step
                )]
            )
            self.declare_parameter(name, default, descriptor)

        # Declare dynamic parameters
        declare_double('t_clip_min', params['t_clip_min'], 0.0, 1.5, 0.01)
        declare_double('t_clip_max', params['t_clip_max'], 0.0, 10.0, 0.01)
        declare_double('m_l1', params['m_l1'], 0.0, 1.0, 0.001)
        declare_double('q_l1', params['q_l1'], -1.0, 1.0, 0.001)
        declare_double('speed_lookahead', params['speed_lookahead'], 0.0, 1.0, 0.01)
        declare_double('lat_err_coeff', params['lat_err_coeff'], 0.0, 1.0, 0.01)
        declare_double('acc_scaler_for_steer', params['acc_scaler_for_steer'], 0.0, 1.5, 0.01)
        declare_double('dec_scaler_for_steer', params['dec_scaler_for_steer'], 0.0, 1.5, 0.01)
        declare_double('start_scale_speed', params['start_scale_speed'], 0.0, 10.0, 0.01)
        declare_double('end_scale_speed', params['end_scale_speed'], 0.0, 10.0, 0.01)
        declare_double('downscale_factor', params['downscale_factor'], 0.0, 0.5, 0.01)
        declare_double('speed_lookahead_for_steer', params['speed_lookahead_for_steer'], 0.0, 0.2, 0.01)
        declare_double('diff_threshold', params['diff_threshold'], 0.0, 20.0, 0.1)
        declare_double('deacc_gain', params['deacc_gain'], 0.0, 1.0, 0.01)

    def declare_pp_dynamic_parameters_from_yaml(self, yaml_path: str):
        """Declare dynamic parameters from PP YAML configuration."""
        with open(yaml_path, 'r') as f:
            root = yaml.safe_load(f)

        if 'crazy_controller' not in root or 'ros__parameters' not in root['crazy_controller']:
            raise RuntimeError("Invalid pp_params YAML: missing crazy_controller.ros__parameters")

        params = root['crazy_controller']['ros__parameters']

        def declare_double(name: str, default: float, range_from: float, range_to: float, step: float):
            descriptor = ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(
                    from_value=range_from,
                    to_value=range_to,
                    step=step
                )]
            )
            self.declare_parameter(name, default, descriptor)

        # Declare dynamic parameters
        declare_double('t_clip_min', params['t_clip_min'], 0.0, 1.5, 0.01)
        declare_double('t_clip_max', params['t_clip_max'], 0.0, 10.0, 0.01)
        declare_double('m_l1', params['m_l1'], 0.0, 1.0, 0.001)
        declare_double('q_l1', params['q_l1'], -1.0, 1.0, 0.001)
        declare_double('speed_lookahead', params['speed_lookahead'], 0.0, 1.0, 0.01)
        declare_double('lat_err_coeff', params['lat_err_coeff'], 0.0, 1.0, 0.01)
        declare_double('acc_scaler_for_steer', params['acc_scaler_for_steer'], 0.0, 1.5, 0.01)
        declare_double('dec_scaler_for_steer', params['dec_scaler_for_steer'], 0.0, 1.5, 0.01)
        declare_double('start_scale_speed', params['start_scale_speed'], 0.0, 10.0, 0.01)
        declare_double('end_scale_speed', params['end_scale_speed'], 0.0, 10.0, 0.01)
        declare_double('downscale_factor', params['downscale_factor'], 0.0, 0.5, 0.01)
        declare_double('speed_lookahead_for_steer', params['speed_lookahead_for_steer'], 0.0, 0.2, 0.01)
        declare_double('diff_threshold', params['diff_threshold'], 0.0, 20.0, 0.1)
        declare_double('deacc_gain', params['deacc_gain'], 0.0, 1.0, 0.01)

    def declare_aug_dynamic_parameters_from_yaml(self, yaml_path: str):
        """Declare dynamic parameters from AUG YAML configuration."""
        with open(yaml_path, 'r') as f:
            root = yaml.safe_load(f)

        if 'crazy_controller' not in root or 'ros__parameters' not in root['crazy_controller']:
            raise RuntimeError("Invalid aug_params YAML: missing crazy_controller.ros__parameters")

        params = root['crazy_controller']['ros__parameters']

        def declare_double(name: str, default: float, range_from: float, range_to: float, step: float):
            descriptor = ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(
                    from_value=range_from,
                    to_value=range_to,
                    step=step
                )]
            )
            self.declare_parameter(name, default, descriptor)

        # Declare dynamic parameters
        declare_double('t_clip_min', params['t_clip_min'], 0.0, 1.5, 0.01)
        declare_double('t_clip_max', params['t_clip_max'], 0.0, 10.0, 0.01)
        declare_double('m_l1', params['m_l1'], 0.0, 1.0, 0.001)
        declare_double('q_l1', params['q_l1'], -1.0, 1.0, 0.001)
        declare_double('speed_lookahead', params['speed_lookahead'], 0.0, 1.0, 0.01)
        declare_double('lat_err_coeff', params['lat_err_coeff'], 0.0, 1.0, 0.01)
        declare_double('acc_scaler_for_steer', params['acc_scaler_for_steer'], 0.0, 1.5, 0.01)
        declare_double('dec_scaler_for_steer', params['dec_scaler_for_steer'], 0.0, 1.5, 0.01)
        declare_double('start_scale_speed', params['start_scale_speed'], 0.0, 10.0, 0.01)
        declare_double('end_scale_speed', params['end_scale_speed'], 0.0, 10.0, 0.01)
        declare_double('downscale_factor', params['downscale_factor'], 0.0, 0.5, 0.01)
        declare_double('speed_lookahead_for_steer', params['speed_lookahead_for_steer'], 0.0, 0.2, 0.01)
        declare_double('diff_threshold', params['diff_threshold'], 0.0, 20.0, 0.1)
        declare_double('deacc_gain', params['deacc_gain'], 0.0, 1.0, 0.01)

        # Declare vehicle parameters
        declare_double('Cf', params['Cf'], 0.0, 200.0, 0.001)
        declare_double('Cr', params['Cr'], 0.0, 200.0, 0.001)
        declare_double('L', params['L'], 0.0, 1.0, 0.0001)
        declare_double('lf', params['lf'], 0.0, 1.0, 0.00001)
        declare_double('lr', params['lr'], 0.0, 1.0, 0.00001)
        declare_double('m', params['m'], 0.0, 20.0, 0.01)

        # Declare tire stiffness adjustment coefficients
        declare_double('kf', params['kf'], 0.0, 0.01, 0.00001)
        declare_double('kr', params['kr'], 0.0, 0.01, 0.00001)

    def wait_for_messages(self):
        """Wait for all required messages before starting control loop."""
        self.get_logger().info("Controller waiting for messages...")
        waypoint_array_received = False
        car_state_received = False

        start_time = self.get_clock().now()
        while rclpy.ok() and (not waypoint_array_received or not car_state_received):
            rclpy.spin_once(self, timeout_sec=0.005)

            if self.waypoint_array_in_map.shape[0] > 0 and not waypoint_array_received:
                self.get_logger().info(
                    f"✓ Received waypoint array ({self.waypoint_array_in_map.shape[0]} waypoints)"
                )
                waypoint_array_received = True

            if (self.speed_now is not None and self.position_in_map is not None and
                self.position_in_map_frenet is not None and not car_state_received):
                self.get_logger().info(
                    f"✓ Received car state: pos=({self.position_in_map[0]:.2f},{self.position_in_map[1]:.2f}), "
                    f"speed={self.speed_now:.2f}, frenet=({self.position_in_map_frenet[0]:.2f},{self.position_in_map_frenet[1]:.2f})"
                )
                car_state_received = True

        self.get_logger().info("✓ All required messages received. Controller ready to start!")

    def control_loop(self):
        """Main control loop - called at fixed rate."""
        import time
        
        # Start timing measurement
        loop_start_time = time.perf_counter()
        
        # Check if shutdown was requested
        if self.shutdown_requested:
            self.publish_stop_command()
            return

        # Update position from TF at control loop frequency for real-time accuracy
        if not self.update_position_from_tf():
            return  # Skip this cycle if TF lookup fails

        # Check if all required data is available
        if (self.speed_now is None or self.position_in_map is None or
            self.position_in_map_frenet is None or self.waypoint_array_in_map.shape[0] == 0):
            return

        try:
            # Call appropriate control cycle
            if self.mode == "MAP":
                speed, steer = self.map_cycle()
            elif self.mode == "PP":
                speed, steer = self.pp_cycle()
            elif self.mode == "AUG":
                speed, steer = self.aug_cycle()
            else:
                return

            # Create and publish drive message
            ack = AckermannDriveStamped()
            now = self.get_clock().now()
            ack.header.stamp = now.to_msg()
            ack.header.frame_id = 'base_link'
            ack.drive.steering_angle = float(steer)
            ack.drive.speed = float(speed)

            self.drive_pub.publish(ack)

            # End timing measurement and log every 5 seconds (200 cycles at 40Hz)
            loop_end_time = time.perf_counter()
            duration_ms = (loop_end_time - loop_start_time) * 1000.0
            
            self.timing_log_counter += 1
            if self.timing_log_counter >= 200:  # 5 seconds at 40Hz
                self.get_logger().info(f"[{self.mode}] Control loop execution time: {duration_ms:.3f} ms")
                self.timing_log_counter = 0

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            traceback.print_exc()

    def map_cycle(self) -> Tuple[float, float]:
        """Execute MAP control cycle."""
        with self.state_lock:
            res = self.map_controller.main_loop(
                self.position_in_map,
                self.waypoint_array_in_map,
                self.speed_now,
                np.array([self.position_in_map_frenet[0], self.position_in_map_frenet[1]]),
                self.acc_now
            )

        self.waypoint_safety_counter += 1
        if self.waypoint_safety_counter >= self.rate * 5:  # 5 second timeout
            self.get_logger().warn("No fresh local waypoints. STOPPING!!")
            res.speed = 0.0
            res.steering_angle = 0.0

        return res.speed, res.steering_angle

    def pp_cycle(self) -> Tuple[float, float]:
        """Execute PP control cycle."""
        # import time
        
        with self.state_lock:
            res = self.pp_controller.main_loop(
                self.position_in_map,
                self.waypoint_array_in_map,
                self.speed_now,
                np.array([self.position_in_map_frenet[0], self.position_in_map_frenet[1]]),
                self.acc_now
            )

        # Artificial delay to test impact of computation time on control performance
        # time.sleep(0.020)  # 20ms delay

        self.waypoint_safety_counter += 1
        if self.waypoint_safety_counter >= self.rate * 5:  # 5 second timeout
            self.get_logger().warn("No fresh local waypoints. STOPPING!!")
            res.speed = 0.0
            res.steering_angle = 0.0

        return res.speed, res.steering_angle

    def aug_cycle(self) -> Tuple[float, float]:
        """Execute AUG control cycle."""
        # import time

        with self.state_lock:
            res = self.aug_controller.main_loop(
                self.position_in_map,
                self.waypoint_array_in_map,
                self.speed_now,
                np.array([self.position_in_map_frenet[0], self.position_in_map_frenet[1]]),
                self.acc_now
            )

        # Artificial delay to test impact of computation time on control performance
        # time.sleep(0.020)  # 20ms delay

        self.waypoint_safety_counter += 1
        if self.waypoint_safety_counter >= self.rate * 5:  # 5 second timeout
            self.get_logger().warn("No fresh local waypoints. STOPPING!!")
            res.speed = 0.0
            res.steering_angle = 0.0

        return res.speed, res.steering_angle

    # ========== Callback Functions ==========

    def local_waypoint_cb(self, msg: WpntArray):
        """Process incoming waypoint array."""
        N = len(msg.wpnts)
        if N <= 0:
            return

        # Convert waypoint array to matrix format [x, y, speed, ratio, s, kappa, psi, ax]
        with self.state_lock:
            self.waypoint_array_in_map = np.zeros((N, 8))
            for i, w in enumerate(msg.wpnts):
                ratio = (min(w.d_left, w.d_right) / (w.d_right + w.d_left)
                         if (w.d_left + w.d_right) != 0.0 else 0.0)
                
                self.waypoint_array_in_map[i, 0] = w.x_m
                self.waypoint_array_in_map[i, 1] = w.y_m
                self.waypoint_array_in_map[i, 2] = w.vx_mps
                self.waypoint_array_in_map[i, 3] = ratio
                self.waypoint_array_in_map[i, 4] = w.s_m
                self.waypoint_array_in_map[i, 5] = w.kappa_radpm
                self.waypoint_array_in_map[i, 6] = w.psi_rad
                self.waypoint_array_in_map[i, 7] = w.ax_mps2

            self.waypoint_safety_counter = 0

    def update_position_from_tf(self) -> bool:
        """Update position from TF transform. Returns True on success."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_link_frame,
                rclpy.time.Time()  # Get latest available transform
            )

            # Extract position
            x = transform.transform.translation.x
            y = transform.transform.translation.y

            # Extract orientation (yaw)
            q = transform.transform.rotation
            # Convert quaternion to yaw
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            with self.state_lock:
                self.position_in_map = np.array([x, y, yaw])

            return True

        except TransformException as ex:
            self.get_logger().warn(
                f"Could not get {self.map_frame}->{self.base_link_frame} transform: {ex}",
                throttle_duration_sec=2.0
            )
            return False

    def car_state_cb(self, msg: Odometry):
        """Process odometry message - only update velocity."""
        # Get velocity from odom message (position updated from TF in control loop)
        with self.state_lock:
            self.speed_now = msg.twist.twist.linear.x

    def car_state_frenet_cb(self, msg: Odometry):
        """Process Frenet frame odometry."""
        s = msg.pose.pose.position.x
        d = msg.pose.pose.position.y
        vs = msg.twist.twist.linear.x
        vd = msg.twist.twist.linear.y

        with self.state_lock:
            self.position_in_map_frenet = np.array([s, d, vs, vd])

    def imu_cb(self, msg: Imu):
        """Process IMU message - extract x-axis acceleration."""
        acc_x = msg.linear_acceleration.x

        # Maintain a sliding window of the most recent 10 acceleration values
        window_size = 10

        with self.state_lock:
            if len(self.acc_now) < window_size:
                # Append new value
                self.acc_now = np.append(self.acc_now, acc_x)
            else:
                # Shift left and add new value at the end
                self.acc_now[:-1] = self.acc_now[1:]
                self.acc_now[-1] = acc_x

    def on_parameter_event(self, event: ParameterEvent):
        """Handle parameter change events."""
        if event.node != '/controller_manager':
            return

        if self.mode == "MAP" and hasattr(self, 'map_controller'):
            # Update MAP controller parameters
            self.map_controller.set_t_clip_min(self.get_parameter('t_clip_min').value)
            self.map_controller.set_t_clip_max(self.get_parameter('t_clip_max').value)
            self.map_controller.set_m_l1(self.get_parameter('m_l1').value)
            self.map_controller.set_q_l1(self.get_parameter('q_l1').value)
            self.map_controller.set_speed_lookahead(self.get_parameter('speed_lookahead').value)
            self.map_controller.set_lat_err_coeff(self.get_parameter('lat_err_coeff').value)
            self.map_controller.set_acc_scaler_for_steer(self.get_parameter('acc_scaler_for_steer').value)
            self.map_controller.set_dec_scaler_for_steer(self.get_parameter('dec_scaler_for_steer').value)
            self.map_controller.set_start_scale_speed(self.get_parameter('start_scale_speed').value)
            self.map_controller.set_end_scale_speed(self.get_parameter('end_scale_speed').value)
            self.map_controller.set_downscale_factor(self.get_parameter('downscale_factor').value)
            self.map_controller.set_speed_lookahead_for_steer(self.get_parameter('speed_lookahead_for_steer').value)
            self.map_controller.set_diff_threshold(self.get_parameter('diff_threshold').value)
            self.map_controller.set_deacc_gain(self.get_parameter('deacc_gain').value)
            
            self.get_logger().info("Updated MAP parameters")

        elif self.mode == "PP" and hasattr(self, 'pp_controller'):
            # Update PP controller parameters
            self.pp_controller.set_t_clip_min(self.get_parameter('t_clip_min').value)
            self.pp_controller.set_t_clip_max(self.get_parameter('t_clip_max').value)
            self.pp_controller.set_m_l1(self.get_parameter('m_l1').value)
            self.pp_controller.set_q_l1(self.get_parameter('q_l1').value)
            self.pp_controller.set_speed_lookahead(self.get_parameter('speed_lookahead').value)
            self.pp_controller.set_lat_err_coeff(self.get_parameter('lat_err_coeff').value)
            self.pp_controller.set_acc_scaler_for_steer(self.get_parameter('acc_scaler_for_steer').value)
            self.pp_controller.set_dec_scaler_for_steer(self.get_parameter('dec_scaler_for_steer').value)
            self.pp_controller.set_start_scale_speed(self.get_parameter('start_scale_speed').value)
            self.pp_controller.set_end_scale_speed(self.get_parameter('end_scale_speed').value)
            self.pp_controller.set_downscale_factor(self.get_parameter('downscale_factor').value)
            self.pp_controller.set_speed_lookahead_for_steer(self.get_parameter('speed_lookahead_for_steer').value)

            self.get_logger().info("Updated PP parameters")

        elif self.mode == "AUG" and hasattr(self, 'aug_controller'):
            # Update AUG controller parameters
            self.aug_controller.set_t_clip_min(self.get_parameter('t_clip_min').value)
            self.aug_controller.set_t_clip_max(self.get_parameter('t_clip_max').value)
            self.aug_controller.set_m_l1(self.get_parameter('m_l1').value)
            self.aug_controller.set_q_l1(self.get_parameter('q_l1').value)
            self.aug_controller.set_speed_lookahead(self.get_parameter('speed_lookahead').value)
            self.aug_controller.set_lat_err_coeff(self.get_parameter('lat_err_coeff').value)
            self.aug_controller.set_acc_scaler_for_steer(self.get_parameter('acc_scaler_for_steer').value)
            self.aug_controller.set_dec_scaler_for_steer(self.get_parameter('dec_scaler_for_steer').value)
            self.aug_controller.set_start_scale_speed(self.get_parameter('start_scale_speed').value)
            self.aug_controller.set_end_scale_speed(self.get_parameter('end_scale_speed').value)
            self.aug_controller.set_downscale_factor(self.get_parameter('downscale_factor').value)
            self.aug_controller.set_speed_lookahead_for_steer(self.get_parameter('speed_lookahead_for_steer').value)
            self.aug_controller.set_diff_threshold(self.get_parameter('diff_threshold').value)
            self.aug_controller.set_deacc_gain(self.get_parameter('deacc_gain').value)
            self.aug_controller.set_kf(self.get_parameter('kf').value)
            self.aug_controller.set_kr(self.get_parameter('kr').value)
            self.aug_controller.set_Cf(self.get_parameter('Cf').value)
            self.aug_controller.set_Cr(self.get_parameter('Cr').value)

            self.get_logger().info("Updated AUG parameters")

    # ========== Emergency Stop Functions ==========

    def publish_stop_command(self):
        """Publish emergency stop command."""
        ack = AckermannDriveStamped()
        ack.header.stamp = self.get_clock().now().to_msg()
        ack.header.frame_id = 'base_link'
        ack.drive.speed = 0.0
        ack.drive.steering_angle = 0.0
        ack.drive.acceleration = -5.0  # Emergency brake
        ack.drive.jerk = 0.0
        ack.drive.steering_angle_velocity = 0.0

        self.drive_pub.publish(ack)

        self.get_logger().warn(
            "Publishing stop command - Vehicle stopped for safety",
            throttle_duration_sec=1.0
        )

    def shutdown_handler(self):
        """Handle shutdown signal - stop vehicle safely."""
        self.get_logger().warn("Shutdown signal received - stopping vehicle safely")
        self.shutdown_requested = True

        # Publish multiple stop commands to ensure vehicle stops
        for _ in range(5):
            self.publish_stop_command()
            rclpy.spin_once(self, timeout_sec=0.05)


# Global node instance for signal handler
g_node = None


def signal_handler(sig, frame):
    """Handle OS signals for graceful shutdown."""
    global g_node
    if g_node is not None:
        g_node.shutdown_handler()
    sys.exit(0)


def main(args=None):
    """Main entry point for controller node."""
    rclpy.init(args=args)

    global g_node
    g_node = ControllerManager()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    g_node.get_logger().info("Starting Controller Node with graceful shutdown")

    # Wait for required messages before starting control loop
    g_node.wait_for_messages()

    try:
        rclpy.spin(g_node)
    except KeyboardInterrupt:
        pass
    finally:
        g_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
