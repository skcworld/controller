"""
Parameter event handler for monitoring ROS2 dynamic parameter changes.
Wrapper around rclpy parameter event callbacks.
"""

from typing import Callable, Optional
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterEvent
from rclpy.qos import QoSProfile


class ParameterEventHandler:
    """
    Monitors parameter events and triggers callbacks when parameters change.
    Wraps rclpy's parameter event subscription functionality.
    """

    def __init__(self, node: Node, qos: QoSProfile):
        """
        Initialize parameter event handler.
        
        Args:
            node: ROS2 node instance
            qos: QoS profile for parameter events subscription
        """
        self.node = node
        self.qos = qos
        self.callbacks: list = []
        
        # Subscribe to parameter events
        self.subscription = self.node.create_subscription(
            ParameterEvent,
            '/parameter_events',
            self._parameter_event_callback,
            qos
        )

    def add_parameter_event_callback(self, callback: Callable[[ParameterEvent], None]) -> int:
        """
        Register a callback for parameter events.
        
        Args:
            callback: Function to call when parameter event occurs
        
        Returns:
            Callback handle (index in callback list)
        """
        self.callbacks.append(callback)
        return len(self.callbacks) - 1

    def remove_parameter_event_callback(self, handle: int) -> None:
        """
        Remove a registered callback.
        
        Args:
            handle: Callback handle returned by add_parameter_event_callback
        """
        if 0 <= handle < len(self.callbacks):
            self.callbacks[handle] = None

    def _parameter_event_callback(self, event: ParameterEvent) -> None:
        """
        Internal callback that dispatches to registered callbacks.
        
        Args:
            event: Parameter event message
        """
        for callback in self.callbacks:
            if callback is not None:
                callback(event)
