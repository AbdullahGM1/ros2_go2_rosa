#!/usr/bin/env python3
"""Main entry point for the Unitree Go2 ROSA Agent."""

import rclpy
import threading
import asyncio
import time
import signal
import sys
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rosa import ROSA

from .cli.interface import RichGo2CLI
from .prompts.system_prompts import create_system_prompts
from .tools.robot_tools import RobotTools
from .llm.model import initialize_llm


class Go2AgentNode(Node):
    """Main ROS2 node for Unitree Go2 ROSA Agent."""
    
    def __init__(self):
        super().__init__('go2_agent_node')
        
        # ============================= PARAMETERS =============================
        self._declare_and_get_parameters()
        
        # ============================= INITIALIZE NODE =============================
        self._initialize_node()
        
        # ============================= SETUP AGENT =============================
        self.setup_agent()
        
        self.get_logger().info("ROSA Go2 Agent is ready. Type a command:")
    
    def _declare_and_get_parameters(self):
        """Declare and get all ROS parameters."""
        # Declare parameters with default values
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom/ground_truth')
        self.declare_parameter('camera_topic', '/go2_rgb')
        self.declare_parameter('linear_speed', 2.0)
        self.declare_parameter('angular_speed', 0.8)
        self.declare_parameter('control_rate', 20.0)  # Hz
        self.declare_parameter('position_tolerance', 0.05)  # meters
        self.declare_parameter('angle_tolerance', 0.05)  # radians (~3 degrees)
        self.declare_parameter('llm_model', 'qwen3:8b')
        
        # Get parameters
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.control_rate = self.get_parameter('control_rate').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.angle_tolerance = self.get_parameter('angle_tolerance').value
        self.llm_model = self.get_parameter('llm_model').value
        
        # Log parameters
        self.get_logger().info(f"Parameters: cmd_vel_topic={self.cmd_vel_topic}, "
                               f"odom_topic={self.odom_topic}, "
                               f"camera_topic={self.camera_topic}")
        self.get_logger().info(f"Control parameters: linear_speed={self.linear_speed}, "
                               f"angular_speed={self.angular_speed}, "
                               f"control_rate={self.control_rate}, "
                               f"position_tolerance={self.position_tolerance}, "
                               f"angle_tolerance={self.angle_tolerance}")
        self.get_logger().info(f"Using LLM model: {self.llm_model}")
    
    def _initialize_node(self):
        """Initialize node components."""
        # Create QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Initialize publishers with QoS
        self.publisher_ = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            qos_profile
        )
        
        # Initialize CV Bridge for image processing
        self.bridge = CvBridge()
        
        # Camera thread control
        self.camera_active = False
        self.camera_thread = None
        self.camera_lock = threading.Lock()
        
        # For graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle SIGINT (Ctrl+C) gracefully."""
        self.get_logger().info("Shutdown signal received, cleaning up...")
        self.running = False
        
        # Stop camera if active
        with self.camera_lock:
            if self.camera_active:
                self.camera_active = False
        
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        
        # Stop robot movement
        twist = Twist()
        self.publisher_.publish(twist)
        
        # Perform node shutdown
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    
    def setup_agent(self):
        """Setup the ROSA agent with LLM and tools."""
        # Initialize LLM
        local_llm = initialize_llm(self.llm_model)
        
        # Create tools
        robot_tools = RobotTools(self)
        tools = robot_tools.create_tools()
        
        # Create prompts
        prompts = create_system_prompts()
        
        # Initialize ROSA
        self.agent = ROSA(
            ros_version=2,
            llm=local_llm,
            tools=tools,
            prompts=prompts
        )


def main(args=None):
    """Main function to run the Go2AgentNode with rich CLI."""
    rclpy.init(args=args)
    
    # Create the node
    node = Go2AgentNode()
    
    # Create a separate thread for processing ROS callbacks
    def spin_thread():
        while node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    
    ros_thread = threading.Thread(target=spin_thread)
    ros_thread.daemon = True
    ros_thread.start()
    
    # Create and run the rich CLI
    cli = RichGo2CLI(node)
    try:
        asyncio.run(cli.run())
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup
        node.running = False
        if ros_thread.is_alive():
            ros_thread.join(timeout=1.0)
        
        # Cleanup ROS
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()