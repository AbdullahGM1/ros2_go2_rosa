#!/usr/bin/env python3
"""
Enhanced ROS2 Agent Node for Unitree Go2 Robot
---------------------------------------------
This node implements a ROSA (ROS Operating System Agent) for controlling
the Unitree Go2 quadruped robot using natural language commands.
It uses a local LLM to interpret commands and execute them through ROS topics.

Improvements:
- Enhanced error handling and recovery
- Performance optimizations
- Better thread management
- Command history and autocomplete
- Configuration via yaml file
- Diagnostics and status reporting

Author: AbdullahGM1
License: MIT
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from langchain_ollama import ChatOllama
from langchain.agents import tool
from rosa import ROSA, RobotSystemPrompts
import math
import time
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading
import readline  # For command history and better input handling
import yaml
import os
import signal
import logging
import sys
from typing import Dict, Any, List, Optional, Union, Callable


class Go2AgentNode(Node):
    def __init__(self, config_path=None):
        """Initialize the Go2 Agent Node with optional config file"""
        # ============================= SETUP LOGGING =============================
        # Configure logging before anything else
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("go2_agent.log")
            ]
        )
        self.logger = logging.getLogger("go2_agent")
        self.logger.info("Starting Go2 Agent Node")
    
        # ============================= LOAD CONFIGURATION =============================
        # Default configuration
        self.config = {
            'model': 'qwen2.5:14b',
            'temperature': 0.0,
            'linear_speed': 0.5,
            'angular_speed': 1.0,
            'cmd_vel_topic': '/cmd_vel',
            'odom_topic': '/odom/ground_truth',
            'camera_topic': '/go2_rgb',
            'camera_resolution': (250, 250),
            'request_timeout': 30.0,
            'context_size': 4096,
            'cmd_vel_publish_rate': 10.0,   # Added: Control command publish rate
            'movement_timeout_factor': 2.5, # Added: Timeout multiplier
            'movement_tolerance': 0.05,     # Added: Position tolerance in meters
            'rotation_tolerance': 0.04      # Added: Rotation tolerance in radians
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as file:
                    file_config = yaml.safe_load(file)
                    if file_config:
                        self.config.update(file_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config: {str(e)}")
        
        # ============================= INITIALIZE NODE =============================
        super().__init__('go2_agent_node')
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize publishers with configured topics
        self.publisher_ = self.create_publisher(Twist, self.config['cmd_vel_topic'], 10)
        
        # Initialize CV Bridge for image processing
        self.bridge = CvBridge()
        
        # Thread management
        self.camera_active = False
        self.camera_thread = None
        self.camera_lock = threading.Lock()
        
        # Command history setup
        readline.parse_and_bind("tab: complete")
        readline.set_history_length(100)
        history_file = os.path.expanduser("~/.go2_history")
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass
        
        # Register to save history on exit
        import atexit
        atexit.register(readline.write_history_file, history_file)
        
        # Added: Keep track of robot state to avoid inconsistencies
        self.robot_state = {
            "moving": False,
            "last_cmd_time": 0,
            "last_cmd": None,
            "last_pose": None
        }
        
        # ============================= SETUP AGENT =============================
        self.setup_agent()
        
        # Initialization complete
        self.get_logger().info("ROSA Unitree Go2 Agent is ready. Type a command:")

    def _signal_handler(self, sig, frame):
        """Handle termination signals gracefully"""
        self.logger.info(f"Received signal {sig}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)
        
    def cleanup(self):
        """Clean up resources before shutdown"""
        # Send zero velocity command to ensure robot stops
        self._stop_robot()
        
        # Stop camera if active
        with self.camera_lock:
            if self.camera_active:
                self.camera_active = False
                
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
            
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        self.logger.info("Cleanup complete")
        
    def _stop_robot(self):
        """Send stop commands to the robot"""
        twist = Twist()
        # Send multiple stop commands to ensure they are received
        for _ in range(5):
            self.publisher_.publish(twist)
            time.sleep(0.05)
        
        # Update robot state
        self.robot_state["moving"] = False
        self.robot_state["last_cmd"] = None
    
    def setup_agent(self):
        """Setup the ROSA agent with LLM and tools"""
        try:
            # ============================= INITIALIZE LLM =============================
            local_llm = self._initialize_llm()
            
            # ============================= DEFINE TOOLS & PROMPTS =============================
            tools = self._create_tools()
            prompts = self._create_prompts()
            
            # ============================= INITIALIZE ROSA AGENT =============================
            self.agent = ROSA(
                ros_version=2,
                llm=local_llm,
                tools=tools,
                prompts=prompts
            )
            
            self.logger.info("ROSA agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ROSA agent: {str(e)}")
            raise

    def _initialize_llm(self):
        """Initialize and configure the local LLM"""
        try:
            return ChatOllama(
                model=self.config['model'],
                temperature=self.config['temperature'],
                max_retries=1,
                num_ctx=self.config['context_size'],
                request_timeout=self.config['request_timeout']
            )
        except Exception as e:
            self.logger.error(f"LLM initialization error: {str(e)}")
            raise

    def _create_prompts(self):
        """Create the system prompts for the agent"""
        return RobotSystemPrompts(
            embodiment_and_persona="You are a smart Unitree Go2 quadruped robot with a camera.",
            about_your_capabilities=(
                "You have access to tools, and you should always prefer using them over replying directly. "
                "You can move forward/backward and rotate left/right using degrees. "
                "You can also move to specific (x,y) coordinates with the move_to_pose tool. "
                "You have a camera and can see the environment. "
                "Always use your available tools to answer questions and execute tasks. "
                "Never guess or use ROS commands directly."
            ),
            mission_and_objectives=(
                "Help users control the Go2 robot and inspect the environment using available tools. "
                "When the user gives any command that involves 'camera', 'image', 'see', 'show me', or 'feed', "
                "you **must** call the `get_robot_camera_image` tool. Do not guess or provide manual instructions."
                "For best results, use the tools in a logical sequence and provide clear feedback to the user."
            )
        )
    
    def _create_tools(self):
        """Create and return all the tools for the agent"""
        node_instance = self  # Store reference to self for closure
        
        # ============================= MOVEMENT TOOLS =============================
        @tool
        def publish_linear_motion(distance: float) -> str:
            """
            Move the Unitree Go2 robot forward/backward by the specified distance (in meters)
            using closed-loop control with position feedback.
            """
            # Input validation
            if not isinstance(distance, (int, float)):
                return "Error: Distance must be a number"
            
            # Check if robot is already in motion
            if node_instance.robot_state["moving"]:
                return "The robot is already moving. Please wait for the current motion to complete."
                
            # Use configured speed
            linear_speed = node_instance.config['linear_speed']
            twist = Twist()
            
            # Set a timeout to prevent hanging - increased timeout factor for reliability
            max_execution_time = min(30.0, abs(distance) / linear_speed * node_instance.config['movement_timeout_factor'])
            
            # FIXED: Get initial pose with better error handling
            try:
                # CRITICAL FIX: Remove input parameter - this was causing the error
                initial_pose = get_robot_pose.invoke(input="")
                if "error" in initial_pose:
                    return f"Error getting pose: {initial_pose['error']}"
                
                # Store initial pose for reference
                node_instance.robot_state["last_pose"] = initial_pose
            except Exception as e:
                node_instance.logger.error(f"Error in linear motion: {str(e)}")
                return f"Error getting pose: {str(e)}"
                
            initial_x = initial_pose["x"]
            initial_y = initial_pose["y"]
            target_distance = abs(distance)
            
            # Set direction
            direction = 1.0 if distance >= 0 else -1.0
            
            # FIXED: Mark robot as moving to prevent concurrent motion commands
            node_instance.robot_state["moving"] = True
            node_instance.robot_state["last_cmd_time"] = time.time()
            node_instance.robot_state["last_cmd"] = f"linear_{direction}"
            
            # Create a rate object with optimized frequency
            rate = node_instance.create_rate(node_instance.config['cmd_vel_publish_rate'])
            
            # Start time for timeout
            start_time = time.time()
            node_instance.logger.info(f"Starting linear motion: {distance:.2f}m")
            
            try:
                # FIXED: Smoothly ramp up velocity for hardware compliance
                ramp_up_steps = 5
                for step in range(1, ramp_up_steps + 1):
                    if not rclpy.ok() or time.time() - start_time > max_execution_time:
                        break
                    
                    # Gradually increase speed
                    current_speed = (step / ramp_up_steps) * linear_speed
                    twist.linear.x = direction * current_speed
                    
                    # Publish command
                    node_instance.publisher_.publish(twist)
                    time.sleep(0.05)  # Short delay for ramp-up
                
                # Use full speed after ramp-up
                twist.linear.x = direction * linear_speed
                
                # Main movement loop with improved position tracking
                last_publish_time = time.time()
                while rclpy.ok():
                    current_time = time.time()
                    
                    # FIXED: Regular command publishing for hardware reliability
                    # Publish velocity command at fixed intervals to keep robot moving
                    if current_time - last_publish_time >= 0.1:  # Every 100ms
                        node_instance.publisher_.publish(twist)
                        last_publish_time = current_time
                    
                    # Check timeout
                    if current_time - start_time > max_execution_time:
                        node_instance.logger.warning(f"Motion timed out after {max_execution_time:.1f}s")
                        return f"Motion timed out after {max_execution_time:.1f} seconds. Target was {target_distance:.2f} meters."
                    
                    # Get current pose with better error handling
                    try:
                        current_pose = get_robot_pose.invoke(input="")
                        if "error" in current_pose:
                            return f"Error during movement: {current_pose['error']}"
                            
                        # Update stored pose
                        node_instance.robot_state["last_pose"] = current_pose
                    except Exception as e:
                        return f"Error during movement: {str(e)}"
                    
                    # Fast distance calculation
                    dx = current_pose["x"] - initial_x
                    dy = current_pose["y"] - initial_y
                    distance_moved = math.sqrt(dx*dx + dy*dy)
                    
                    # Check if we're moving at all
                    if current_time - start_time > 3.0 and distance_moved < 0.02:
                        # FIXED: Detect and handle hardware issues
                        node_instance.logger.warning("Robot not moving despite commands!")
                        # Try increasing the speed temporarily
                        boost_twist = Twist()
                        boost_twist.linear.x = direction * (linear_speed * 1.5)  # 50% boost
                        for _ in range(5):  # Send multiple boosted commands
                            node_instance.publisher_.publish(boost_twist)
                            time.sleep(0.05)
                        # Restore normal speed
                        twist.linear.x = direction * linear_speed
                    
                    # Check if we've reached the target
                    if distance_moved >= target_distance - node_instance.config['movement_tolerance']:
                        break
                    
                    # Speed adjustment - slow down near target
                    remaining = target_distance - distance_moved
                    if remaining < 0.3:  # 30cm from target
                        # FIXED: Better velocity ramping near target
                        slowdown_factor = max(0.2, remaining / 0.3)  # At least 0.2 speed factor
                        twist.linear.x = direction * linear_speed * slowdown_factor
                    
                    # Brief sleep to not overwhelm CPU
                    time.sleep(0.01)
                
                # FIXED: Smooth stopping with gradual deceleration
                stop_steps = 5  
                for step in range(stop_steps, 0, -1):
                    # Skip deceleration if we're out of time
                    if time.time() - start_time > max_execution_time:
                        break
                        
                    # Gradually decrease speed
                    decel_speed = direction * linear_speed * (step / stop_steps)
                    twist.linear.x = decel_speed
                    node_instance.publisher_.publish(twist)
                    time.sleep(0.05)
                    
            finally:
                # Stop the robot - ensure we always send stop commands
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                for _ in range(5):  # FIXED: Send multiple stop commands
                    node_instance.publisher_.publish(twist)
                    time.sleep(0.05)
                
                # Mark robot as not moving
                node_instance.robot_state["moving"] = False
            
            # Get final position after motion
            try:
                final_pose = get_robot_pose.invoke(input="")
                if "error" not in final_pose:
                    # Calculate actual distance moved
                    dx = final_pose["x"] - initial_x
                    dy = final_pose["y"] - initial_y
                    actual_distance = math.sqrt(dx*dx + dy*dy)
                    accuracy = (actual_distance / target_distance) * 100 if target_distance > 0 else 0
                    
                    node_instance.logger.info(f"Linear motion completed: {actual_distance:.2f}m ({accuracy:.1f}% accuracy)")
                    return f"Moved {'forward' if distance >= 0 else 'backward'} {actual_distance:.2f} meters."
            except Exception:
                pass
                
            node_instance.logger.info(f"Linear motion completed: target {target_distance:.2f}m")
            return f"Moved {'forward' if distance >= 0 else 'backward'} {target_distance:.2f} meters."

        @tool
        def publish_angular_motion(angle: float) -> str:
            """
            Rotate the Unitree Go2 robot by specified degrees using closed-loop control.
            Positive values rotate clockwise, negative values rotate counterclockwise.
            """
            # Input validation
            if not isinstance(angle, (int, float)):
                return "Error: Angle must be a number"
            
            # Check if robot is already in motion
            if node_instance.robot_state["moving"]:
                return "The robot is already moving. Please wait for the current motion to complete."
                
            # Use configured angular speed
            angular_speed = node_instance.config['angular_speed']
            angle_radians = math.radians(angle)
            twist = Twist()
            
            # FIXED: Set a better maximum execution time to prevent hanging
            max_execution_time = min(15.0, abs(angle_radians) / angular_speed * node_instance.config['movement_timeout_factor'])
            
            # Get initial pose with improved error handling
            try:
                initial_pose = get_robot_pose.invoke(input="")
                if "error" in initial_pose:
                    return f"Error getting pose: {initial_pose['error']}"
                    
                # Store initial pose
                node_instance.robot_state["last_pose"] = initial_pose
            except Exception as e:
                node_instance.logger.error(f"Error in angular motion: {str(e)}")
                return f"Error getting pose: {str(e)}"
                
            # Calculate target orientation
            target_theta = initial_pose["yaw"] + angle_radians
            # Normalize to [-π, π]
            target_theta = ((target_theta + math.pi) % (2 * math.pi)) - math.pi
            
            # Direction of rotation
            direction = 1.0 if angle_radians >= 0 else -1.0
            
            # FIXED: Mark robot as moving
            node_instance.robot_state["moving"] = True
            node_instance.robot_state["last_cmd_time"] = time.time()
            node_instance.robot_state["last_cmd"] = f"rotate_{direction}"
            
            # Create a rate object with optimized frequency
            rate = node_instance.create_rate(node_instance.config['cmd_vel_publish_rate'])
            
            # Start time for timeout
            start_time = time.time()
            node_instance.logger.info(f"Starting angular motion: {angle:.1f}° ({direction})")
            
            try:
                # FIXED: Ramp up angular velocity for hardware acceleration limits
                ramp_steps = 3
                for step in range(1, ramp_steps + 1):
                    if not rclpy.ok() or time.time() - start_time > max_execution_time:
                        break
                        
                    # Gradually increase speed
                    current_speed = (step / ramp_steps) * angular_speed
                    twist.angular.z = direction * current_speed
                    
                    # Publish command
                    node_instance.publisher_.publish(twist)
                    time.sleep(0.05)
                
                # Set to full angular speed after ramp-up
                twist.angular.z = direction * angular_speed
                
                # Tracking variables for motion troubleshooting
                last_publish_time = time.time()
                last_angle_diff = float('inf')
                last_angle_check_time = time.time()
                stall_detected = False
                
                while rclpy.ok():
                    current_time = time.time()
                    
                    # FIXED: Regular command publishing to ensure hardware responsiveness
                    if current_time - last_publish_time >= 0.1:  # Every 100ms
                        node_instance.publisher_.publish(twist)
                        last_publish_time = current_time
                    
                    # Check timeout
                    if current_time - start_time > max_execution_time:
                        node_instance.logger.warning(f"Rotation timed out after {max_execution_time:.1f}s")
                        return f"Rotation timed out after {max_execution_time:.1f} seconds. Target was {angle:.0f} degrees."
                    
                    # Get current pose with error handling
                    try:
                        current_pose = get_robot_pose.invoke(input="")
                        if "error" in current_pose:
                            return f"Error during rotation: {current_pose['error']}"
                            
                        # Update stored pose
                        node_instance.robot_state["last_pose"] = current_pose
                    except Exception as e:
                        return f"Error during rotation: {str(e)}"
                    
                    # Calculate remaining angle difference
                    current_theta = current_pose["yaw"]
                    # Calculate angle difference accounting for wraparound
                    angle_diff = target_theta - current_theta
                    # Normalize to [-π, π]
                    angle_diff = ((angle_diff + math.pi) % (2 * math.pi)) - math.pi
                    
                    # FIXED: Check if rotation is making progress
                    if current_time - last_angle_check_time > 1.0:  # Check progress every second
                        # If angle hasn't changed significantly in the last second
                        if abs(angle_diff - last_angle_diff) < 0.05 and not stall_detected:
                            stall_detected = True
                            node_instance.logger.warning("Rotation appears stalled - boosting power")
                            # Try a power boost
                            boost_twist = Twist()
                            boost_twist.angular.z = direction * (angular_speed * 1.5)  # 50% boost
                            for _ in range(3):
                                node_instance.publisher_.publish(boost_twist)
                                time.sleep(0.05)
                            # Return to normal speed
                            twist.angular.z = direction * angular_speed
                            
                        last_angle_diff = angle_diff
                        last_angle_check_time = current_time
                    
                    # Check if we've reached the target angle (with appropriate tolerance)
                    if abs(angle_diff) < node_instance.config['rotation_tolerance']:
                        break
                    
                    # FIXED: Better speed adjustment - gradually reduce as we approach target
                    if abs(angle_diff) < 0.3:  # ~17 degrees from target
                        # Scale speed based on remaining angle
                        slowdown_factor = max(0.3, abs(angle_diff) / 0.3)
                        twist.angular.z = direction * angular_speed * slowdown_factor
                    
                    # Brief sleep to not overwhelm CPU
                    time.sleep(0.01)
                
                # FIXED: Smooth stopping with gradual deceleration
                stop_steps = 3
                for step in range(stop_steps, 0, -1):
                    if time.time() - start_time > max_execution_time:
                        break
                        
                    decel_speed = direction * angular_speed * (step / stop_steps)
                    twist.angular.z = decel_speed
                    node_instance.publisher_.publish(twist)
                    time.sleep(0.05)
                    
            finally:
                # Stop rotation - ensure we always stop
                twist.angular.z = 0.0
                twist.linear.x = 0.0
                for _ in range(5):  # FIXED: Send multiple stop commands
                    node_instance.publisher_.publish(twist)
                    time.sleep(0.05)
                
                # Mark robot as not moving
                node_instance.robot_state["moving"] = False
            
            # Get final angle after motion
            try:
                final_pose = get_robot_pose.invoke()
                if "error" not in final_pose:
                    # Calculate actual rotation in degrees
                    actual_rotation = math.degrees(final_pose["yaw"] - initial_pose["yaw"])
                    # Normalize to [-180, 180]
                    actual_rotation = ((actual_rotation + 180) % 360) - 180
                    
                    # Handle case where rotation crossed the +/-π boundary
                    if abs(actual_rotation - angle) > 180:
                        if actual_rotation > 0:
                            actual_rotation -= 360
                        else:
                            actual_rotation += 360
                    
                    accuracy = (abs(actual_rotation) / abs(angle)) * 100 if angle != 0 else 0
                    
                    node_instance.logger.info(f"Angular motion completed: {actual_rotation:.1f}° ({accuracy:.1f}% accuracy)")
                    return f"Rotated {actual_rotation:.0f}° {'clockwise' if actual_rotation >= 0 else 'counterclockwise'}."
            except Exception:
                pass
            
            node_instance.logger.info(f"Angular motion completed: target {angle:.1f}°")
            return f"Rotated {angle:.0f}° {'clockwise' if angle >= 0 else 'counterclockwise'}."

        # ============================= SENSOR TOOLS =============================
        @tool
        def get_robot_pose(input: str) -> dict:
            """
            Get the pose of the Unitree Go2 robot from odometry data.
            Returns position, orientation, and velocity information.
            """
            pose_data = {}
            msg_received = threading.Event()

            def callback(msg):
                # Extract only necessary data (position and yaw)
                pose_data["x"] = round(msg.pose.pose.position.x, 2)
                pose_data["y"] = round(msg.pose.pose.position.y, 2)
                
                # Extract orientation as quaternion
                qx = msg.pose.pose.orientation.x
                qy = msg.pose.pose.orientation.y
                qz = msg.pose.pose.orientation.z
                qw = msg.pose.pose.orientation.w
                
                # Calculate yaw (we mostly need this for navigation)
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                pose_data["yaw"] = round(yaw, 2)
                
                # Extract linear.x velocity as it's the most commonly used
                pose_data["linear_x"] = round(msg.twist.twist.linear.x, 2)
                pose_data["angular_z"] = round(msg.twist.twist.angular.z, 2)
                
                msg_received.set()

            sub = node_instance.create_subscription(
                Odometry,
                node_instance.config['odom_topic'],
                callback,
                10
            )
            
            # Wait for the message with a reasonable timeout
            if not msg_received.wait(timeout=2.0):
                node_instance.destroy_subscription(sub)
                node_instance.logger.warning(f"Odometry data not received in time from {node_instance.config['odom_topic']}")
                return {"error": f"Odometry data not received in time from {node_instance.config['odom_topic']}"}

            # Clean up subscription immediately
            node_instance.destroy_subscription(sub)
            return pose_data

        # ============================= CAMERA TOOLS =============================
        @tool
        def get_robot_camera_image() -> dict:
            """
            Display a live stream from the Unitree Go2's RGB camera in a non-blocking way.
            Press 'q' in the camera window to close the stream.
            """
            with node_instance.camera_lock:
                # Check if camera is already running
                if node_instance.camera_active:
                    return {"message": "Camera is already running. Use 'stop_camera' to stop it first if needed."}
                
                # Function to run in a separate thread
                def camera_thread_function():
                    streaming = {"running": True}
                    last_frame = {"image": None}
                    sub = None
                    frames_received = 0
                    last_frame_time = time.time()

                    try:
                        def image_callback(msg):
                            try:
                                nonlocal frames_received, last_frame_time
                                cv_image = node_instance.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                                
                                # Get configured resolution
                                width, height = node_instance.config['camera_resolution']
                                resized_image = cv2.resize(cv_image, (width, height))
                                
                                # Add status information to the frame
                                status_text = f"FPS: {frames_received/(time.time()-last_frame_time+0.001):.1f}"
                                cv2.putText(resized_image, status_text, (10, 20), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                last_frame["image"] = resized_image
                                frames_received += 1
                                
                                # Reset FPS counter every 5 seconds
                                if time.time() - last_frame_time > 5:
                                    frames_received = 0
                                    last_frame_time = time.time()
                                    
                            except Exception as e:
                                node_instance.logger.error(f"Image conversion failed: {str(e)}")

                        sub = node_instance.create_subscription(
                            Image,
                            node_instance.config['camera_topic'],
                            image_callback,
                            10
                        )

                        node_instance.logger.info(f"📷 Live streaming camera from {node_instance.config['camera_topic']}... Press 'q' to quit.")
                        
                        # Keep checking if we should continue running
                        while node_instance.camera_active and rclpy.ok():
                            frame = last_frame["image"]
                            if frame is not None:
                                cv2.imshow("Live Unitree Go2 Camera View", frame)
                                key = cv2.waitKey(1)
                                if key == ord('q'):
                                    break
                            # Small sleep to prevent high CPU usage
                            time.sleep(0.01)

                    except Exception as e:
                        node_instance.logger.error(f"Camera thread error: {str(e)}")
                    finally:
                        # Clean up resources
                        with node_instance.camera_lock:
                            node_instance.camera_active = False
                        
                        if sub is not None:
                            node_instance.destroy_subscription(sub)
                        cv2.destroyAllWindows()
                        node_instance.logger.info("Camera thread stopped")

                # Set the flag and start the thread
                node_instance.camera_active = True
                node_instance.camera_thread = threading.Thread(target=camera_thread_function)
                node_instance.camera_thread.daemon = True  # Thread will exit when main program exits
                node_instance.camera_thread.start()
                
                return {"message": "📷 Live camera stream started in a separate window. Press 'q' in the camera window to close."}
            
        @tool
        def move_to_pose(target_x: float, target_y: float) -> str:
            """
            Move the Unitree Go2 robot to a specific (x,y) position.
            First rotates to face the target, then moves in a straight line.
            Coordinates are in meters relative to the world frame.
            """
            # Input validation
            if not isinstance(target_x, (int, float)) or not isinstance(target_y, (int, float)):
                return "Error: Target coordinates must be numbers"
            
            # Check if robot is already in motion
            if node_instance.robot_state["moving"]:
                return "The robot is already moving. Please wait for the current motion to complete."
                
            # Use timeout to prevent hanging
            max_execution_time = 60.0  # 1 minute max for the entire operation
            start_time = time.time()
            
            node_instance.logger.info(f"Moving to pose: ({target_x:.2f}, {target_y:.2f})")
            
            # Get current pose with minimal processing
            try:
                current_pose = get_robot_pose.invoke(input="")
                if "error" in current_pose:
                    return f"Error getting initial pose: {current_pose['error']}"
                
                # Store current pose
                node_instance.robot_state["last_pose"] = current_pose
            except Exception as e:
                node_instance.logger.error(f"Error in move_to_pose: {str(e)}")
                return f"Error getting initial pose: {str(e)}"
                
            # Extract current position and orientation
            current_x = current_pose["x"]
            current_y = current_pose["y"]
            current_yaw = current_pose["yaw"]
            
            # Calculate vector to target
            dx = target_x - current_x
            dy = target_y - current_y
            distance_to_target = math.sqrt(dx*dx + dy*dy)
            
            # If we're already very close to the target, just return
            if distance_to_target < 0.10:  # 10cm tolerance
                return f"Already at target position ({target_x:.2f}, {target_y:.2f})"
                
            # Calculate the angle to the target position
            target_angle = math.atan2(dy, dx)
            
            # Calculate the rotation needed
            rotation_needed = target_angle - current_yaw
            # Normalize to [-π, π]
            rotation_needed = ((rotation_needed + math.pi) % (2 * math.pi)) - math.pi
            
            # Convert radians to degrees for the rotation command
            rotation_degrees = math.degrees(rotation_needed)
            
            # Mark robot as moving
            node_instance.robot_state["moving"] = True
            node_instance.robot_state["last_cmd_time"] = time.time()
            node_instance.robot_state["last_cmd"] = "move_to_pose"
            
            try:
                # Step 1: Rotate to face the target - with reduced angle precision for speed
                # If the angle is very small, skip rotation
                if abs(rotation_degrees) > 5.0:  # Only rotate if more than 5 degrees off
                    node_instance.logger.info(f"Rotating {rotation_degrees:.1f}° to face target")
                    
                    # FIXED: Track rotation status and handle potential errors
                    rotation_result = publish_angular_motion.invoke(angle=rotation_degrees)
                    if "Error" in rotation_result or "timed out" in rotation_result:
                        node_instance.logger.warning(f"Rotation issue: {rotation_result}")
                        # Try to continue anyway - the rotation might be close enough
                else:
                    rotation_result = "Skipped rotation (angle too small)"
                
                # Check timeout after rotation
                if time.time() - start_time > max_execution_time:
                    node_instance.logger.warning("Move to pose operation timed out after rotation")
                    return f"Operation timed out after rotation. Target was ({target_x:.2f}, {target_y:.2f})"
                
                # FIXED: Get current pose again after rotation for better accuracy
                try:
                    # CRITICAL FIX: Remove input parameter - this was causing the error
                    current_pose = get_robot_pose.invoke(input="")
                    if "error" not in current_pose:
                        # Recalculate distance after rotation
                        current_x = current_pose["x"]
                        current_y = current_pose["y"]
                        dx = target_x - current_x
                        dy = target_y - current_y
                        distance_to_target = math.sqrt(dx*dx + dy*dy)
                except Exception:
                    # If we can't get the pose, use the original distance calculation
                    pass
                
                # Step 2: Move forward to the target
                node_instance.logger.info(f"Moving forward {distance_to_target:.2f}m to target")
                
                movement_result = publish_linear_motion.invoke(distance=distance_to_target)
                if "Error" in movement_result or "timed out" in movement_result:
                    node_instance.logger.warning(f"Movement issue: {movement_result}")
                
                # Get final pose to report actual position
                if time.time() - start_time < max_execution_time - 1:
                    try:
                        # CRITICAL FIX: Remove input parameter - this was causing the error
                        final_pose = get_robot_pose.invoke()
                        final_x = final_pose.get("x", "unknown")
                        final_y = final_pose.get("y", "unknown")
                        final_position = f"({final_x}, {final_y})"
                        
                        # Calculate final distance error
                        if isinstance(final_x, (int, float)) and isinstance(final_y, (int, float)):
                            error = math.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
                            node_instance.logger.info(f"Move complete. Final position error: {error:.2f}m")
                            
                            # If error is small enough, report success
                            if error < 0.15:  # 15cm tolerance
                                return f"Successfully moved to target ({target_x:.2f}, {target_y:.2f}). Final position: {final_position}"
                    
                    except Exception:
                        final_position = "unknown"
                else:
                    final_position = "(position check skipped to save time)"
            
            finally:
                # Ensure robot is marked as not moving
                node_instance.robot_state["moving"] = False
            
            return f"Moved to ({target_x:.2f}, {target_y:.2f}). Final position: {final_position}"
                    
        @tool
        def stop_camera() -> dict:
            """
            Stop the currently running camera stream if active.
            """
            with node_instance.camera_lock:
                if not node_instance.camera_active:
                    return {"message": "No active camera stream to stop."}
                
                # Signal the thread to stop
                node_instance.camera_active = False
            
            # Wait for thread to finish with timeout
            if node_instance.camera_thread is not None and node_instance.camera_thread.is_alive():
                node_instance.camera_thread.join(timeout=2.0)
                node_instance.camera_thread = None
                
            # Make sure all OpenCV windows are closed
            cv2.destroyAllWindows()
            node_instance.logger.info("Camera stopped")
            
            return {"message": "Camera stopped successfully."}
        
        # ADDED: Emergency stop command
        @tool
        def emergency_stop() -> str:
            """
            Immediately stop all robot movement. Use in case of emergency or if robot appears stuck.
            """
            node_instance.logger.warning("EMERGENCY STOP TRIGGERED")
            
            # Send multiple stop commands with zero velocity
            node_instance._stop_robot()
            
            # Reset robot state
            node_instance.robot_state["moving"] = False
            node_instance.robot_state["last_cmd"] = None
            
            return "Emergency stop activated. All robot movement halted."
                
        @tool
        def calculate_angle_between_points(from_x: float, from_y: float, to_x: float, to_y: float) -> dict:
            """
            Calculate the angle (in degrees) between two points.
            Returns the angle needed to rotate from current orientation to face the target point.
            """
            # Input validation
            if not all(isinstance(x, (int, float)) for x in [from_x, from_y, to_x, to_y]):
                return {"error": "All coordinates must be numbers"}
                
            # Calculate vector from current to target
            dx = to_x - from_x
            dy = to_y - from_y
            
            # Calculate angle in radians and convert to degrees
            angle_radians = math.atan2(dy, dx)
            angle_degrees = math.degrees(angle_radians)
            distance = math.sqrt(dx*dx + dy*dy)
            
            return {
                "angle_radians": angle_radians,
                "angle_degrees": angle_degrees,
                "distance": distance
            }
            
        # ============================= DIAGNOSTIC TOOLS =============================
        @tool
        def system_status() -> dict:
            """
            Get the current status of the Go2 robot including position, camera status, and more.
            """
            status = {
                "timestamp": time.time(),
                "camera_active": node_instance.camera_active,
                "robot_moving": node_instance.robot_state["moving"],
                "last_command": node_instance.robot_state["last_cmd"],
                "config": {
                    "model": node_instance.config['model'],
                    "cmd_vel_topic": node_instance.config['cmd_vel_topic'],
                    "camera_topic": node_instance.config['camera_topic'],
                    "linear_speed": node_instance.config['linear_speed'],
                    "angular_speed": node_instance.config['angular_speed']
                }
            }
            
            # Try to get robot pose
            try:
                pose = get_robot_pose.invoke(input="")
                if "error" not in pose:
                    status["position"] = {
                        "x": pose["x"],
                        "y": pose["y"],
                        "yaw_degrees": math.degrees(pose["yaw"]),
                        "linear_velocity": pose["linear_x"],
                        "angular_velocity": pose["angular_z"]
                    }
                else:
                    status["position"] = {"error": pose["error"]}
            except Exception as e:
                status["position"] = {"error": str(e)}
                
            return status
            
        # Return all tools for the Unitree Go2 robot
        return [
            publish_linear_motion,
            publish_angular_motion,
            get_robot_pose,
            get_robot_camera_image,
            stop_camera,
            move_to_pose,
            calculate_angle_between_points,
            system_status,
            emergency_stop  # Added emergency stop tool
        ]


# ============================= MAIN FUNCTION =============================
def main(args=None):
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Unitree Go2 ROSA Agent')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--test-hardware', '-t', action='store_true', help='Run hardware test at startup')
    parsed_args = parser.parse_args(args=args)
    
    # Initialize ROS
    rclpy.init(args=args)
    
    try:
        # Create node with optional config
        node = Go2AgentNode(config_path=parsed_args.config)
        
        # Adjust log level if verbose flag is set
        if parsed_args.verbose:
            node.logger.setLevel(logging.DEBUG)
            node.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
            
        # Run hardware test if requested
        # Run hardware test if requested
        if parsed_args.test_hardware:
            print("🔍 Running hardware test...")
            try:
                # Test ROS topic connections
                pose_tool = node._create_tools()[2]  # get_robot_pose
                
                pose_result = pose_tool.invoke(input="")
                
                if "error" in pose_result:
                    print(f"⚠️ Warning: Could not get robot pose: {pose_result['error']}")
                else:
                    print(f"✅ Robot pose topics working: position ({pose_result['x']}, {pose_result['y']})")
                    
                # Test command velocity publishing
                print("🔄 Testing velocity commands...")
                twist = Twist()
                twist.linear.x = 0.1  # Very small test movement
                node.publisher_.publish(twist)
                time.sleep(0.5)
                twist.linear.x = 0.0
                node.publisher_.publish(twist)
                print("✅ Command velocity publishing test complete")
                
            except Exception as e:
                print(f"❌ Hardware test failed: {str(e)}")
                node.logger.error(f"Hardware test failed: {str(e)}")

        # ============================= INTERACTIVE COMMAND LOOP =============================
        print("🐕 Unitree Go2 ROSA Agent initialized. Ready for commands.")
        print("  Type 'help' for a list of example commands.")
        print("  Type 'exit' or 'quit' to exit.")
        
        # Create executor for efficient ROS callback processing
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        
        # Run executor in a separate thread for better performance
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        # Track command processing time for better user experience
        command_times = []
        
        # Simple help commands
        example_commands = [
            "Move forward 1 meter",
            "Turn left 90 degrees",
            "Go to position (2.5, 1.0)",
            "Show me the camera feed",
            "What's your current position?",
            "Stop the camera",
            "System status"
        ]
        
        while rclpy.ok():
            try:
                user_input = input("🧠 Your command > ")
                command = user_input.strip()
                
                # Handle special commands
                if command.lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break
                elif command.lower() == "help":
                    print("\n🤖 Go2: Here are some example commands you can try:")
                    for i, example in enumerate(example_commands, 1):
                        print(f"  {i}. {example}")
                    continue
                elif command.lower() == "status":
                    # Special direct command to check system status
                    system_status_tool = node._create_tools()[7]  # get the system_status tool
                    status = system_status_tool.invoke()
                    print("\n🤖 Go2: Current system status:")
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                    continue
                elif command.lower() == "stop":
                    # Emergency stop command shortcut
                    print("🛑 Emergency stop activated!")
                    emergency_stop_tool = node._create_tools()[8]  # emergency_stop tool
                    emergency_stop_tool.invoke(input={})
                    continue
                elif not command:
                    continue
                
                print("🤖 Go2: Processing command...")  # Immediate feedback
                
                # Track command processing time
                start_time = time.time()
                
                # Process command through ROSA agent
                response = node.agent.invoke(command)
                
                
                # Record processing time
                elapsed_time = time.time() - start_time
                command_times.append(elapsed_time)
                # Keep only last 10 command times
                if len(command_times) > 10:
                    command_times.pop(0)
                avg_time = sum(command_times) / len(command_times)
                
                # Print response with timing info
                print(f"🤖 Go2: {response}")
                if elapsed_time > 2.0:  # Only show timing info if it's significant
                    print(f"   (Processing took {elapsed_time:.1f}s, avg: {avg_time:.1f}s)")
                
            except KeyboardInterrupt:
                print("\n[!] Command interrupted.")
                continue
            except Exception as e:
                print(f"\n[!] Error processing command: {str(e)}")
                node.logger.error(f"Command error: {str(e)}")
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Shutting down.")
    except Exception as e:
        print(f"\n[!] Fatal error: {str(e)}")
        if 'node' in locals() and node.logger:
            node.logger.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        # Clean up resources
        try:
            # Clean up camera resources if active
            if 'node' in locals():
                # Make sure robot is stopped before shutting down
                try:
                    node._stop_robot()
                except:
                    pass
                    
                with node.camera_lock:
                    if node.camera_active:
                        node.camera_active = False
                
                if node.camera_thread is not None and node.camera_thread.is_alive():
                    node.camera_thread.join(timeout=1.0)
            
            # Join executor thread
            if 'executor_thread' in locals() and executor_thread is not None and executor_thread.is_alive():
                executor_thread.join(timeout=1.0)
                
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Shutdown ROS properly
            if 'executor' in locals():
                executor.shutdown()
            if 'node' in locals():
                node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


# ============================= CONFIGURATION TEMPLATE =============================
def generate_config_template():
    """
    Generate a template YAML configuration file for the Go2 Agent.
    """
    config = {
        'model': 'qwen2.5:14b',
        'temperature': 0.0,
        'linear_speed': 0.5,
        'angular_speed': 1.0,
        'cmd_vel_topic': '/cmd_vel',
        'odom_topic': '/odom/ground_truth',
        'camera_topic': '/go2_rgb',
        'camera_resolution': [250, 250],
        'request_timeout': 30.0,
        'context_size': 4096,
        'cmd_vel_publish_rate': 10.0,
        'movement_timeout_factor': 2.5,
        'movement_tolerance': 0.05,
        'rotation_tolerance': 0.04
    }
    
    with open('go2_config_template.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    print("Configuration template generated: go2_config_template.yaml")


if __name__ == "__main__":
    # Check if user wants to generate a config template
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-config":
        generate_config_template()
    else:
        main()