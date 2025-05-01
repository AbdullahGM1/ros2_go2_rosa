#!/usr/bin/env python3
"""
ROS2 Agent Node for Unitree Go2 Robot
-------------------------------------
This node implements a ROSA (ROS Operating System Agent) for controlling
the Unitree Go2 quadruped robot using natural language commands.
It uses a local LLM to interpret commands and execute them through ROS topics.

Author: AbdullahGM1
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
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
import asyncio
import signal
import sys
from functools import partial


class Go2AgentNode(Node):
    def __init__(self):
        super().__init__('go2_agent_node')
        
        # ============================= PARAMETERS =============================
        # Declare and get parameters with default values
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
        
        # ============================= INITIALIZE NODE =============================
        # Create QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Initialize publishers with QoS
        self.publisher_ = self.create_publisher(Twist, self.cmd_vel_topic, qos_profile)
        
        # Initialize CV Bridge for image processing
        self.bridge = CvBridge()
        
        # Flag to control active camera threads
        self.camera_active = False
        self.camera_thread = None
        self.camera_lock = threading.Lock()  # Add a lock for thread safety
        
        # For graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # ============================= SETUP AGENT =============================
        # Setup the agent
        self.setup_agent()
        
        self.get_logger().info("ROSA Go2 Agent is ready. Type a command:")

    def signal_handler(self, sig, frame):
        """Handle SIGINT (Ctrl+C) gracefully"""
        self.get_logger().info("Shutdown signal received, cleaning up...")
        self.running = False
        
        # Stop camera if active
        with self.camera_lock:
            if self.camera_active:
                self.camera_active = False
                
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
            
        # Make sure all OpenCV windows are closed
        cv2.destroyAllWindows()
        
        # Stop robot movement
        twist = Twist()
        self.publisher_.publish(twist)  # Send zero command
        
        # Perform node shutdown
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    def setup_agent(self):
        """Setup the ROSA agent with LLM and tools"""
        # ============================= INITIALIZE LLM =============================
        # Initialize the local LLM (Ollama)
        local_llm = self._initialize_llm()
        
        # ============================= DEFINE TOOLS & PROMPTS =============================
        # Define tools
        tools = self._create_tools()
        
        # Setup agent prompts
        prompts = self._create_prompts()
        
        # ============================= INITIALIZE ROSA AGENT =============================
        # Initialize ROSA
        self.agent = ROSA(
            ros_version=2,
            llm=local_llm,
            tools=tools,
            prompts=prompts
        )

    def _initialize_llm(self):
        """Initialize and configure the local LLM"""
        return ChatOllama(
            model=self.llm_model,
            temperature=0.0,
            max_retries=2,
            num_ctx=8192,
        )
    
    def _create_prompts(self):
        """Create the system prompts for the agent"""
        return RobotSystemPrompts(
            embodiment_and_persona=(
                "You are a smart Unitree Go2 quadruped robot with a camera. "
                "You can navigate indoor and outdoor environments on four legs."
            ),
            about_your_capabilities=(
                "You have access to tools, and you should always prefer using them over replying directly. "
                "You can move forward/backward and rotate left/right using degrees. "
                "You can also move to specific (x,y) coordinates with the move_to_pose tool. "
                "You have a camera and can see the environment. "
                "You can analyze your current position and orientation in space. "
                "Always use your available tools to answer questions and execute tasks. "
                "Never guess or use ROS commands directly."
            ),
            mission_and_objectives=(
                "Help users control the Go2 robot and inspect the environment using available tools. "
                "When the user gives any command that involves 'camera', 'image', 'see', 'show me', or 'feed', "
                "you **must** call the `get_robot_camera_image` tool. "
                "Execute movement commands precisely and provide feedback on progress. "
                "If a command is ambiguous, ask for clarification rather than guessing."
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
            Positive values move forward, negative values move backward.
            """
            linear_speed = node_instance.linear_speed
            twist = Twist()
            
            # Get initial pose
            try:
                initial_pose = get_robot_pose.invoke({})
                if "error" in initial_pose:
                    return f"Error getting pose: {initial_pose['error']}"
            except Exception as e:
                return f"Error getting pose: {str(e)}"
                
            target_distance = abs(distance)  # We'll handle direction separately
            distance_moved = 0.0
            initial_x = initial_pose["x"]
            initial_y = initial_pose["y"]
            
            # Set direction
            direction = 1.0 if distance >= 0 else -1.0
            twist.linear.x = direction * linear_speed
            
            # Create a rate object for controlling loop frequency
            rate = node_instance.create_rate(node_instance.control_rate)
            
            # Debug output
            node_instance.get_logger().info(f"Starting motion: target={target_distance}, direction={direction}")
            node_instance.get_logger().info(f"Initial position: x={initial_x}, y={initial_y}")
            
            start_time = time.time()
            movement_timeout = max(10.0, target_distance * 2.0)  # Timeout based on distance
            
            while rclpy.ok() and node_instance.running:
                # Check for timeout
                if time.time() - start_time > movement_timeout:
                    twist.linear.x = 0.0
                    node_instance.publisher_.publish(twist)
                    return f"Movement timed out after {movement_timeout:.1f} seconds. " \
                           f"Moved approximately {distance_moved:.2f} meters."
                
                # Get current pose
                try:
                    current_pose = get_robot_pose.invoke({})
                    if "error" in current_pose:
                        twist.linear.x = 0.0
                        node_instance.publisher_.publish(twist)
                        return f"Error during movement: {current_pose['error']}"
                except Exception as e:
                    twist.linear.x = 0.0
                    node_instance.publisher_.publish(twist)
                    return f"Error during movement: {str(e)}"
                
                # Calculate distance moved using odometry
                dx = current_pose["x"] - initial_x
                dy = current_pose["y"] - initial_y
                distance_moved = math.sqrt(dx*dx + dy*dy)
                
                # Debug output every 0.5 seconds (approximately)
                if int(distance_moved * 10) % 5 == 0:
                    node_instance.get_logger().debug(
                        f"Current position: x={current_pose['x']:.2f}, y={current_pose['y']:.2f}, "
                        f"moved={distance_moved:.2f}, target={target_distance:.2f}"
                    )
                
                # Check if we've reached or exceeded the target distance
                if distance_moved >= target_distance:
                    break
                
                # Adjust speed as we approach target
                remaining = target_distance - distance_moved
                if remaining < 1.0:  # Within 1 unit of target
                    twist.linear.x = direction * max(0.5, remaining)  # Slow down, min 0.5
                
                # Publish command
                node_instance.publisher_.publish(twist)
                rclpy.spin_once(node_instance)
                rate.sleep()
            
            # Stop the robot
            twist.linear.x = 0.0
            node_instance.publisher_.publish(twist)
            # Send stop command multiple times to ensure it stops
            for _ in range(5):
                node_instance.publisher_.publish(twist)
                time.sleep(0.01)
            
            node_instance.get_logger().info(f"Motion complete: moved={distance_moved:.2f}, target={target_distance:.2f}")
            
            return f"Moved {'forward' if distance >= 0 else 'backward'} {distance_moved:.2f} meters (target: {target_distance:.2f})."

        @tool
        def publish_angular_motion(angle: float) -> str:
            """
            Rotate the Unitree Go2 robot by specified degrees using closed-loop control.
            Positive values rotate clockwise, negative values rotate counterclockwise.
            """
            angular_speed = node_instance.angular_speed  # radians/second
            angle_radians = math.radians(angle)
            twist = Twist()
            
            # Get initial pose
            try:
                initial_pose = get_robot_pose.invoke({})
                if "error" in initial_pose:
                    return f"Error getting pose: {initial_pose['error']}"
            except Exception as e:
                return f"Error getting pose: {str(e)}"
                
            # Calculate target orientation
            target_theta = initial_pose["yaw"] + angle_radians
            # Normalize to [-π, π]
            target_theta = ((target_theta + math.pi) % (2 * math.pi)) - math.pi
            
            # Direction of rotation
            direction = 1.0 if angle_radians >= 0 else -1.0
            twist.angular.z = direction * angular_speed
            
            # Create a rate object for controlling loop frequency
            rate = node_instance.create_rate(node_instance.control_rate)
            
            start_time = time.time()
            rotation_timeout = max(10.0, abs(angle) / 45.0 * 5.0)  # 5 seconds per 45 degrees
            
            while rclpy.ok() and node_instance.running:
                # Check for timeout
                if time.time() - start_time > rotation_timeout:
                    twist.angular.z = 0.0
                    node_instance.publisher_.publish(twist)
                    return f"Rotation timed out after {rotation_timeout:.1f} seconds. " \
                           f"Please check if the robot is physically able to rotate."
                
                # Get current pose
                try:
                    current_pose = get_robot_pose.invoke({})
                    if "error" in current_pose:
                        twist.angular.z = 0.0
                        node_instance.publisher_.publish(twist)
                        return f"Error during rotation: {current_pose['error']}"
                except Exception as e:
                    twist.angular.z = 0.0
                    node_instance.publisher_.publish(twist)
                    return f"Error during rotation: {str(e)}"
                
                # Calculate remaining angle difference
                current_theta = current_pose["yaw"]
                # Calculate angle difference accounting for wraparound
                angle_diff = target_theta - current_theta
                # Normalize to [-π, π]
                angle_diff = ((angle_diff + math.pi) % (2 * math.pi)) - math.pi
                
                # Check if we've reached the target angle (with small tolerance)
                if abs(angle_diff) < node_instance.angle_tolerance:
                    break
                
                # Adjust speed as we approach target
                if abs(angle_diff) < 0.5:  # ~30 degrees from target
                    # Slow down proportionally to remaining angle
                    twist.angular.z = direction * max(0.2, abs(angle_diff))
                else:
                    # Keep constant speed
                    twist.angular.z = direction * angular_speed
                
                # If the shortest path changed, adjust direction
                if (angle_diff * direction) < 0:
                    direction = -direction
                    twist.angular.z = direction * abs(twist.angular.z)
                
                # Publish command
                node_instance.publisher_.publish(twist)
                rclpy.spin_once(node_instance)
                rate.sleep()
            
            # Stop rotation
            twist.angular.z = 0.0
            node_instance.publisher_.publish(twist)
            
            # Get final orientation for reporting
            try:
                final_pose = get_robot_pose.invoke({})
                final_yaw_degrees = math.degrees(final_pose["yaw"])
                actual_rotation = math.degrees(normalize_angle(final_pose["yaw"] - initial_pose["yaw"]))
            except Exception:
                final_yaw_degrees = "unknown"
                actual_rotation = "unknown"
            
            return f"Rotated {actual_rotation}° {'clockwise' if angle >= 0 else 'counterclockwise'}. " \
                   f"Current orientation: {final_yaw_degrees}°"

        # ============================= SENSOR TOOLS =============================
        @tool
        def get_robot_pose() -> dict:
            """
            Get the pose of the Unitree Go2 robot from odometry data.
            Returns position, orientation, and velocity information.
            """
            pose_data = {}
            odom_received = threading.Event()

            def callback(msg):
                # Extract position
                pose_data["x"] = round(msg.pose.pose.position.x, 2)
                pose_data["y"] = round(msg.pose.pose.position.y, 2)
                pose_data["z"] = round(msg.pose.pose.position.z, 2)
                
                # Extract orientation as quaternion
                qx = msg.pose.pose.orientation.x
                qy = msg.pose.pose.orientation.y
                qz = msg.pose.pose.orientation.z
                qw = msg.pose.pose.orientation.w
                
                # Convert quaternion to euler angles (roll, pitch, yaw)
                # Roll (rotation around x-axis)
                sinr_cosp = 2 * (qw * qx + qy * qz)
                cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                roll = math.atan2(sinr_cosp, cosr_cosp)
                
                # Pitch (rotation around y-axis)
                sinp = 2 * (qw * qy - qz * qx)
                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
                else:
                    pitch = math.asin(sinp)
                
                # Yaw (rotation around z-axis)
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                pose_data["roll"] = round(roll, 3)
                pose_data["pitch"] = round(pitch, 3)
                pose_data["yaw"] = round(yaw, 3)
                pose_data["roll_degrees"] = round(math.degrees(roll), 1)
                pose_data["pitch_degrees"] = round(math.degrees(pitch), 1)
                pose_data["yaw_degrees"] = round(math.degrees(yaw), 1)
                
                # Extract twist (velocity)
                pose_data["linear_x"] = round(msg.twist.twist.linear.x, 2)
                pose_data["linear_y"] = round(msg.twist.twist.linear.y, 2)
                pose_data["linear_z"] = round(msg.twist.twist.linear.z, 2)
                pose_data["angular_x"] = round(msg.twist.twist.angular.x, 2)
                pose_data["angular_y"] = round(msg.twist.twist.angular.y, 2)
                pose_data["angular_z"] = round(msg.twist.twist.angular.z, 2)
                
                # Calculate speed
                linear_speed = math.sqrt(
                    msg.twist.twist.linear.x**2 + 
                    msg.twist.twist.linear.y**2 + 
                    msg.twist.twist.linear.z**2
                )
                pose_data["speed"] = round(linear_speed, 2)
                
                odom_received.set()

            # Create QoS profile for better reliability
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            
            sub = node_instance.create_subscription(
                Odometry,
                node_instance.odom_topic,
                callback,
                qos_profile=qos
            )
            
            # Wait for the message with a timeout
            timeout = 5.0
            if not odom_received.wait(timeout):
                node_instance.destroy_subscription(sub)
                return {"error": f"Odometry data not received in {timeout} seconds. Is the topic '{node_instance.odom_topic}' publishing?"}

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
                    
                    # Flag to detect if we've received any frames
                    image_received = False
                    start_time = time.time()

                    try:
                        def image_callback(msg):
                            nonlocal image_received
                            try:
                                cv_image = node_instance.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                                # Add timestamp to the image
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                cv2.putText(cv_image, timestamp, (10, 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                # Draw a crosshair in the center
                                h, w = cv_image.shape[:2]
                                cv2.line(cv_image, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 2)
                                cv2.line(cv_image, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 2)
                                
                                # Draw "Press 'q' to exit" text
                                cv2.putText(cv_image, "Press 'q' to exit", (10, h-20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                # Flag that we've received at least one image
                                image_received = True
                                
                                # Store the image for display
                                last_frame["image"] = cv_image
                            except Exception as e:
                                node_instance.get_logger().error(f"Image conversion failed: {str(e)}")

                        # Create QoS profile for camera
                        qos = QoSProfile(
                            reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE,
                            history=HistoryPolicy.KEEP_LAST,
                            depth=1
                        )
                        
                        sub = node_instance.create_subscription(
                            Image,
                            node_instance.camera_topic,
                            image_callback,
                            qos_profile=qos
                        )

                        node_instance.get_logger().info("📷 Live streaming camera... Press 'q' to quit.")
                        
                        # Keep checking if we should continue running
                        while node_instance.camera_active and node_instance.running:
                            # Check for timeout waiting for first frame
                            if not image_received and time.time() - start_time > 5.0:
                                node_instance.get_logger().warn(
                                    f"No camera images received after 5 seconds. "
                                    f"Check if topic '{node_instance.camera_topic}' is publishing."
                                )
                                break
                                
                            frame = last_frame["image"]
                            if frame is not None:
                                try:
                                    cv2.imshow("Unitree Go2 Camera View", frame)
                                    key = cv2.waitKey(1)
                                    if key == ord('q'):
                                        break
                                except Exception as e:
                                    node_instance.get_logger().error(f"Error displaying camera frame: {str(e)}")
                                    break
                            
                            # Small sleep to prevent high CPU usage
                            time.sleep(0.01)

                    except Exception as e:
                        node_instance.get_logger().error(f"Camera thread error: {str(e)}")
                    finally:
                        # Clean up resources
                        with node_instance.camera_lock:
                            node_instance.camera_active = False
                        
                        if sub is not None:
                            node_instance.destroy_subscription(sub)
                        cv2.destroyAllWindows()
                        node_instance.get_logger().info("Camera thread stopped")
                        
                        # Return a status message about whether we received any frames
                        if not image_received:
                            return {"message": f"No camera images received. Check if topic '{node_instance.camera_topic}' is publishing."}

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
            # Get current pose
            try:
                current_pose = get_robot_pose.invoke({})
                if "error" in current_pose:
                    return f"Error getting initial pose: {current_pose['error']}"
            except Exception as e:
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
            if distance_to_target < node_instance.position_tolerance:
                return f"Already at target position ({target_x:.2f}, {target_y:.2f})"
                
            # Calculate the angle to the target position
            target_angle = math.atan2(dy, dx)
            
            # Calculate the rotation needed
            rotation_needed = target_angle - current_yaw
            # Normalize to [-π, π]
            rotation_needed = normalize_angle(rotation_needed)
            
            # Convert radians to degrees for the rotation command
            rotation_degrees = math.degrees(rotation_needed)
            
            # Log the calculations
            node_instance.get_logger().info(
                f"Moving to pose: current=({current_x:.2f}, {current_y:.2f}, {math.degrees(current_yaw):.1f}°), "
                f"target=({target_x:.2f}, {target_y:.2f})"
            )
            node_instance.get_logger().info(
                f"Calculations: distance={distance_to_target:.2f}, target_angle={math.degrees(target_angle):.1f}°, "
                f"rotation_needed={rotation_degrees:.1f}°"
            )
            
            # Step 1: Rotate to face the target
            rotation_result = publish_angular_motion.invoke({"angle": rotation_degrees})
            node_instance.get_logger().info(f"Rotation result: {rotation_result}")
            
            # Step 2: Move forward to the target
            movement_result = publish_linear_motion.invoke({"distance": distance_to_target})
            node_instance.get_logger().info(f"Movement result: {movement_result}")
            
            # Get final pose to report actual position
            try:
                final_pose = get_robot_pose.invoke({})
                final_x = final_pose.get("x", "unknown")
                final_y = final_pose.get("y", "unknown")
                final_position = f"({final_x}, {final_y})"
                
                # Calculate remaining distance to target
                dx = target_x - final_x
                dy = target_y - final_y
                remaining_distance = math.sqrt(dx*dx + dy*dy)
                
                accuracy_message = ""
                if isinstance(final_x, float) and isinstance(final_y, float):
                    if remaining_distance < node_instance.position_tolerance:
                        accuracy_message = " Target position reached successfully!"
                    else:
                        accuracy_message = f" Remaining distance to target: {remaining_distance:.2f} meters."
            except Exception:
                final_position = "unknown"
                accuracy_message = ""
            
            return (f"Moved to position ({target_x:.2f}, {target_y:.2f}). "
                    f"First rotated {rotation_degrees:.1f}° then moved forward {distance_to_target:.2f} meters. "
                    f"Final position: {final_position}.{accuracy_message}")
                    
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
            
            return {"message": "Camera stream stopped successfully."}
                
        @tool
        def calculate_angle_between_points(from_x: float, from_y: float, to_x: float, to_y: float) -> dict:
            """
            Calculate the angle (in degrees) between two points.
            Returns the angle needed to rotate from current orientation to face the target point.
            """
            # Calculate vector from current to target
            dx = to_x - from_x
            dy = to_y - from_y
            
            # Calculate angle in radians and convert to degrees
            angle_radians = math.atan2(dy, dx)
            angle_degrees = math.degrees(angle_radians)
            
            return {
                "angle_radians": angle_radians,
                "angle_degrees": angle_degrees,
                "distance": math.sqrt(dx*dx + dy*dy)
            }
        
        @tool
        def patrol_area(width: float, height: float, loops: int = 1) -> str:
            """
            Make the robot patrol a rectangular area of specified width and height.
            Starts from current position and performs a rectangular patrol pattern.
            
            Args:
                width: Width of the patrol rectangle in meters
                height: Height of the patrol rectangle in meters
                loops: Number of times to patrol the rectangle (default: 1)
            """
            if width <= 0 or height <= 0:
                return "Error: Width and height must be positive values."
            
            if loops <= 0:
                return "Error: Number of loops must be positive."
            
            # Get starting position
            try:
                start_pose = get_robot_pose.invoke({})
                if "error" in start_pose:
                    return f"Error getting initial pose: {start_pose['error']}"
                
                start_x = start_pose["x"]
                start_y = start_pose["y"]
            except Exception as e:
                return f"Error getting initial pose: {str(e)}"
            
            loop_counter = 0
            result_messages = [f"Starting patrol of {width}x{height}m rectangle for {loops} {'loop' if loops == 1 else 'loops'}..."]
            
            try:
                # Perform the patrol loop
                while loop_counter < loops and node_instance.running:
                    # Calculate the four corners relative to starting position
                    corners = [
                        (start_x, start_y),  # Starting position
                        (start_x + width, start_y),  # Right
                        (start_x + width, start_y + height),  # Top right
                        (start_x, start_y + height),  # Top left
                        (start_x, start_y)   # Back to start
                    ]
                    
                    # Move to each corner
                    for i, (x, y) in enumerate(corners):
                        corner_name = ["start", "right", "top-right", "top-left", "start"][i]
                        result = move_to_pose.invoke({"target_x": x, "target_y": y})
                        result_messages.append(f"Corner {i+1} ({corner_name}): {result}")
                    
                    loop_counter += 1
                    if loop_counter < loops:
                        result_messages.append(f"Completed loop {loop_counter}/{loops}")
                
                # Get final position
                final_pose = get_robot_pose.invoke({})
                final_x = final_pose.get("x", "unknown")
                final_y = final_pose.get("y", "unknown")
                
                result_messages.append(f"Patrol completed. Final position: ({final_x}, {final_y})")
                return "\n".join(result_messages)
                
            except Exception as e:
                return f"Error during patrol: {str(e)}\nPartial results:\n" + "\n".join(result_messages)

        # Helper function for normalizing angles
        def normalize_angle(angle):
            """Normalize angle to [-π, π]"""
            return ((angle + math.pi) % (2 * math.pi)) - math.pi
                    
        # Return all tools for the Unitree Go2 robot
        return [
            publish_linear_motion,
            publish_angular_motion,
            get_robot_pose,
            get_robot_camera_image,
            stop_camera,
            move_to_pose,
            calculate_angle_between_points,
            patrol_area
        ]


def main(args=None):
    """Main function to initialize and run the Go2AgentNode with improved CLI including visible thinking"""
    import readline  # Add command history and editing capabilities
    import os
    from colorama import init, Fore, Style, Back  # For colored output
    import threading
    import time
    import re  # Import regex for pattern matching
    
    # Initialize colorama
    init()
    
    # Initialize ROS
    rclpy.init(args=args)
    node = Go2AgentNode()
    
    # Command history file
    history_file = os.path.expanduser('~/.go2agent_history')
    try:
        readline.read_history_file(history_file)
        # Set history file size
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    
    # Create a separate thread for processing ROS callbacks
    def spin_thread():
        while node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    
    ros_thread = threading.Thread(target=spin_thread)
    ros_thread.daemon = True
    ros_thread.start()

    try:
        # ============================= INTERACTIVE COMMAND LOOP =============================
        print(f"{Fore.GREEN}🐕 Unitree Go2 ROSA Agent initialized. Ready for commands.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Type 'exit' or 'quit' to exit, 'help' for available commands.{Style.RESET_ALL}")
        
        # Simple built-in commands with improved output
        def show_help():
            help_text = f"""
{Fore.YELLOW}=== Available Commands ==={Style.RESET_ALL}
{Fore.GREEN}Basic Commands:{Style.RESET_ALL}
  {Fore.CYAN}help{Style.RESET_ALL}             Show this help message
  {Fore.CYAN}status{Style.RESET_ALL}           Show robot status
  {Fore.CYAN}stop{Style.RESET_ALL}             Emergency stop the robot
  {Fore.CYAN}quit, exit{Style.RESET_ALL}       Exit the program

{Fore.GREEN}Movement Commands (Natural Language):{Style.RESET_ALL}
  {Fore.CYAN}move forward 1.5{Style.RESET_ALL}       Move forward 1.5 meters
  {Fore.CYAN}turn right 90{Style.RESET_ALL}          Turn 90 degrees clockwise
  {Fore.CYAN}go to position 2.5 3.0{Style.RESET_ALL} Move to position (2.5, 3.0)
  {Fore.CYAN}patrol area 4 5{Style.RESET_ALL}        Patrol a 4x5 meter rectangle

{Fore.GREEN}Sensor Commands:{Style.RESET_ALL}
  {Fore.CYAN}show camera{Style.RESET_ALL}            Show camera feed
  {Fore.CYAN}what's my position{Style.RESET_ALL}     Show current position
  {Fore.CYAN}stop camera{Style.RESET_ALL}            Stop camera feed

You can use natural language to control the robot. Examples:
- "Move forward 2 meters then turn left 45 degrees"
- "Go to the coordinates 3.5, 4.2"
- "Show me what you see"
- "Patrol a 5 by 6 meter area twice"
            """
            print(help_text)
        
        def emergency_stop():
            try:
                twist = Twist()
                node.publisher_.publish(twist)
                for _ in range(5):  # Send multiple stop commands to ensure it stops
                    node.publisher_.publish(twist)
                    time.sleep(0.01)
                print(f"{Fore.RED}🛑 EMERGENCY STOP ACTIVATED - Robot movement halted{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error during emergency stop: {str(e)}{Style.RESET_ALL}")
        
        # Map built-in commands
        builtin_commands = {
            'help': show_help,
            'status': lambda: print(f"{Fore.YELLOW}Getting robot status...{Style.RESET_ALL}"),
            'stop': emergency_stop,
            'robot status': lambda: print(f"{Fore.YELLOW}Getting robot status...{Style.RESET_ALL}"),
        }
        
        # Function to extract and format thinking process
        def extract_thinking(response):
            """Extract thinking process from <think> tags and format it nicely"""
            thinking = ""
            response_text = response
            
            # Find all <think> blocks
            think_pattern = r'<think>([\s\S]*?)</think>'
            think_matches = re.findall(think_pattern, response)
            
            if think_matches:
                # Join all thinking blocks if there are multiple
                thinking = "\n".join(think_match.strip() for think_match in think_matches)
                
                # Remove the <think> blocks from the response
                response_text = re.sub(think_pattern, '', response).strip()
            
            return thinking, response_text
        
        # Command history
        command_history = []
        
        while node.running:
            try:
                # Display a personalized prompt with robot status
                prompt = f"{Fore.GREEN}🤖 Go2 [✓] >{Style.RESET_ALL} "
                user_input = input(prompt)
                readline.write_history_file(history_file)
                
                input_lower = user_input.strip().lower()
                command_history.append(input_lower)
                
                # Handle exit commands
                if input_lower in ["exit", "quit"]:
                    print(f"{Fore.YELLOW}Exiting...{Style.RESET_ALL}")
                    break
                    
                # Handle built-in commands
                elif input_lower in builtin_commands:
                    builtin_commands[input_lower]()
                    
                    # For status commands, still process with the agent
                    if input_lower not in ['help', 'stop']:
                        pass  # Continue to agent processing
                    else:
                        continue
                    
                # Handle empty input
                elif not input_lower:
                    continue
                    
                # Process via ROSA agent
                print(f"{Fore.CYAN}Processing: {input_lower}{Style.RESET_ALL}")
                
                # Create a thread for the agent processing
                result = [None]
                error = [None]
                
                def process_command():
                    try:
                        result[0] = node.agent.invoke(input_lower)
                    except Exception as e:
                        error[0] = str(e)
                
                agent_thread = threading.Thread(target=process_command)
                agent_thread.daemon = True
                agent_thread.start()
                
                # Wait for processing to complete (with timeout)
                agent_thread.join(timeout=60)  # 60 seconds max wait
                
                if agent_thread.is_alive():
                    print(f"{Fore.RED}Response taking too long, consider emergency stop if needed{Style.RESET_ALL}")
                elif error[0]:
                    print(f"{Fore.RED}Error: {error[0]}{Style.RESET_ALL}")
                else:
                    # Extract thinking and format the response
                    response = result[0]
                    thinking, clean_response = extract_thinking(response)
                    
                    # Display the thinking process in a clearly marked section
                    if thinking:
                        print(f"\n{Back.BLUE}{Fore.WHITE} THINKING PROCESS {Style.RESET_ALL}")
                        # Format the thinking text with indentation and yellow color
                        formatted_thinking = ""
                        for line in thinking.split('\n'):
                            formatted_thinking += f"{Fore.YELLOW}  │ {line}{Style.RESET_ALL}\n"
                        print(formatted_thinking)
                        print(f"{Back.BLUE}{Fore.WHITE} END THINKING {Style.RESET_ALL}\n")
                    
                    # Display the actual response
                    print(f"{Fore.GREEN}🤖 Go2:{Style.RESET_ALL} {clean_response}")
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}[!] Command interrupted. Type 'stop' for emergency stop.{Style.RESET_ALL}")
                continue
            except Exception as e:
                print(f"{Fore.RED}Error processing command: {str(e)}{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"\n{Fore.RED}[!] Error: {str(e)}{Style.RESET_ALL}")
    finally:
        print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
        # Make sure to clean up the camera thread if it's running
        with node.camera_lock:
            if node.camera_active:
                node.camera_active = False
                
        if node.camera_thread is not None and node.camera_thread.is_alive():
            node.camera_thread.join(timeout=1.0)
            
        # Make sure all OpenCV windows are closed
        cv2.destroyAllWindows()
        
        # Stop the robot
        node.publisher_.publish(Twist())
        
        # Set running flag to False to stop the spin thread
        node.running = False
        
        # Wait for ROS thread to finish
        if ros_thread.is_alive():
            ros_thread.join(timeout=1.0)
            
        # Cleanup ROS
        node.destroy_node()
        rclpy.shutdown()