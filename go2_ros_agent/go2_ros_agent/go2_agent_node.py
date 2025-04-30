#!/usr/bin/env python3
"""
ROS2 Agent Node for Unitree Go2 Robot
-------------------------------------
This node implements a ROSA (ROS Operating System Agent) for controlling
the Unitree Go2 quadruped robot using natural language commands.
It uses a local LLM to interpret commands and execute them through ROS topics.

Author: AbdullaGo2_robot_agent_noce (code)hGM1
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


class Go2AgentNode(Node):
    def __init__(self):
        super().__init__('go2_agent_node')
        
        # ============================= INITIALIZE NODE =============================
        # Initialize publishers
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Initialize CV Bridge for image processing
        self.bridge = CvBridge()
        
        # Flag to control active camera threads
        self.camera_active = False
        self.camera_thread = None
        self.camera_lock = threading.Lock()  # Add a lock for thread safety
        
        # ============================= SETUP AGENT =============================
        # Setup the agent
        self.setup_agent()
        
        self.get_logger().info("ROSA Unitree Go2 Agent is ready. Type a command:")

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
            model="qwen2.5:14b",  # Use a smaller model for faster inference
            temperature=0.0,
            max_retries=1,  # Reduced retry count
            num_ctx=4096,    # Reduced context size for faster processing
            request_timeout=30.0  # Add timeout to prevent hanging
        )
    
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
            # Use faster speed for quicker execution
            linear_speed = 0.5  # 0.5 m/s is a reasonable walking speed for Go2
            twist = Twist()
            
            # Set a timeout to prevent hanging
            max_execution_time = min(30.0, abs(distance) / linear_speed * 2.0)
            
                            # Get initial pose - use a fast version that doesn't capture all data
            try:
                initial_pose = get_robot_pose.invoke()
                if "error" in initial_pose:
                    return f"Error getting pose: {initial_pose['error']}"
            except Exception as e:
                return f"Error getting pose: {str(e)}"
                
            initial_x = initial_pose["x"]
            initial_y = initial_pose["y"]
            target_distance = abs(distance)
            
            # Set direction
            direction = 1.0 if distance >= 0 else -1.0
            twist.linear.x = direction * linear_speed
            
            # Create a rate object with higher frequency
            rate = node_instance.create_rate(10)  # Reduced from 20Hz to 10Hz
            
            # Start time for timeout
            start_time = time.time()
            
            # Use simpler loop condition
            while rclpy.ok():
                # Check timeout
                if time.time() - start_time > max_execution_time:
                    twist.linear.x = 0.0
                    node_instance.publisher_.publish(twist)
                    return f"Motion timed out after {max_execution_time:.1f} seconds. Target was {target_distance:.2f} meters."
                
                # Get current pose with minimal data
                try:
                    current_pose = get_robot_pose.invoke()
                    if "error" in current_pose:
                        twist.linear.x = 0.0
                        node_instance.publisher_.publish(twist)
                        return f"Error during movement: {current_pose['error']}"
                except Exception as e:
                    twist.linear.x = 0.0
                    node_instance.publisher_.publish(twist)
                    return f"Error during movement: {str(e)}"
                
                # Fast distance calculation
                dx = current_pose["x"] - initial_x
                dy = current_pose["y"] - initial_y
                distance_moved = math.sqrt(dx*dx + dy*dy)
                
                # Check if we've reached the target
                if distance_moved >= target_distance:
                    break
                
                # Simplify speed adjustment - only slow down very close to target
                if target_distance - distance_moved < 0.3:  # 30cm from target
                    twist.linear.x = direction * 0.2  # Slow down to 0.2 m/s
                
                # Publish command
                node_instance.publisher_.publish(twist)
                rate.sleep()
            
            # Stop the robot - send stop command multiple times
            twist.linear.x = 0.0
            for _ in range(3):  # Reduced from 5 to 3
                node_instance.publisher_.publish(twist)
                time.sleep(0.01)
            
            return f"Moved {'forward' if distance >= 0 else 'backward'} {target_distance:.2f} meters."

        @tool
        def publish_angular_motion(angle: float) -> str:
            """
            Rotate the Unitree Go2 robot by specified degrees using closed-loop control.
            Positive values rotate clockwise, negative values rotate counterclockwise.
            """
            # Use faster angular speed
            angular_speed = 1.0  # radians/second - increased from 0.8
            angle_radians = math.radians(angle)
            twist = Twist()
            
            # Set a maximum execution time to prevent hanging
            max_execution_time = min(15.0, abs(angle_radians) / angular_speed * 2.0)
            
            # Get initial pose with minimal data
            try:
                initial_pose = get_robot_pose.invoke()
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
            
            # Create a rate object with lower frequency
            rate = node_instance.create_rate(10)  # Reduced from 20Hz to 10Hz
            
            # Start time for timeout
            start_time = time.time()
            
            while rclpy.ok():
                # Check timeout
                if time.time() - start_time > max_execution_time:
                    twist.angular.z = 0.0
                    node_instance.publisher_.publish(twist)
                    return f"Rotation timed out after {max_execution_time:.1f} seconds. Target was {angle:.0f} degrees."
                
                # Get current pose with minimal data
                try:
                    current_pose = get_robot_pose.invoke()
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
                
                # Check if we've reached the target angle (with wider tolerance)
                if abs(angle_diff) < 0.08:  # ~4.5 degrees tolerance (increased from 0.05)
                    break
                
                # Simplified speed adjustment - only slow down very close to target
                if abs(angle_diff) < 0.3:  # ~17 degrees from target
                    twist.angular.z = direction * 0.4  # Fixed slower speed
                
                # Publish command
                node_instance.publisher_.publish(twist)
                rate.sleep()
            
            # Stop rotation
            twist.angular.z = 0.0
            for _ in range(3):  # Send stop command multiple times
                node_instance.publisher_.publish(twist)
                time.sleep(0.01)
            
            return f"Rotated {angle:.0f}° {'clockwise' if angle >= 0 else 'counterclockwise'}."

        # ============================= SENSOR TOOLS =============================
        @tool
        def get_robot_pose() -> dict:
            """
            Get the pose of the Unitree Go2 robot from odometry data (/odom/ground_truth).
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
                
                # Only calculate yaw (we mostly need this for navigation)
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                pose_data["yaw"] = round(yaw, 2)
                
                # Only extract linear.x velocity as it's the most commonly used
                pose_data["linear_x"] = round(msg.twist.twist.linear.x, 2)
                pose_data["angular_z"] = round(msg.twist.twist.angular.z, 2)
                
                msg_received.set()

            sub = node_instance.create_subscription(
                Odometry,
                "/odom/ground_truth",
                callback,
                10
            )
            
            # Wait for the message with a shorter timeout (2 seconds)
            if not msg_received.wait(timeout=2.0):
                node_instance.destroy_subscription(sub)
                return {"error": "Odometry data not received in time. Is the topic available?"}

            # Clean up subscription immediately
            node_instance.destroy_subscription(sub)
            return pose_data

        # ============================= CAMERA TOOLS =============================
        @tool
        def get_robot_camera_image() -> dict:
            """
            Display a live stream from the Unitree Go2's RGB camera (/go2_rgb) in a non-blocking way.
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

                    try:
                        def image_callback(msg):
                            try:
                                cv_image = node_instance.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                                resized_image = cv2.resize(cv_image, (250, 250))
                                last_frame["image"] = resized_image
                            except Exception as e:
                                node_instance.get_logger().error(f"Image conversion failed: {str(e)}")

                        sub = node_instance.create_subscription(
                            Image,
                            "/go2_rgb",
                            image_callback,
                            10
                        )

                        node_instance.get_logger().info("📷 Live streaming camera... Press 'q' to quit.")
                        
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
                        node_instance.get_logger().error(f"Camera thread error: {str(e)}")
                    finally:
                        # Clean up resources
                        with node_instance.camera_lock:
                            node_instance.camera_active = False
                        
                        if sub is not None:
                            node_instance.destroy_subscription(sub)
                        cv2.destroyAllWindows()
                        node_instance.get_logger().info("Camera thread stopped")

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
            # Use timeout to prevent hanging
            max_execution_time = 60.0  # 1 minute max for the entire operation
            start_time = time.time()
            
            # Get current pose with minimal processing
            try:
                current_pose = get_robot_pose.invoke()
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
            
            # Skip detailed logging to improve speed
            
            # Step 1: Rotate to face the target - with reduced angle precision for speed
            # If the angle is very small, skip rotation
            if abs(rotation_degrees) > 5.0:  # Only rotate if more than 5 degrees off
                rotation_result = publish_angular_motion.invoke(rotation_degrees)
            else:
                rotation_result = "Skipped rotation (angle too small)"
            
            # Check timeout after rotation
            if time.time() - start_time > max_execution_time:
                return f"Operation timed out after rotation. Target was ({target_x:.2f}, {target_y:.2f})"
            
            # Step 2: Move forward to the target
            movement_result = publish_linear_motion.invoke(distance_to_target)
            
            # Get final pose to report actual position - but only if we have time
            if time.time() - start_time < max_execution_time - 1:
                try:
                    final_pose = get_robot_pose.invoke()
                    final_x = final_pose.get("x", "unknown")
                    final_y = final_pose.get("y", "unknown")
                    final_position = f"({final_x}, {final_y})"
                except Exception:
                    final_position = "unknown"
            else:
                final_position = "(position check skipped to save time)"
            
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
            
            return {"message": "Camera stopped successfully."}
                
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
            
        # Return all tools for the Unitree Go2 robot
        return [
            publish_linear_motion,
            publish_angular_motion,
            get_robot_pose,
            get_robot_camera_image,
            stop_camera,
            move_to_pose,
            calculate_angle_between_points
        ]


# ============================= MAIN FUNCTION =============================
def main(args=None):
    rclpy.init(args=args)
    node = Go2AgentNode()

    try:
        # ============================= INTERACTIVE COMMAND LOOP =============================
        print("🐕 Unitree Go2 ROSA Agent initialized. Ready for commands.")
        
        # Create executor for efficient ROS callback processing
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        
        # Run executor in a separate thread for better performance
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        while rclpy.ok():
            user_input = input("🧠 Your command > ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            print("🤖 Go2: Processing command...")  # Immediate feedback
            response = node.agent.invoke(user_input)
            print(f"🤖 Go2: {response}")
            
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Shutting down.")
    finally:
        # Make sure to clean up resources
        with node.camera_lock:
            if node.camera_active:
                node.camera_active = False
                
        if node.camera_thread is not None and node.camera_thread.is_alive():
            node.camera_thread.join(timeout=1.0)
        
        # Also join the executor thread
        if 'executor_thread' in locals() and executor_thread is not None and executor_thread.is_alive():
            executor_thread.join(timeout=1.0)
            
        # Make sure all OpenCV windows are closed
        cv2.destroyAllWindows()
        
        # Shutdown ROS properly
        if 'executor' in locals():
            executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()