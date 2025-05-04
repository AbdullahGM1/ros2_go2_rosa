#!/usr/bin/env python3
"""
Robot tools for the Unitree Go2 ROSA Agent.
This module contains all robot-specific tools and functions.
"""

import math
import time
import threading
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from langchain.agents import tool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class RobotTools:
    """Collection of tools for Unitree Go2 robot control."""
    
    def __init__(self, node):
        self.node = node
        
    def create_tools(self):
        """Create and return all robot tools."""
        
        @tool
        def publish_linear_motion(distance: float) -> str:
            """
            Move the Unitree Go2 robot forward/backward by the specified distance (in meters)
            using closed-loop control with position feedback.
            Positive values move forward, negative values move backward.
            """
            linear_speed = self.node.linear_speed
            twist = Twist()
            
            # Get initial pose
            try:
                initial_pose = get_robot_pose.invoke({})
                if "error" in initial_pose:
                    return f"Error getting pose: {initial_pose['error']}"
            except Exception as e:
                return f"Error getting pose: {str(e)}"
                
            target_distance = abs(distance)
            distance_moved = 0.0
            initial_x = initial_pose["x"]
            initial_y = initial_pose["y"]
            
            # Set direction
            direction = 1.0 if distance >= 0 else -1.0
            twist.linear.x = direction * linear_speed
            
            # Create a rate object for controlling loop frequency
            rate = self.node.create_rate(self.node.control_rate)
            
            start_time = time.time()
            movement_timeout = max(10.0, target_distance * 2.0)
            
            while self.node.running:
                # Check for timeout
                if time.time() - start_time > movement_timeout:
                    twist.linear.x = 0.0
                    self.node.publisher_.publish(twist)
                    return f"Movement timed out after {movement_timeout:.1f} seconds. " \
                           f"Moved approximately {distance_moved:.2f} meters."
                
                # Get current pose
                try:
                    current_pose = get_robot_pose.invoke({})
                    if "error" in current_pose:
                        twist.linear.x = 0.0
                        self.node.publisher_.publish(twist)
                        return f"Error during movement: {current_pose['error']}"
                except Exception as e:
                    twist.linear.x = 0.0
                    self.node.publisher_.publish(twist)
                    return f"Error during movement: {str(e)}"
                
                # Calculate distance moved using odometry
                dx = current_pose["x"] - initial_x
                dy = current_pose["y"] - initial_y
                distance_moved = math.sqrt(dx*dx + dy*dy)
                
                # Check if we've reached or exceeded the target distance
                if distance_moved >= target_distance:
                    break
                
                # Adjust speed as we approach target
                remaining = target_distance - distance_moved
                if remaining < 1.0:  # Within 1 unit of target
                    twist.linear.x = direction * max(0.5, remaining)  # Slow down, min 0.5
                
                # Publish command
                self.node.publisher_.publish(twist)
                rate.sleep()
            
            # Stop the robot
            twist.linear.x = 0.0
            self.node.publisher_.publish(twist)
            # Send stop command multiple times to ensure it stops
            for _ in range(5):
                self.node.publisher_.publish(twist)
                time.sleep(0.01)
            
            return f"Moved {'forward' if distance >= 0 else 'backward'} {distance_moved:.2f} meters (target: {target_distance:.2f})."

        @tool
        def publish_angular_motion(angle: float) -> str:
            """
            Rotate the Unitree Go2 robot by specified degrees using closed-loop control.
            Positive values rotate clockwise, negative values rotate counterclockwise.
            """
            angular_speed = self.node.angular_speed  # radians/second
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
            target_theta = self.normalize_angle(target_theta)
            
            # Direction of rotation
            direction = 1.0 if angle_radians >= 0 else -1.0
            twist.angular.z = direction * angular_speed
            
            # Create a rate object for controlling loop frequency
            rate = self.node.create_rate(self.node.control_rate)
            
            start_time = time.time()
            rotation_timeout = max(10.0, abs(angle) / 45.0 * 5.0)  # 5 seconds per 45 degrees
            
            while self.node.running:
                # Check for timeout
                if time.time() - start_time > rotation_timeout:
                    twist.angular.z = 0.0
                    self.node.publisher_.publish(twist)
                    return f"Rotation timed out after {rotation_timeout:.1f} seconds. " \
                           f"Please check if the robot is physically able to rotate."
                
                # Get current pose
                try:
                    current_pose = get_robot_pose.invoke({})
                    if "error" in current_pose:
                        twist.angular.z = 0.0
                        self.node.publisher_.publish(twist)
                        return f"Error during rotation: {current_pose['error']}"
                except Exception as e:
                    twist.angular.z = 0.0
                    self.node.publisher_.publish(twist)
                    return f"Error during rotation: {str(e)}"
                
                # Calculate remaining angle difference
                current_theta = current_pose["yaw"]
                angle_diff = target_theta - current_theta
                angle_diff = self.normalize_angle(angle_diff)
                
                # Check if we've reached the target angle
                if abs(angle_diff) < self.node.angle_tolerance:
                    break
                
                # Adjust speed as we approach target
                if abs(angle_diff) < 0.5:  # ~30 degrees from target
                    twist.angular.z = direction * max(0.2, abs(angle_diff))
                else:
                    twist.angular.z = direction * angular_speed
                
                # Publish command
                self.node.publisher_.publish(twist)
                rate.sleep()
            
            # Stop rotation
            twist.angular.z = 0.0
            self.node.publisher_.publish(twist)
            
            # Get final orientation for reporting
            try:
                final_pose = get_robot_pose.invoke({})
                final_yaw_degrees = math.degrees(final_pose["yaw"])
                actual_rotation = math.degrees(self.normalize_angle(final_pose["yaw"] - initial_pose["yaw"]))
            except Exception:
                final_yaw_degrees = "unknown"
                actual_rotation = "unknown"
            
            return f"Rotated {actual_rotation}° {'clockwise' if angle >= 0 else 'counterclockwise'}. " \
                   f"Current orientation: {final_yaw_degrees}°"

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
                roll, pitch, yaw = self.quaternion_to_euler(qx, qy, qz, qw)
                
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
            
            sub = self.node.create_subscription(
                Odometry,
                self.node.odom_topic,
                callback,
                qos_profile=qos
            )
            
            # Wait for the message with a timeout
            timeout = 5.0
            if not odom_received.wait(timeout):
                self.node.destroy_subscription(sub)
                return {"error": f"Odometry data not received in {timeout} seconds. Is the topic '{self.node.odom_topic}' publishing?"}
            self.node.destroy_subscription(sub)
            return pose_data

        @tool
        def get_robot_camera_image() -> dict:
            """
            Display a live stream from the Unitree Go2's RGB camera in a non-blocking way.
            Press 'q' in the camera window to close the stream.
            """
            with self.node.camera_lock:
                # Check if camera is already running
                if self.node.camera_active:
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
                                cv_image = self.node.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
                                self.node.get_logger().error(f"Image conversion failed: {str(e)}")
                        
                        # Create QoS profile for camera
                        qos = QoSProfile(
                            reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE,
                            history=HistoryPolicy.KEEP_LAST,
                            depth=1
                        )
                        
                        sub = self.node.create_subscription(
                            Image,
                            self.node.camera_topic,
                            image_callback,
                            qos_profile=qos
                        )
                        self.node.get_logger().info("� Live streaming camera... Press 'q' to quit.")
                        
                        # Keep checking if we should continue running
                        while self.node.camera_active and self.node.running:
                            # Check for timeout waiting for first frame
                            if not image_received and time.time() - start_time > 5.0:
                                self.node.get_logger().warn(
                                    f"No camera images received after 5 seconds. "
                                    f"Check if topic '{self.node.camera_topic}' is publishing."
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
                                    self.node.get_logger().error(f"Error displaying camera frame: {str(e)}")
                                    break
                            
                            # Small sleep to prevent high CPU usage
                            time.sleep(0.01)
                    except Exception as e:
                        self.node.get_logger().error(f"Camera thread error: {str(e)}")
                    finally:
                        # Clean up resources
                        with self.node.camera_lock:
                            self.node.camera_active = False
                        
                        if sub is not None:
                            self.node.destroy_subscription(sub)
                        cv2.destroyAllWindows()
                        self.node.get_logger().info("Camera thread stopped")
                        
                        # Return a status message about whether we received any frames
                        if not image_received:
                            return {"message": f"No camera images received. Check if topic '{self.node.camera_topic}' is publishing."}
                
                # Set the flag and start the thread
                self.node.camera_active = True
                self.node.camera_thread = threading.Thread(target=camera_thread_function)
                self.node.camera_thread.daemon = True  # Thread will exit when main program exits
                self.node.camera_thread.start()
                
                return {"message": "� Live camera stream started in a separate window. Press 'q' in the camera window to close."}

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
            if distance_to_target < self.node.position_tolerance:
                return f"Already at target position ({target_x:.2f}, {target_y:.2f})"
                
            # Calculate the angle to the target position
            target_angle = math.atan2(dy, dx)
            
            # Calculate the rotation needed
            rotation_needed = target_angle - current_yaw
            # Normalize to [-π, π]
            rotation_needed = self.normalize_angle(rotation_needed)
            
            # Convert radians to degrees for the rotation command
            rotation_degrees = math.degrees(rotation_needed)
            
            # Step 1: Rotate to face the target
            rotation_result = publish_angular_motion.invoke({"angle": rotation_degrees})
            
            # Step 2: Move forward to the target
            movement_result = publish_linear_motion.invoke({"distance": distance_to_target})
            
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
                    if remaining_distance < self.node.position_tolerance:
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
            with self.node.camera_lock:
                if not self.node.camera_active:
                    return {"message": "No active camera stream to stop."}
                
                # Signal the thread to stop
                self.node.camera_active = False
            
            # Wait for thread to finish with timeout
            if self.node.camera_thread is not None and self.node.camera_thread.is_alive():
                self.node.camera_thread.join(timeout=2.0)
                self.node.camera_thread = None
                
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
                while loop_counter < loops and self.node.running:
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

        # Helper functions
        def normalize_angle(angle):
            """Normalize angle to [-π, π]"""
            return ((angle + math.pi) % (2 * math.pi)) - math.pi
        
        def quaternion_to_euler(x, y, z, w):
            """Convert quaternion to euler angles (roll, pitch, yaw)"""
            # Roll (rotation around x-axis)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (rotation around y-axis)
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
            else:
                pitch = math.asin(sinp)
            
            # Yaw (rotation around z-axis)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return roll, pitch, yaw
        
        # Make helper functions accessible to tools
        self.normalize_angle = normalize_angle
        self.quaternion_to_euler = quaternion_to_euler
        
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