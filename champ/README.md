

#  Unitree Go2 ROSA Agent

![Go2 Velodyne Gazebo RViz Launch](.docs/gazebo_velodyne_rviz_launch.png)

## 🤖 Overview

This package provides a ROS2 agent node for controlling the Unitree Go2 quadruped robot using natural language commands. The system uses:

- 🧠 Local LLM (Qwen2.5) for natural language processing
- 🔄 ROSA (ROS Operating System Agent) for command interpretation
- 🎮 ROS2 control interfaces for the Go2 robot
- 📷 Camera integration for visual feedback

## 📦 Package Contents

- **Go2 Robot Simulation**: Complete simulation environment for Unitree Go2
- **go2_ros_agent**: ROS2 node for controlling the robot via natural language

## 🛠️ Installation

### System Requirements

- 💻 Ubuntu 22.04
- 🤖 ROS2 Humble

### 🔧 Simulation Setup

1. Install ROS-based dependencies:
```bash
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-xacro
sudo apt install ros-humble-robot-localization
sudo apt install ros-humble-ros2-controllers
sudo apt install ros-humble-ros2-control
sudo apt install ros-humble-velodyne
sudo apt install ros-humble-velodyne-gazebo-plugins
sudo apt-get install ros-humble-velodyne-description
```

2. Clone the package into your workspace:
```bash
git clone https://github.com/AbdullahGM1/ros2_go2_rosa.git ~/ros2_ws/src/
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build your workspace:
```bash
colcon build
source install/setup.bash
```

### 🧠 ROSA Agent Setup

4. Install ROSA package:
```bash
pip3 install jpl-rosa
```

5. Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

6. Download the LLM model:
```bash
ollama run qwen2.5:14b
```

## 🚀 Usage

### Running the Simulation

1. Launch the Go2 simulation with RViz visualization:
```bash
ros2 launch go2_config gazebo_velodyne.launch.py rviz:=true
```

2. Run the ROSA agent node to control the robot with natural language:
```bash
ros2 run go2_ros_agent go2_agent_node
```

3. Enter commands in natural language at the prompt:
```
🧠 Your command > Walk forward 2 meters
```

## 🙏 Acknowledgments

This project builds upon the following excellent works:

- [NASA JPL ROSA](https://github.com/nasa-jpl/rosa) - ROS Operating System Agent framework
- [Unitree Go2 ROS2](https://github.com/anujjain-dev/unitree-go2-ros2) - ROS2 integration for Unitree Go2
