"""
ROS2 Agent for Unitree Go2 Robot
--------------------------------
This package provides a ROSA (ROS Operating System Agent) for controlling
the Unitree Go2 quadruped robot using natural language commands.
"""

from .go2_agent_node import Go2AgentNode

__all__ = ['Go2AgentNode']
