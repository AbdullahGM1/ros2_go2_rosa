from setuptools import find_packages, setup

package_name = 'go2_ros_agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AbdullagGN1',
    maintainer_email='agm.musalami@gmail.com',
    description='ROS2 Agent for Unitree Go2 Quadruped Robot with Natural Language Control using Large Language Models',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'go2_agent_node = go2_ros_agent.go2_agent_node:main'
        ],
    },
)
