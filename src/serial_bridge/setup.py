from setuptools import setup

package_name = 'serial_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@email.com',
    description='Serial bridge between ROS2 and ESP32 motor controller',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'serial_bridge_node = serial_bridge.serial_bridge_node:main',
        ],
    },
)
