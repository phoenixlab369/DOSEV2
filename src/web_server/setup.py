from setuptools import setup

package_name = 'web_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'fastapi',
        'uvicorn',
        'websockets'
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@email.com',
    description='FastAPI web server publishing ROS2 cmd_vel',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'web_ros2_server = web_server.web_ros2_server:main',
        ],
    },
)
