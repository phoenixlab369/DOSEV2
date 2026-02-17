from setuptools import setup

package_name = 'oakd_demo'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools', 'rclpy', 'sensor_msgs', 'cv_bridge', 'opencv-python'],
    zip_safe=True,
    author='You',
    maintainer='You',
    description='OAK-D Lite publisher and subscriber',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'oakd_publisher = oakd_demo.oakd_publisher:main',
            'oakd_subscriber = oakd_demo.oakd_subscriber:main',
        ],
    },
)
