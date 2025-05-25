from setuptools import setup
import os
from glob import glob

package_name = 'navaid_handler'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools', 'numpy', 'Jetson.GPIO', 'tf-transformations'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='marcus.j.hsieh@gmail.com',
    description='NavAid core logic and LED control',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navaid_processor = navaid_handler.navaid_processor_node:main',
            'led_controller = navaid_handler.led_controller_node:main',
        ],
    },
)
