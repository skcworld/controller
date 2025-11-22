from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='skcworld',
    maintainer_email='your_email@example.com',
    description='Python implementation of MAP, PP, and AUG controllers for autonomous racing',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller_manager = controller.controller_manager:main',
        ],
    },
)
