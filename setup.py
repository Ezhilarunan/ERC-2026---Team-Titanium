import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'team_titanium'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Registers your launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Registers your future OpenCV template images
        (os.path.join('share', package_name, 'templates'), glob(os.path.join('templates', '*.png'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='ERC 2026 Phase 1 Solution',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_node = team_titanium.main_node:main',
        ],
    },
)