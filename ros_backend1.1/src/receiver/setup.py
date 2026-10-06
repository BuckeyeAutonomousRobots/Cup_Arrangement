from setuptools import setup

package_name = 'receiver'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Standalone Quest UDP receiver node for teleop bridge',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quest_controller_receiver = receiver.quest_controller_receiver:main',
        ],
    },
)
