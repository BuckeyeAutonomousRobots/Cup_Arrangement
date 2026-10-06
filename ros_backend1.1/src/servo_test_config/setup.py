from setuptools import setup
import os
from glob import glob

package_name = "servo_test_config"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="User",
    maintainer_email="user@example.com",
    description="Minimal UR5e fake-hardware + MoveIt Servo test launch.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "joint_states_filter = servo_test_config.joint_states_filter:main",
        ],
    },
)
