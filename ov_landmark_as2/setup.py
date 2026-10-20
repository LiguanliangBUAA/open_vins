from setuptools import setup

package_name = 'ov_landmark_as2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liguanliang',
    maintainer_email='guanliang.li@alumnos.upm.es',
    description='Aerostack2 detections to ov_landmark_msgs',
    license='GPL-3.0-only',
    entry_points={'console_scripts': ['as2_to_landmarks = ov_landmark_as2.as2_to_landmarks:main']},
)
