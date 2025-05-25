import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    navaid_handler_pkg_dir = get_package_share_directory('navaid_handler')

    rviz_config_file = os.path.join(navaid_handler_pkg_dir, 'rviz', 'navaid_view.rviz')

    rviz_node = Node(
       package='rviz2',
       executable='rviz2',
       name='rviz2_navaid_viewer',
       arguments=['-d', rviz_config_file],
       output='screen'
    )

    return LaunchDescription([
        rviz_node
    ])