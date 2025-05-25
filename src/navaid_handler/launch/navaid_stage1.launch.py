import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    ldlidar_pkg_dir = get_package_share_directory('ldlidar_sl_ros2')

    lidar_port_arg = DeclareLaunchArgument(
        'lidar_port',
        default_value='/dev/ttyUSB0',
        description='Serial port for the LDLIDAR'
    )
    lidar_port = LaunchConfiguration('lidar_port')

    # 1. LIDAR Node
    #    pub base_link -> base_laser TF
    ldlidar_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ldlidar_pkg_dir, 'launch', 'ld14p.launch.py')
        ),
        launch_arguments={'port_name': lidar_port}.items()
    )

    # 2. IMU Publisher Node
    imu_publisher_node = Node(
        package='imu_driver',
        executable='imu_node',
        name='physical_imu_publisher',
        output='screen'
    )

    ### RUN THIS IN SEPARATE TERMINAL BY ITSELF ###
    # 3. IMU Converter Node
    # imu_converter_node = Node(
    #     package='imu_driver',
    #     executable='imu_converter',
    #     name='imu_data_converter',
    #     output='screen'
    # )
    
    # 4. Static Transform for IMU (base_link -> imu_link)
    static_tf_imu_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_imu_link_tf_pub',
        arguments=['0', '0', '0.05', '0', '0', '0', 'base_link', 'imu_link'] 
                  # x   y   z     yaw pitch roll parent     child
    )

    # 5. NavAid Processor Node
    navaid_processor = Node(
        package='navaid_handler',
        executable='navaid_processor',
        name='navaid_processor_node',
        output='screen'
    )

    # 6. LED Controller Node
    led_controller = Node(
        package='navaid_handler',
        executable='led_controller',
        name='led_controller_node',
        output='screen'
        # prefix=['sudo -E'],
    )

    return LaunchDescription([
        lidar_port_arg,
        ldlidar_launch_include,
        imu_publisher_node,
        # imu_converter_node,
        static_tf_imu_node,
        navaid_processor,
        led_controller
    ])