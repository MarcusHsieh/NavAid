# Launch

> Run this once
```bash
colcon build
```

> Run this in every terminal 
```bash
source install/setup.bash
```

> Terminal 1 - Core NavAid logic
```bash
ros2 launch navaid_handler navaid_stage1.launch.py
```

> Terminal 2 - IMU calibration node
> `CTRL + C` to recalibrate  
> `CTRL + Z` to terminate program
```bash
ros2 run imu_driver imu_converter
```

> Terminal 3 OR separate computer - 2D LIDAR + IMU visualization
```bash
colcon build --packages-up-to navaid_handler
source install/setup.bash
ros2 launch navaid_handler navaid_stage1_viewer.launch.py
```
OR
```bash
rviz2
# then open navaid_view.rviz config
```

> Terminal 4 OR separate computer - 3D IMU visualization  
> `imu_node` and `imu_converter` should be running already
```bash
ros2 run imu_driver imu_subscriber
```

# Misc

> RTPS Error Debug
```bash
sudo rm -f /dev/shm/fastrtps*
```