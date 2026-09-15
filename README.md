# Team Titanium - ERC 2026 Solution Package

This repository contains the ROS 2 Humble autonomous solution for the Emirates Robotics Competition (ERC) book-retrieval task. The architecture features custom finite-state machine control, OpenCV-based computer vision alignment, and mathematical trajectory planning.

## System Architecture & Features
* **Custom State Machine (`main_node.py`):** Manages the sequential phases of the task—including rotation, stationary scanning, diagonal approach, visual strafing, grasping, retraction, drop positioning, and bin delivery.
* **Computer Vision Pipeline:** Utilizes adaptive thresholding and template matching (`cv2.matchTemplate`) to identify shelf numbers and dynamic color masking to locate target books and the collection bin.
* **Kinematic & Collision Management:** Incorporates custom row-specific pixel offsets, early-braking momentum control, and extended 0.75-meter reverse retraction clearances to prevent physical collisions with upper shelf geometry.

---

## Package Structure
* `team_titanium/`
  * `main_node.py`: Core autonomous control and computer vision processing node.
  * `launch/solution.launch.py`: Primary ROS 2 launch file to initialize the system and pass parameters.
  * `templates/`: Contains binary template images (1-5) used for shelf column identification.
  * `erc_images/`: Directory where runtime verification screenshots are logged.

---

## Dependencies
This package depends on ROS 2 Humble and the following packages (fully declared in `package.xml`):
* `rclpy`
* `geometry_msgs` & `sensor_msgs` & `nav_msgs` & `trajectory_msgs`
* `tf2_ros`
* `cv_bridge`
* `ros_gz_interfaces`
* `OpenCV` & `NumPy`

---

## Build Instructions
Clone the workspace into your ROS 2 environment and build using `colcon`:

```bash
cd ~/erc_ws/src
git clone <your-github-repo-link> team_titanium
cd ~/erc_ws
colcon build --packages-select team_titanium --symlink-install
source install/setup.bash