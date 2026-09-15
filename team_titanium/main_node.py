import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import os
from std_msgs.msg import Int32
from datetime import datetime
from rclpy.qos import qos_profile_sensor_data

class ERCSolutionNode(Node):
    def __init__(self):
        super().__init__('erc_solution_node')
        
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.column_pub = self.create_publisher(Int32, '/erc/shelf_column_identification', 10)
        self.row_pub = self.create_publisher(Int32, '/erc/shelf_row_identification', 10)
        
        self.torso_pub = self.create_publisher(JointTrajectory, '/torso_controller/joint_trajectory', 10)
        self.arm_left_pub = self.create_publisher(JointTrajectory, '/arm_left_controller/joint_trajectory', 10)
        self.arm_right_pub = self.create_publisher(JointTrajectory, '/arm_right_controller/joint_trajectory', 10)
        self.gripper_left_pub = self.create_publisher(JointTrajectory, '/gripper_left_controller/joint_trajectory',10)
        self.head_pub = self.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)

        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/color/image_raw',
            self.image_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.declare_parameter('depth_topic', '/head_front_camera/head_front_camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/head_front_camera/head_front_camera/color/camera_info')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('gripper_frame', 'arm_left_7_link')
        self.declare_parameter('gripper_lateral_offset', 0.20)

        self.depth_sub = self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            qos_profile_sensor_data
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.info_callback,
            qos_profile_sensor_data
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.base_frame = self.get_parameter('base_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.gripper_frame = self.get_parameter('gripper_frame').value
        self.gripper_dy_fallback = float(self.get_parameter('gripper_lateral_offset').value)

        self.latest_depth = None
        self.depth_is_mm = False
        self.fx = self.fy = self.cx = self.cy = None
        self.color_frame = None
        self.color_stamp = None

        self.declare_parameter('shelf_column_number', 3)
        self.declare_parameter('book_colour', 'red')

        self.image_dir = '/opt/erc_ws/src/team_titanium/erc_images'
        os.makedirs(self.image_dir, exist_ok=True)
        self.saved_shelf_image = False
        
        self.target_shelf = str(self.get_parameter('shelf_column_number').get_parameter_value().integer_value)
        raw_color = self.get_parameter('book_colour').get_parameter_value().string_value
        self.target_color = raw_color.capitalize()
        
        self.color_bounds = {
            'Red': [(0, 100, 100), (10, 255, 255)],
            'Green': [(40, 50, 50), (80, 255, 255)],
            'Blue': [(100, 150, 0), (140, 255, 255)],
            'Yellow': [(20, 100, 100), (30, 255, 255)]
        }
        
        self.get_logger().info(f"Competition Target Received -> Column: {self.target_shelf}, Color: {self.target_color}")
        
        self.robot_state = 'ROTATING'
        self.lateral_offsets = [2.4, 1.4, 0.4, -0.5, -1.3]
        self.target_lateral = None
        
        self.current_yaw = 0.0
        self.target_yaw = None
        self.start_x = None
        self.start_y = None
        self.current_x = 0.0
        self.current_y = 0.0    
        self.odom_initialized = False
        self.start_time = None
        self.brake_start = None
        self.shift_start_time = None
        
        self.book_error_x = 0.0
        self.book_visible = False

        self.templates = {}
        base_path = "/opt/erc_ws/src/team_titanium/templates"
        for i in range(1, 6):
            path = f"{base_path}/{i}.png"
            if os.path.exists(path):
                template_img = cv2.imread(path, 0)
                _, binary_template = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY)
                self.templates[str(i)] = cv2.resize(binary_template, (50, 75))

        self.create_timer(0.1, self.control_loop)

    def calculate_torso_limits(self):
        target_fw = 1.167
        lat = self.target_lateral if self.target_lateral else 0.0
        target_hypotenuse = math.sqrt(target_fw**2 + lat**2)
        
        base_speed = 1.0 
        self.expected_travel_time = target_hypotenuse / base_speed
        
        torso_max_velocity = 0.05 
        self.max_torso_rise = torso_max_velocity * self.expected_travel_time
        
        row_heights = {1: 0.35, 2: 0.25, 3: 0.15, 4: 0.00}
        required_height = row_heights.get(getattr(self, 'final_row', 1), 0.0)
        
        if required_height > self.max_torso_rise:
            self.requires_post_stop_lift = True
        else:
            self.requires_post_stop_lift = False

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.odom_initialized = True
        
        q = msg.pose.pose.orientation
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(t3, t4)

    def info_callback(self, msg):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_is_mm = (msg.encoding == '16UC1')
            self.depth_frame = msg.header.frame_id
        except Exception as e:
            pass 

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            debug_frame = cv_image.copy()
            height, width = cv_image.shape[:2]
            
            self.color_frame = msg.header.frame_id
            self.color_stamp = msg.header.stamp
            
            cv2.putText(debug_frame, f"STATE: {self.robot_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if self.robot_state == 'STATIONARY_SCAN':
                gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
                
                cutoff_y = int(height * 0.40)  
                thresh[cutoff_y:height, :] = 0 
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if self.templates:
                    current_detections = []
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        if h == 0: continue
                        if cv2.contourArea(contour) > 40 and 0.2 < (w / float(h)) < 1.5:
                            crop = thresh[y:y+h, x:x+w]
                            crop_resized = cv2.resize(crop, (50, 75))
                            best_score, best_match = -1, None
                            
                            for label, template in self.templates.items():
                                res = cv2.matchTemplate(crop_resized, template, cv2.TM_CCOEFF_NORMED)
                                if res[0][0] > best_score:
                                    best_score, best_match = res[0][0], label
                                    
                            if best_score > 0.55:
                                current_detections.append((x, y, w, h, best_match))
                                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                
                    if len(current_detections) == 5:
                        current_detections.sort(key=lambda item: item[0])
                        for idx, (px, py, pw, ph, label) in enumerate(current_detections):
                            if label == self.target_shelf:
                                margin = 35 
                                col_x_start = max(0, px - margin)
                                col_x_end = min(width, px + pw + margin)
                                
                                column_mask = np.zeros((height, width), dtype=np.uint8)
                                column_mask[:, col_x_start:col_x_end] = 255
                                
                                hsv_full = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                                
                                target_lower = np.array(self.color_bounds[self.target_color][0], dtype=np.uint8)
                                target_upper = np.array(self.color_bounds[self.target_color][1], dtype=np.uint8)
                                target_mask = cv2.inRange(hsv_full, target_lower, target_upper)
                                
                                if self.target_color == 'Red':
                                    u_red = cv2.inRange(hsv_full, np.array([170, 50, 50]), np.array([180, 255, 255]))
                                    target_mask = cv2.bitwise_or(target_mask, u_red)
                                    
                                target_column = cv2.bitwise_and(target_mask, column_mask)
                                target_contours, _ = cv2.findContours(target_column, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                valid_target = [c for c in target_contours if cv2.contourArea(c) > 50]
                                
                                if valid_target:
                                    target_contour = max(valid_target, key=cv2.contourArea)
                                    bx, by, bw, bh = cv2.boundingRect(target_contour)
                                    target_y = by + (bh // 2)
                                    
                                    y_offset = target_y - py
                                    ratio = y_offset / float(ph)
                                    
                                    self.get_logger().info(f"Row Ratio: {ratio:.2f} (Offset: {y_offset}px, Ruler: {ph}px)")
                                    
                                    if ratio < 3.8:   
                                        self.final_row = 1
                                    elif ratio < 5.5: 
                                        self.final_row = 2
                                    elif ratio < 7.1: 
                                        self.final_row = 3
                                    else:
                                        self.final_row = 4

                                    self.target_lateral = self.lateral_offsets[idx]
                                    
                                    self.get_logger().info(f"Target matched to Row {self.final_row}. Approaching left edge.")
                                    cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
                                else:
                                    self.get_logger().warn(f"Column {label} found, but book missing! Defaulting to Row 1.")
                                    self.final_row = 1
                                    self.target_lateral = self.lateral_offsets[idx]
                                    
                                self.calculate_torso_limits()
                                col_msg = Int32()
                                col_msg.data = int(label)
                                self.column_pub.publish(col_msg)

                                row_msg = Int32()
                                row_msg.data = int(self.final_row)
                                self.row_pub.publish(row_msg)
                                
                                if not self.saved_shelf_image:
                                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                    display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    cv2.rectangle(debug_frame, (col_x_start, (py + ph)), (col_x_end, int(height * 0.95)), (0, 165, 255), 3)
                                    cv2.putText(debug_frame, f"TIMESTAMP: {display_time}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                                    filename = f"{self.image_dir}/shelf_col_{label}_{timestamp}.png"
                                    cv2.imwrite(filename, debug_frame)
                                    self.saved_shelf_image = True
                                    
                                self.robot_state = 'APPROACH_DIAGONAL'
                                break
                                
            elif self.robot_state == 'STRAFE_ALIGN':
                roi_x_start = int(width * 0.40)
                roi_x_end = int(width)
                
                roi_y_start = int(height * 0.05) 
                roi_y_end = int(height * 0.95)
                
                roi = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                lower_bound = np.array(self.color_bounds[self.target_color][0], dtype=np.uint8)
                upper_bound = np.array(self.color_bounds[self.target_color][1], dtype=np.uint8)
                color_mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
                
                if self.target_color == 'Red':
                    upper_red_mask = cv2.inRange(hsv_roi, np.array([170, 50, 50]), np.array([180, 255, 255]))
                    color_mask = cv2.bitwise_or(color_mask, upper_red_mask)
                    
                color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = [c for c in color_contours if cv2.contourArea(c) >50]
                
                if valid_contours:
                    self.book_visible = True
                    camera_center_x = width / 2.0
                    
                    def distance_to_center(c):
                        cx, cy, cw, ch = cv2.boundingRect(c)
                        return abs((cx + (cw // 2) + roi_x_start) - camera_center_x)
                        
                    target_contour = min(valid_contours, key=distance_to_center)
                    bx, by, bw, bh = cv2.boundingRect(target_contour)
                    book_center_x = bx + (bw // 2.0) + roi_x_start
                    
                    row_offsets = {
                        1: 60,  
                        2: 41,  
                        3: 14,  
                        4: 51   
                    }
                    current_row = getattr(self, 'final_row', 1)
                    pixel_offset = row_offsets.get(current_row, 60)

                    if current_row == 4 and getattr(self, 'target_shelf', '') == '3':
                        pixel_offset = 61 
                        self.get_logger().info(f"Applying Middle Shelf Override: Offset set to {pixel_offset}")
                    
                    arm_target_x = camera_center_x - pixel_offset
                    
                    self.book_error_x = book_center_x - arm_target_x
                    
                    cv2.rectangle(debug_frame, (int(bx + roi_x_start), int(by + roi_y_start)), (int(bx + bw + roi_x_start), int(by + bh + roi_y_start)), (0, 255, 0), 2)
                    cv2.line(debug_frame, (int(arm_target_x), 0), (int(arm_target_x), height), (255, 0, 255), 2)
                else:
                    self.book_visible = False

            elif self.robot_state in ['VISION_ALIGN_BIN', 'VISION_APPROACH_BIN']:
                hsv_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                
                mask1 = cv2.inRange(hsv_frame, np.array([0, 100, 100]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv_frame, np.array([170, 50, 50]), np.array([180, 255, 255]))
                bin_mask = cv2.bitwise_or(mask1, mask2)
                
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = [c for c in contours if cv2.contourArea(c) > 400]
                
                if valid_contours:
                    self.bin_visible = True
                    bin_contour = max(valid_contours, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(bin_contour)
                    
                    bin_center_x = bx + (bw // 2)
                    camera_center_x = width / 2.0
                    
    
                    target_x = camera_center_x - 170 
                    
                    self.bin_error_x = bin_center_x - target_x
                    
                    cv2.rectangle(debug_frame, (bx, by), (bx+bw, by+bh), (0, 0, 255), 3)
                    cv2.line(debug_frame, (int(target_x), 0), (int(target_x), height), (255, 255, 0), 2)
                    cv2.putText(debug_frame, "TARGET BIN LOCATED", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    self.bin_visible = False
            cv2.imshow("Robot Vision Debugger", debug_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def control_loop(self):
        msg = Twist()
        
        if self.robot_state == 'ROTATING':
            if self.odom_initialized:
                if self.target_yaw is None:
                    self.target_yaw = self.current_yaw - (math.pi / 2.0)
                    if self.target_yaw < -math.pi: self.target_yaw += 2 * math.pi
                        
                diff = self.current_yaw - self.target_yaw
                if diff > math.pi: diff = 2 * math.pi - diff
                if diff < -math.pi: diff = 2 * math.pi + diff

                if abs(diff) < 0.01:
                    msg.angular.z = 0.0
                    self.robot_state = 'STATIONARY_SCAN' 
                else:
                    msg.angular.z = -0.25
                    
        elif self.robot_state == 'STATIONARY_SCAN':
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.0
                
        elif self.robot_state == 'APPROACH_DIAGONAL':
            if self.start_x is None:
                self.start_x = self.current_x
                self.start_y = self.current_y
                self.start_time = self.get_clock().now()
                
                row = getattr(self, 'final_row', 1)
                
                if row == 1:
                    torso_msg = JointTrajectory()
                    torso_msg.joint_names = ['torso_lift_joint']
                    point = JointTrajectoryPoint()
                    point.positions = [0.3]
                    point.time_from_start = Duration(sec=6, nanosec=0)
                    torso_msg.points.append(point)
                    self.torso_pub.publish(torso_msg)
                
                right_msg = JointTrajectory()
                right_msg.joint_names = [f'arm_right_{i}_joint' for i in range(1, 8)]
                rp = JointTrajectoryPoint()
                rp.positions = [0.0, 1.1, 0.0, -1.8, 0.0, 0.0, 0.0]
                rp.time_from_start = Duration(sec=3, nanosec=0)
                right_msg.points.append(rp)
                self.arm_right_pub.publish(right_msg)
                
                left_msg = JointTrajectory()
                left_msg.joint_names = [f'arm_left_{i}_joint' for i in range(1, 8)]
                lp = JointTrajectoryPoint()
                
                if row == 1: lp.positions = [0.1, 3.0, 0.0, -1.1, -0.2, 1.6, -0.1]
                elif row == 2: lp.positions = [0.3, 3.0, 0.0, -0.5, -0.2, 1.1, -0.2]
                elif row == 3: lp.positions = [0.6, 1.0, 0.0, 0.0, 0.0, 0.3, -0.5]
                else: lp.positions = [0.7, 0.5, 0.0, 0.0, 0.0, 0.0, -0.6]
                    
                lp.time_from_start = Duration(sec=3, nanosec=0)
                left_msg.points.append(lp)
                self.arm_left_pub.publish(left_msg)

                gripper_msg = JointTrajectory()
                gripper_msg.joint_names = ['gripper_left_finger_joint']
                gp = JointTrajectoryPoint()
                gp.positions = [0.068]
                gp.time_from_start = Duration(sec=3, nanosec=0)
                gripper_msg.points.append(gp)
                self.gripper_left_pub.publish(gripper_msg)
                
                head_msg = JointTrajectory()
                head_msg.joint_names = ['head_1_joint', 'head_2_joint']
                hp = JointTrajectoryPoint()
                
                if row == 4:
                    hp.positions = [0.0, -0.7] 
                else:
                    hp.positions = [0.0, 0.0]
                    
                hp.time_from_start = Duration(sec=3, nanosec=0)
                head_msg.points.append(hp)
                self.head_pub.publish(head_msg)
                
                self.get_logger().info(f"Deploying arms and head for Row {row} concurrently with approach.")
                
            target_fw = 1.167 
            lat = self.target_lateral 
            
            braking_offset = 0.066 
            trigger_fw = target_fw - braking_offset
            
            trigger_hypotenuse = math.sqrt(trigger_fw**2 + lat**2)
            target_hypotenuse = math.sqrt(target_fw**2 + lat**2)
            
            dist_traveled = math.sqrt((self.current_x - self.start_x)**2 + (self.current_y - self.start_y)**2)
            
            if dist_traveled >= trigger_hypotenuse:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                self.brake_start = self.get_clock().now()
                self.robot_state = 'BRAKING'
            else:
                base_speed = 1.0 
                if target_hypotenuse == 0: 
                    msg.linear.x, msg.linear.y = 0.0, 0.0
                else:
                    msg.linear.x = base_speed * (target_fw / target_hypotenuse)
                    msg.linear.y = base_speed * (lat / target_hypotenuse) 
                msg.angular.z = 0.0
                
        elif self.robot_state == 'BRAKING':
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.0

            if (self.get_clock().now() - self.brake_start).nanoseconds / 1e9 > 2.5:
                self.get_logger().info("Braking complete. Initiating visual strafe.")
                self.robot_state = 'STRAFE_ALIGN'

        elif self.robot_state == 'STRAFE_ALIGN':
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            
            if getattr(self, 'book_visible', False):
                error = getattr(self, 'book_error_x', 0.0)
                
                if abs(error) <= 20:
                    self.get_logger().info("Target visually centered! Initiating blind shift...")
                    self.shift_start_time = self.get_clock().now()
                    self.robot_state = 'BLIND_SHIFT'
                else:
                    raw_speed = -error * 0.002 
                    if raw_speed > 0:
                        msg.linear.y = max(0.04, min(0.15, raw_speed))
                    else:
                        msg.linear.y = min(-0.04, max(-0.15, raw_speed))
            else:
                msg.linear.y = 0.0

        elif self.robot_state == 'BLIND_SHIFT':
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            
            if (self.get_clock().now() - self.shift_start_time).nanoseconds / 1e9 > 2.4:
                msg.linear.y = 0.0
                self.get_logger().info("Blind shift complete. Moving forward to grasp book...")
                
                self.grasp_start_x = self.current_x
                self.grasp_start_y = self.current_y
                self.robot_state = 'FORWARD_GRASP'
            else:
                msg.linear.y = -0.05

        elif self.robot_state == 'FORWARD_GRASP':
            msg.linear.y = 0.0
            msg.angular.z = 0.0
            
            dist_traveled = math.sqrt((self.current_x - getattr(self, 'grasp_start_x', self.current_x))**2 + 
                                      (self.current_y - getattr(self, 'grasp_start_y', self.current_y))**2)
            
            row_grasp_distances = {
                1: 0.81,  
                2: 0.61,  
                3: 0.44,  
                4: 0.33   
            }
            current_row = getattr(self, 'final_row', 1)
            target_distance = row_grasp_distances.get(current_row, 0.42)
            
            if dist_traveled >= target_distance:
                msg.linear.x = 0.0
                self.get_logger().info(f"Forward approach for Row {current_row} complete. Closing gripper to 0.015...")
                
                gripper_msg = JointTrajectory()
                gripper_msg.joint_names = ['gripper_left_finger_joint']
                gp = JointTrajectoryPoint()
                gp.positions = [0.015]
                gp.time_from_start = Duration(sec=2, nanosec=0)
                gripper_msg.points.append(gp)
                self.gripper_left_pub.publish(gripper_msg)
                
                self.close_start_time = self.get_clock().now()
                self.robot_state = 'CLOSE_GRIPPER'
            else:
                msg.linear.x = 0.1
                
        elif self.robot_state == 'CLOSE_GRIPPER':
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.0
            
            if (self.get_clock().now() - self.close_start_time).nanoseconds / 1e9 > 7.5:
                self.get_logger().info("Gripper secured. Retracting 50 cm...")
                self.retract_start_x = self.current_x
                self.retract_start_y = self.current_y
                self.robot_state = 'REVERSE_RETRACT'

        elif self.robot_state == 'REVERSE_RETRACT':
            msg.linear.y = 0.0
            msg.angular.z = 0.0
            
            dist_traveled = math.sqrt((self.current_x - getattr(self, 'retract_start_x', self.current_x))**2 + 
                                      (self.current_y - getattr(self, 'retract_start_y', self.current_y))**2)
            
            if dist_traveled >= 0.75:
                msg.linear.x = 0.0
                self.get_logger().info("Retraction complete. Lifting arm to safe drop pose...")
                
                torso_msg = JointTrajectory()
                torso_msg.joint_names = ['torso_lift_joint']
                tp = JointTrajectoryPoint()
                tp.positions = [0.35]
                tp.time_from_start = Duration(sec=3, nanosec=0)
                torso_msg.points.append(tp)
                self.torso_pub.publish(torso_msg)
                
                left_msg = JointTrajectory()
                left_msg.joint_names = [f'arm_left_{i}_joint' for i in range(1, 8)]
                lp = JointTrajectoryPoint()
                lp.positions = [0.1, 0.5, 0.0, -0.5, 0.0, 0.0, -0.6]
                lp.time_from_start = Duration(sec=3, nanosec=0)
                left_msg.points.append(lp)
                self.arm_left_pub.publish(left_msg)

                head_msg = JointTrajectory()
                head_msg.joint_names = ['head_1_joint', 'head_2_joint']
                hp = JointTrajectoryPoint()
                hp.positions = [0.0, -0.75]
                hp.time_from_start = Duration(sec=3, nanosec=0)
                head_msg.points.append(hp)
                self.head_pub.publish(head_msg)
                
                self.pose_start_time = self.get_clock().now()
                self.robot_state = 'PREPARE_DROP_POSE'
                self.pose_start_time = self.get_clock().now()
                self.robot_state = 'PREPARE_DROP_POSE'
            else:
                msg.linear.x = -0.15 
                
        elif self.robot_state == 'PREPARE_DROP_POSE':
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            
            if (self.get_clock().now() - self.pose_start_time).nanoseconds / 1e9 > 3.5:
                
                self.turn_target_yaw = math.pi / 2.0
                self.robot_state = 'ROTATE_FACE_TABLE'

        elif self.robot_state == 'ROTATE_FACE_TABLE':
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            
            diff = self.current_yaw - getattr(self, 'turn_target_yaw', self.current_yaw)
            if diff > math.pi: diff -= 2 * math.pi
            if diff < -math.pi: diff += 2 * math.pi

            if abs(diff) < 0.02:
                msg.angular.z = 0.0
                self.get_logger().info("Facing table. Initiating diagonal crab-walk to waypoint.")
                
                self.waypoint_x = 0.20
                self.waypoint_y = -0.10
                self.robot_state = 'DIAGONAL_STRAFE_TO_WAYPOINT'
            else:
                msg.angular.z = max(-0.4, min(0.4, -diff * 1.5))
                
        elif self.robot_state == 'DIAGONAL_STRAFE_TO_WAYPOINT':

            msg.angular.z = 0.0
            
            error_x = self.waypoint_x - self.current_x
            error_y = self.waypoint_y - self.current_y
            dist_to_wp = math.sqrt(error_x**2 + error_y**2)
            
            if dist_to_wp <= 0.05: 
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                self.get_logger().info("Arrived at waypoint. Handing over to CV for bin alignment.")
                self.robot_state = 'VISION_ALIGN_BIN'
            else:
                speed = 0.25
                global_vx = speed * (error_x / dist_to_wp)
                global_vy = speed * (error_y / dist_to_wp)
                
                msg.linear.x = global_vx * math.cos(self.current_yaw) + global_vy * math.sin(self.current_yaw)
                msg.linear.y = -global_vx * math.sin(self.current_yaw) + global_vy * math.cos(self.current_yaw)

        elif self.robot_state == 'VISION_ALIGN_BIN':
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            
            if getattr(self, 'bin_visible', False):
                error = getattr(self, 'bin_error_x', 0.0)
                
                if abs(error) <= 50: 
                    msg.linear.y = 0.0
                    self.get_logger().info("Braking early to coast into alignment!")
                    self.robot_state = 'VISION_APPROACH_BIN'
                else:
                    raw_speed = -error * 0.002 
                    if raw_speed > 0:
                        msg.linear.y = max(0.04, min(0.15, raw_speed))
                    else:
                        msg.linear.y = min(-0.04, max(-0.15, raw_speed))
            else:
                msg.linear.y = 0.0
                self.get_logger().warn("Searching for red bin...")

        elif self.robot_state == 'VISION_APPROACH_BIN':
            msg.angular.z = 0.0
            msg.linear.y = 0.0
            
            if self.current_y >= 0.10:
                msg.linear.x = 0.0
                
                self.vel_pub.publish(msg) 
                
                self.get_logger().info("In position over the bin. Releasing book!")
                
                gripper_msg = JointTrajectory()
                gripper_msg.joint_names = ['gripper_left_finger_joint']
                gp = JointTrajectoryPoint()
                gp.positions = [0.068]
                gp.time_from_start = Duration(sec=2, nanosec=0)
                gripper_msg.points.append(gp)
                self.gripper_left_pub.publish(gripper_msg)
                
                self.drop_start_time = self.get_clock().now()
                self.robot_state = 'DROP_BOOK'
            else:
                msg.linear.x = 0.06
                
        elif self.robot_state == 'DROP_BOOK':
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.0
            
            if (self.get_clock().now() - self.drop_start_time).nanoseconds / 1e9 > 2.5:
                self.get_logger().info("Task Complete! Book delivered.")
                self.robot_state = 'TASK_COMPLETE'
                                
        elif self.robot_state == 'TASK_COMPLETE':
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.0
            
            col_msg = Int32()
            col_msg.data = int(self.target_shelf)
            self.column_pub.publish(col_msg)
            
            if hasattr(self, 'final_row'):
                row_msg = Int32()
                row_msg.data = self.final_row
                self.row_pub.publish(row_msg)
                
        self.vel_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ERCSolutionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()