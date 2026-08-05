#!/usr/bin/env python3

##################################################
# Imports
##################################################

import os
import cv2
import math
import random
import numpy as np
import torch
import rospy

from PIL import Image

from transformers import (
    AutoImageProcessor,
    VideoMAEModel
)

from ultralytics import YOLO

from cv_bridge import CvBridge

from geometry_msgs.msg import (
    Twist,
    Pose
)

from sensor_msgs.msg import (
    LaserScan,
    Image as ROSImage
)

from nav_msgs.msg import Odometry

from std_srvs.srv import Empty

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
    GetModelState
)

from graph_builder import build_graph


##################################################
# Goal Model
##################################################

goal_model_dir = os.path.join(
    os.path.split(os.path.realpath(__file__))[0],
    "..",
    "..",
    "turtlebot3_simulations",
    "turtlebot3_gazebo",
    "models",
    "Target",
    "model.sdf"
)


##################################################
# Pepper Environment
##################################################

class PepperEnvironment:

    ##################################################
    # Initialization
    ##################################################

    def __init__(self, training=True):

        ##################################################
        # Mode
        ##################################################

        self.training = training

        ##################################################
        # Robot State
        ##################################################

        self.robot_pose = Pose()

        self.goal_pose = Pose()

        self.yaw = 0.0

        self.linear_velocity = 0.0

        self.angular_velocity = 0.0

        self.previous_linear_velocity = 0.0

        self.previous_angular_velocity = 0.0

        self.previous_distance = 0.0

        ##################################################
        # Episode
        ##################################################

        self.current_step = 0

        self.max_steps = 500

        self.goal_threshold = 0.20 if training else 0.40

        ##################################################
        # Reward Parameters
        ##################################################

        self.Rg = 100.0

        self.Rf = 100.0

        self.lambda_p = 35.0

        self.lambda_d = 8.0

        self.lambda_s = 10.0

        ##################################################
        # Camera
        ##################################################

        self.bridge = CvBridge()

        self.current_frame = None

        ##################################################
        # Video Buffer
        ##################################################

        self.video_buffer = []

        self.clip_length = 16

        ##################################################
        # Device
        ##################################################

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else
            "cpu"
        )

        ##################################################
        # VideoMAE
        ##################################################

        self.video_processor = AutoImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base"
        )

        self.video_model = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-base"
        ).to(
            self.device
        )

        self.video_model.eval()

        ##################################################
        # YOLOv11 Human Detector
        ##################################################

        self.human_detector = YOLO(
            "yolo11n.pt"
        )

        ##################################################
        # Social Databases
        ##################################################

        self.positive_database = self.load_database(
            "social_database/positive"
        )

        self.negative_database = self.load_database(
            "social_database/negative"
        )

        ##################################################
        # Target Management
        ##################################################

        self.training_targets = []

        self.last_target = None

        ##################################################
        # ROS Publisher
        ##################################################

        self.cmd_pub = rospy.Publisher(
            "/cmd_vel",
            Twist,
            queue_size=10
        )

        ##################################################
        # ROS Subscribers
        ##################################################

        self.odom_sub = rospy.Subscriber(
            "/odom",
            Odometry,
            self.odom_callback
        )

        self.camera_sub = rospy.Subscriber(
            "/camera/rgb/image_raw",
            ROSImage,
            self.camera_callback
        )

        ##################################################
        # Gazebo Services
        ##################################################

        self.reset_world = rospy.ServiceProxy(
            "/gazebo/reset_world",
            Empty
        )

        self.spawn_goal = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model",
            SpawnModel
        )

        self.delete_goal = rospy.ServiceProxy(
            "/gazebo/delete_model",
            DeleteModel
        )

        self.get_model_state = rospy.ServiceProxy(
            "/gazebo/get_model_state",
            GetModelState
        )

    ##################################################
    # Odometry Callback
    ##################################################

    def odom_callback(self, msg):

        self.robot_pose = msg.pose.pose

        orientation = self.robot_pose.orientation

        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w

        self.yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )

    ##################################################
    # Camera Callback
    ##################################################

    def camera_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            self.current_frame = frame

            ##################################################
            # Update Video Buffer
            ##################################################

            self.video_buffer.append(frame)

            if len(self.video_buffer) > self.clip_length:
                self.video_buffer.pop(0)

        except Exception as e:
            rospy.logwarn(e)

    ##################################################
    # Capture Current Video Clip
    ##################################################

    def capture_video_clip(self):

        if len(self.video_buffer) < self.clip_length:
            return None

        return self.video_buffer.copy()

    ##################################################
    # Load Positive / Negative Database
    ##################################################

    def load_database(self, folder):

        database = []

        if not os.path.exists(folder):
            rospy.logwarn(f"{folder} not found.")
            return database

        ##################################################
        # Compute VideoMAE embedding
        # only once for every video
        ##################################################

        for filename in sorted(os.listdir(folder)):

            path = os.path.join(
                folder,
                filename
            )

            if not os.path.isfile(path):
                continue

            embedding = self.compute_video_embedding(
                path
            )

            if embedding is not None:
                database.append(embedding)

        rospy.loginfo(
            f"{len(database)} videos loaded from {folder}"
        )

        return database

    ##################################################
    # Compute Video Embedding
    ##################################################

    def compute_video_embedding(self, video_source):

        ##################################################
        # Read Frames
        ##################################################

        if isinstance(video_source, str):

            cap = cv2.VideoCapture(video_source)

            frames = []

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frames.append(frame)

            cap.release()

        else:

            frames = []

            for frame in video_source:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frames.append(frame)

        ##################################################
        # Enough Frames ?
        ##################################################

        if len(frames) < self.clip_length:
            return None

        ##################################################
        # Uniform Sampling
        ##################################################

        indices = np.linspace(
            0,
            len(frames) - 1,
            self.clip_length,
            dtype=int
        )

        sampled_frames = [
            frames[i]
            for i in indices
        ]

        ##################################################
        # VideoMAE Preprocessing
        ##################################################

        inputs = self.video_processor(
            sampled_frames,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(
            self.device
        )

        ##################################################
        # VideoMAE Forward
        ##################################################

        with torch.no_grad():

            outputs = self.video_model(
                pixel_values
            )

            embedding = outputs.last_hidden_state.mean(
                dim=1
            )

            embedding = embedding / embedding.norm(
                dim=1,
                keepdim=True
            )

        return embedding.squeeze(0)

    ##################################################
    # Human Detection
    ##################################################

    def detect_humans(self, frame):

        results = self.human_detector(
            frame,
            verbose=False
        )

        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:
                    return True

        return False

    ##################################################
    # Compare With Database
    ##################################################

    def compare_database(
            self,
            current_embedding,
            database
    ):

        if current_embedding is None:
            return 0.0

        similarities = []

        for reference_embedding in database:

            similarity = torch.nn.functional.cosine_similarity(
                current_embedding.unsqueeze(0),
                reference_embedding.unsqueeze(0)
            )

            similarities.append(
                similarity.item()
            )

        if len(similarities) == 0:
            return 0.0

        return max(similarities)

    ##################################################
    # Compute Social Similarity
    ##################################################

    def compute_social_similarity(self):

        ##################################################
        # Current Video
        ##################################################

        current_video = self.capture_video_clip()

        if current_video is None:
            return 0.0, 0.0, False

        ##################################################
        # Human Detection
        ##################################################

        human_detected = self.detect_humans(
            current_video[-1]
        )

        if not human_detected:
            return 0.0, 0.0, False

        ##################################################
        # Video Embedding
        ##################################################

        current_embedding = self.compute_video_embedding(
            current_video
        )

        ##################################################
        # Positive Similarity
        ##################################################

        positive_similarity = self.compare_database(
            current_embedding,
            self.positive_database
        )

        ##################################################
        # Negative Similarity
        ##################################################

        negative_similarity = self.compare_database(
            current_embedding,
            self.negative_database
        )

        ##################################################
        # Return
        ##################################################

        return (
            positive_similarity,
            negative_similarity,
            True
        )

    ##################################################
    # Build Graph State
    ##################################################

    def build_state_graph(self, laser_scan):

        ##################################################
        # Robot Node
        ##################################################

        robot_node = {
            "position": [
                self.robot_pose.position.x,
                self.robot_pose.position.y
            ],
            "orientation": self.yaw,
            "linear_velocity": self.linear_velocity,
            "angular_velocity": self.angular_velocity
        }

        ##################################################
        # Goal Node
        ##################################################

        target_node = {
            "position": [
                self.goal_pose.position.x,
                self.goal_pose.position.y
            ]
        }

        ##################################################
        # Obstacle Nodes
        ##################################################

        obstacle_nodes = []

        angle = laser_scan.angle_min

        min_obstacle_distance = float("inf")

        for distance in laser_scan.ranges:

            if np.isinf(distance) or np.isnan(distance):
                angle += laser_scan.angle_increment
                continue

            x = self.robot_pose.position.x + distance * math.cos(
                self.yaw + angle
            )

            y = self.robot_pose.position.y + distance * math.sin(
                self.yaw + angle
            )

            obstacle_nodes.append({
                "position": [x, y],
                "distance": distance
            })

            min_obstacle_distance = min(
                min_obstacle_distance,
                distance
            )

            angle += laser_scan.angle_increment

        ##################################################
        # Human Nodes
        ##################################################

        human_nodes = []

        if self.current_frame is not None:

            results = self.human_detector(
                self.current_frame,
                verbose=False
            )

            for result in results:
                for box in result.boxes:

                    if int(box.cls) != 0:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    cx = (x1 + x2) / 2.0

                    cy = (y1 + y2) / 2.0

                    ##################################################
                    # Approximate human position
                    ##################################################

                    human_nodes.append({
                        "position": [cx, cy]
                    })

        ##################################################
        # Graph
        ##################################################

        graph_state = build_graph(
            robot_node,
            target_node,
            obstacle_nodes,
            human_nodes
        )

        return graph_state, min_obstacle_distance

    ##################################################
    # Get Current State
    ##################################################

    def getState(self, scan):

        done = False

        arrive = False

        ##################################################
        # Build Graph
        ##################################################

        graph_state, min_obstacle_distance = self.build_state_graph(
            scan
        )

        ##################################################
        # Distance To Goal
        ##################################################

        current_distance = math.hypot(
            self.goal_pose.position.x - self.robot_pose.position.x,
            self.goal_pose.position.y - self.robot_pose.position.y
        )

        ##################################################
        # Collision
        ##################################################

        if min_obstacle_distance < 0.20:
            done = True

        ##################################################
        # Goal Reached
        ##################################################

        if current_distance < self.goal_threshold:
            arrive = True

        ##################################################
        # Timeout
        ##################################################

        timeout = False

        if self.current_step >= self.max_steps:
            timeout = True
            done = True

        ##################################################
        # Return
        ##################################################

        return (
            graph_state,
            current_distance,
            min_obstacle_distance,
            timeout,
            done,
            arrive
        )

    ##################################################
    # Reward Function
    ##################################################

    def compute_reward(self,
                       current_distance,
                       target_reached,
                       timeout,
                       current_v,
                       current_w,
                       positive_similarity,
                       negative_similarity,
                       human_detected):

        ##################################################
        # Goal Reward
        ##################################################

        r_goal = self.Rg if target_reached else 0.0

        ##################################################
        # Progress Reward
        ##################################################

        r_progress = self.lambda_p * (
            self.previous_distance - current_distance
        )

        ##################################################
        # Failure Reward
        ##################################################

        r_failure = -self.Rf if timeout else 0.0

        ##################################################
        # Damage Reward
        ##################################################

        r_damage = -self.lambda_d * (
            abs(current_v - self.previous_linear_velocity) +
            abs(current_w - self.previous_angular_velocity)
        )

        ##################################################
        # Social Reward
        ##################################################

        r_social = self.lambda_s * (
            positive_similarity - negative_similarity
        )

        ##################################################
        # Human Indicator
        ##################################################

        delta = 1 if human_detected else 0

        ##################################################
        # Global Reward
        ##################################################

        reward = (
            r_goal +
            r_progress +
            r_failure +
            r_damage +
            delta * r_social
        )

        ##################################################
        # Update Previous Values
        ##################################################

        self.previous_distance = current_distance

        self.previous_linear_velocity = current_v

        self.previous_angular_velocity = current_w

        return reward

    ##################################################
    # Generate Training / Testing Goal
    ##################################################

    def generate_goal(self):

        ##################################################
        # Remove Previous Goal
        ##################################################

        try:
            self.delete_goal("Target")
        except:
            pass

        ##################################################
        # Build 64 Intelligent Targets
        ##################################################

        if len(self.training_targets) == 0:
            for block_x in [-8, -4, 0, 4]:
                for block_y in [-8, -4, 0, 4]:
                    self.training_targets.extend([
                        (block_x + 1.0, block_y + 1.0),
                        (block_x + 3.0, block_y + 1.0),
                        (block_x + 1.0, block_y + 3.0),
                        (block_x + 3.0, block_y + 3.0)
                    ])

        ##################################################
        # Training
        ##################################################

        if self.training:

            if self.last_target is None:
                self.last_target = random.choice(
                    self.training_targets
                )
            else:
                MIN_DISTANCE = 4.0
                candidates = []

                for target in self.training_targets:
                    d = math.hypot(
                        target[0] - self.last_target[0],
                        target[1] - self.last_target[1]
                    )

                    if d >= MIN_DISTANCE:
                        candidates.append(target)

                if len(candidates) == 0:
                    candidates = self.training_targets

                self.last_target = random.choice(
                    candidates
                )

            x = self.last_target[0]
            y = self.last_target[1]

        ##################################################
        # Testing
        ##################################################

        else:

            while True:
                x = random.uniform(-8.0, 8.0)
                y = random.uniform(-8.0, 8.0)

                d = math.hypot(
                    x - self.robot_pose.position.x,
                    y - self.robot_pose.position.y
                )

                if d >= 3.0:
                    break

        ##################################################
        # Goal Pose
        ##################################################

        self.goal_pose.position.x = x

        self.goal_pose.position.y = y

        self.goal_pose.position.z = 0.0

        ##################################################
        # Spawn Goal
        ##################################################

        with open(goal_model_dir, "r") as f:
            goal_xml = f.read()

        self.spawn_goal(
            "Target",
            goal_xml,
            "",
            self.goal_pose,
            "world"
        )

    ##################################################
    # Environment Step
    ##################################################

    def step(self, action):

        ##################################################
        # Execute PPO Action
        ##################################################

        self.linear_velocity = float(action[0])

        self.angular_velocity = float(action[1])

        cmd = Twist()

        cmd.linear.x = self.linear_velocity

        cmd.angular.z = self.angular_velocity

        self.cmd_pub.publish(cmd)

        rospy.sleep(0.10)

        self.current_step += 1

        ##################################################
        # Read Sensors
        ##################################################

        scan = rospy.wait_for_message(
            "/scan",
            LaserScan
        )

        ##################################################
        # Build Current State
        ##################################################

        (
            graph_state,
            current_distance,
            min_obstacle_distance,
            timeout,
            done,
            arrive
        ) = self.getState(scan)

        ##################################################
        # Social Similarity
        ##################################################

        (
            positive_similarity,
            negative_similarity,
            human_detected
        ) = self.compute_social_similarity()

        ##################################################
        # Compute Reward
        ##################################################

        reward = self.compute_reward(
            current_distance,
            arrive,
            timeout,
            self.linear_velocity,
            self.angular_velocity,
            positive_similarity,
            negative_similarity,
            human_detected
        )

        ##################################################
        # Stop Robot if Episode Finished
        ##################################################

        if done:
            stop = Twist()
            self.cmd_pub.publish(stop)

        ##################################################
        # Additional Information
        ##################################################

        info = {
            "goal_distance": current_distance,
            "closest_obstacle": min_obstacle_distance,
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
            "human_detected": human_detected,
            "target_reached": arrive,
            "timeout": timeout,
            "step": self.current_step
        }

        ##################################################
        # Return
        ##################################################

        return (
            graph_state,
            reward,
            done,
            arrive,
            info
        )

    ##################################################
    # Reset Environment
    ##################################################

    def reset(self):

        ##################################################
        # Stop Pepper
        ##################################################

        stop = Twist()

        self.cmd_pub.publish(stop)

        rospy.sleep(0.5)

        ##################################################
        # Reset Gazebo
        ##################################################

        self.reset_world()

        ##################################################
        # Reset Episode Variables
        ##################################################

        self.current_step = 0

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.previous_linear_velocity = 0.0
        self.previous_angular_velocity = 0.0

        self.previous_distance = 0.0

        ##################################################
        # Clear Video Buffer
        ##################################################

        self.video_buffer.clear()

        ##################################################
        # Generate New Goal
        ##################################################

        self.generate_goal()

        ##################################################
        # Wait Until Camera Ready
        ##################################################

        while self.current_frame is None:
            rospy.sleep(0.05)

        ##################################################
        # Read Initial Laser Scan
        ##################################################

        scan = rospy.wait_for_message(
            "/scan",
            LaserScan
        )

        ##################################################
        # Initial Graph State
        ##################################################

        (
            graph_state,
            current_distance,
            _,
            _,
            _,
            _
        ) = self.getState(scan)

        ##################################################
        # Initialize Previous Distance
        ##################################################

        self.previous_distance = current_distance

        ##################################################
        # Return Initial State
        ##################################################

        return graph_state
