#!/usr/bin/env python3

##################################################
# environment.py
################################################## 

##################################################
##### Guidance for MotionBERT #####

# MotionBERT files required by this code are organized as follows:
# infer_wild.py is the official inference script used to convert the extracted
# 2D human pose sequence into a 3D pose sequence. configs/pose3d/MB_ft_h36m.yaml
# defines the MotionBERT model and inference configuration used by the script.
# lib/ contains the complete core MotionBERT library required by the inference
# pipeline, while params/ contains the framework parameters used by its model
# and processing components. checkpoint/pose3d/FT_MB_release_MB_ft_h36m/
# best_epoch.bin is the pretrained MotionBERT checkpoint loaded to perform
# inference with the specified configuration. Finally, requirements.txt lists
# the Python dependencies required to run these MotionBERT components.

##################################################

import os
import cv2
import json
import math
import random
import subprocess
import tempfile

import numpy as np
import torch
import rospy

from transformers import (
    AutoImageProcessor,
    VideoMAEModel,
    TimesformerModel
)

from ultralytics import YOLO

from cv_bridge import CvBridge

from geometry_msgs.msg import Twist, Pose

from sensor_msgs.msg import LaserScan, Image as ROSImage

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

        self.goal_threshold = (
            0.20 if training else 0.40
        )

        ##################################################
        # Reward
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
        # 20-Frame Video Buffer
        ##################################################

        self.video_buffer = []

        self.clip_length = 20

        ##################################################
        # Device
        ##################################################

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        ##################################################
        # VideoMAE
        ##################################################

        self.video_processor = (
            AutoImageProcessor.from_pretrained(
                "MCG-NJU/videomae-base"
            )
        )

        self.video_model = (
            VideoMAEModel.from_pretrained(
                "MCG-NJU/videomae-base"
            ).to(self.device)
        )

        self.video_model.eval()

        ##################################################
        # TimeSformer + IMU Regression
        ##################################################

        self.imu_model_path = os.path.join(
            os.path.split(os.path.realpath(__file__))[0],
            "timesformer_imu_regressor.pt"
        )

        self.imu_processor = (
            AutoImageProcessor.from_pretrained(
                "facebook/timesformer-hr-finetuned-k400"
            )
        )

        self.imu_timesformer = (
            TimesformerModel.from_pretrained(
                "facebook/timesformer-hr-finetuned-k400"
            ).to(self.device)
        )

        self.imu_timesformer.eval()

        self.imu_regression_head = torch.nn.Sequential(

            torch.nn.Linear(
                self.imu_timesformer.config.hidden_size,
                512
            ),

            torch.nn.LayerNorm(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),

            torch.nn.Linear(512, 256),

            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),

            torch.nn.Linear(256, 128),

            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),

            torch.nn.Linear(128, 64),
            torch.nn.GELU(),

            torch.nn.Linear(64, 6)

        ).to(self.device)

        checkpoint = torch.load(
            self.imu_model_path,
            map_location=self.device
        )

        state_dict = checkpoint["model_state_dict"]

        regression_head_state = {
            key.replace(
                "regression_head.",
                ""
            ): value

            for key, value in state_dict.items()

            if key.startswith(
                "regression_head."
            )
        }

        self.imu_regression_head.load_state_dict(
            regression_head_state
        )

        for parameter in self.imu_timesformer.parameters():
            parameter.requires_grad = False

        for parameter in self.imu_regression_head.parameters():
            parameter.requires_grad = False

        self.imu_regression_head.eval()

        ##################################################
        # YOLO Human Detector
        ##################################################

        self.human_detector = YOLO(
            "yolo11n.pt"
        )

        ##################################################
        # YOLO Pose Detector
        ##################################################

        self.human_pose_detector = YOLO(
            "yolo11n-pose.pt"
        )

        ##################################################
        # MotionBERT
        ##################################################

        self.motionbert_root = os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)
            ),
            "MotionBERT"
        )

        self.motionbert_infer = os.path.join(
            self.motionbert_root,
            "infer_wild.py"
        )

        self.motionbert_config = os.path.join(
            self.motionbert_root,
            "configs",
            "pose3d",
            "MB_ft_h36m.yaml"
        )

        ##################################################
        # Fine-tuned H36M checkpoint
        ##################################################

        self.motionbert_checkpoint = os.path.join(
            self.motionbert_root,
            "checkpoint",
            "pose3d",
            "FT_MB_release_MB_ft_h36m",
            "best_epoch.bin"
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
            2.0 * (
                qw * qz +
                qx * qy
            ),
            1.0 -
            2.0 * (
                qy * qy +
                qz * qz
            )
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

            self.video_buffer.append(frame)

            if len(self.video_buffer) > self.clip_length:

                self.video_buffer.pop(0)

        except Exception as e:

            rospy.logwarn(
                str(e)
            )

    ##################################################
    # Capture Video Clip
    ##################################################

    def capture_video_clip(self):

        if len(self.video_buffer) < self.clip_length:

            return None

        return self.video_buffer.copy()

    ##################################################
    # Load Database
    ##################################################

    def load_database(self, folder):

        database = []

        if not os.path.exists(folder):

            rospy.logwarn(
                f"{folder} not found."
            )

            return database

        for filename in sorted(
            os.listdir(folder)
        ):

            path = os.path.join(
                folder,
                filename
            )

            if not os.path.isfile(path):

                continue

            embedding = (
                self.compute_video_embedding(
                    path
                )
            )

            if embedding is not None:

                database.append(
                    embedding
                )

        rospy.loginfo(
            f"{len(database)} videos loaded from {folder}"
        )

        return database

    ##################################################
    # VideoMAE Embedding
    ##################################################

    def compute_video_embedding(
            self,
            video_source
    ):

        if isinstance(
            video_source,
            str
        ):

            cap = cv2.VideoCapture(
                video_source
            )

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

        if len(frames) < self.clip_length:

            return None

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

        inputs = self.video_processor(
            sampled_frames,
            return_tensors="pt"
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(self.device)
        )

        with torch.no_grad():

            outputs = self.video_model(
                pixel_values
            )

            embedding = (
                outputs.last_hidden_state
                .mean(dim=1)
            )

            embedding = (
                embedding /
                embedding.norm(
                    dim=1,
                    keepdim=True
                )
            )

        return embedding.squeeze(0)

    ##################################################
    # TimeSformer → IMU
    ##################################################

    def predict_human_imu(self):

        if len(self.video_buffer) < self.clip_length:

            return np.zeros(
                6,
                dtype=np.float32
            )

        frames = self.video_buffer[
            -self.clip_length:
        ]

        indices = np.linspace(
            0,
            len(frames) - 1,
            16,
            dtype=int
        )

        sampled_frames = [

            cv2.cvtColor(
                frames[i],
                cv2.COLOR_BGR2RGB
            )

            for i in indices

        ]

        inputs = self.imu_processor(
            sampled_frames,
            return_tensors="pt"
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(self.device)
        )

        with torch.no_grad():

            outputs = self.imu_timesformer(
                pixel_values=pixel_values
            )

            embedding = (
                outputs.last_hidden_state
                .mean(dim=1)
            )

            imu_prediction = (
                self.imu_regression_head(
                    embedding
                )
            )

        return (
            imu_prediction
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

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
    # YOLO Pose → Halpe-26
    # YOLO Pose → COCO-17 → Halpe-26 → official MotionBERT → H36M 17×3D → 1020-D human feature → GNN.
    ##################################################

    def coco17_to_halpe26(
            self,
            coco
    ):

        output = np.zeros(
            (26, 3),
            dtype=np.float32
        )

        output[0] = coco[0]
        output[1] = coco[1]
        output[2] = coco[2]
        output[3] = coco[3]
        output[4] = coco[4]

        output[5] = coco[5]
        output[6] = coco[6]

        output[7] = coco[7]
        output[8] = coco[8]

        output[9] = coco[9]
        output[10] = coco[10]

        output[11] = coco[11]
        output[12] = coco[12]

        output[13] = coco[13]
        output[14] = coco[14]

        output[15] = coco[15]
        output[16] = coco[16]

        output[17] = (
            coco[1] +
            coco[2]
        ) * 0.5

        output[18] = (
            coco[5] +
            coco[6]
        ) * 0.5

        output[19] = (
            coco[11] +
            coco[12]
        ) * 0.5

        output[20] = coco[15]
        output[21] = coco[16]

        output[22] = coco[15]
        output[23] = coco[16]

        output[24] = coco[15]
        output[25] = coco[16]

        return output

    ##################################################
    # Extract 2D Pose Sequence
    ##################################################

    def extract_motionbert_pose(self):

        if len(self.video_buffer) < self.clip_length:

            return None, None

        frames = self.video_buffer[
            -self.clip_length:
        ]

        pose_sequence = []

        selected_person = None

        for frame_id, frame in enumerate(frames):

            results = self.human_pose_detector(
                frame,
                verbose=False
            )

            best_person = None
            best_confidence = -1.0

            for result in results:

                if result.keypoints is None:
                    continue

                if result.keypoints.xy is None:
                    continue

                keypoints = (
                    result.keypoints.xy
                    .cpu()
                    .numpy()
                )

                if len(keypoints) == 0:
                    continue

                if result.boxes is not None:

                    confidences = (
                        result.boxes.conf
                        .cpu()
                        .numpy()
                    )

                    person_id = int(
                        np.argmax(
                            confidences
                        )
                    )

                    confidence = float(
                        confidences[person_id]
                    )

                else:

                    person_id = 0
                    confidence = 1.0

                if confidence > best_confidence:

                    best_confidence = confidence

                    best_person = (
                        keypoints[person_id]
                    )

            if best_person is None:

                if selected_person is None:

                    return None, None

                best_person = selected_person.copy()

            else:

                selected_person = (
                    best_person.copy()
                )

            coco17 = np.zeros(
                (17, 3),
                dtype=np.float32
            )

            coco17[:, :2] = (
                best_person[:, :2]
            )

            coco17[:, 2] = 1.0

            halpe26 = (
                self.coco17_to_halpe26(
                    coco17
                )
            )

            pose_sequence.append({

                "image_id": frame_id,

                "idx": 0,

                "keypoints":
                    halpe26.reshape(-1).tolist()

            })

        if len(pose_sequence) != self.clip_length:

            return None, None

        return pose_sequence, frames

    ##################################################
    # MotionBERT 2D → 3D
    ##################################################

    def predict_motionbert(self):

        zero_motion = np.zeros(
            20 * 17 * 3,
            dtype=np.float32
        )

        pose_sequence, frames = (
            self.extract_motionbert_pose()
        )

        if pose_sequence is None:

            return zero_motion

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = os.path.join(
                temp_dir,
                "human_clip.mp4"
            )

            json_path = os.path.join(
                temp_dir,
                "alphapose-results.json"
            )

            output_dir = os.path.join(
                temp_dir,
                "motionbert_output"
            )

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            height, width = frames[0].shape[:2]

            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),
                10,
                (width, height)
            )

            for frame in frames:

                writer.write(frame)

            writer.release()

            with open(
                json_path,
                "w"
            ) as f:

                json.dump(
                    pose_sequence,
                    f
                )

            command = [

                "python",

                self.motionbert_infer,

                "--config",
                self.motionbert_config,

                "--evaluate",
                self.motionbert_checkpoint,

                "--json_path",
                json_path,

                "--vid_path",
                video_path,

                "--out_path",
                output_dir,

                "--clip_len",
                "20"

            ]

            try:

                subprocess.run(
                    command,
                    cwd=self.motionbert_root,
                    check=True
                )

            except Exception as e:

                rospy.logwarn(
                    f"MotionBERT inference failed: {e}"
                )

                return zero_motion

            output_file = os.path.join(
                output_dir,
                "X3D.npy"
            )

            if not os.path.exists(
                output_file
            ):

                rospy.logwarn(
                    "MotionBERT X3D.npy not found."
                )

                return zero_motion

            pose_3d = np.load(
                output_file
            )

            pose_3d = np.asarray(
                pose_3d,
                dtype=np.float32
            )

            if pose_3d.ndim == 4:

                pose_3d = pose_3d[0]

            if pose_3d.ndim != 3:

                return zero_motion

            if pose_3d.shape[0] >= 20:

                pose_3d = pose_3d[-20:]

            else:

                padding = np.repeat(
                    pose_3d[-1:],
                    20 - pose_3d.shape[0],
                    axis=0
                )

                pose_3d = np.concatenate(
                    [
                        pose_3d,
                        padding
                    ],
                    axis=0
                )

            pose_3d = pose_3d[
                :20,
                :17,
                :3
            ]

            if pose_3d.shape != (
                20,
                17,
                3
            ):

                return zero_motion

            return pose_3d.reshape(
                -1
            ).astype(
                np.float32
            )

    ##################################################
    # Compare Database
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

            similarity = (
                torch.nn.functional
                .cosine_similarity(
                    current_embedding.unsqueeze(0),
                    reference_embedding.unsqueeze(0)
                )
            )

            similarities.append(
                similarity.item()
            )

        if len(similarities) == 0:

            return 0.0

        return max(similarities)

    ##################################################
    # Social Similarity
    ##################################################

    def compute_social_similarity(self):

        current_video = (
            self.capture_video_clip()
        )

        if current_video is None:

            return 0.0, 0.0, False

        human_detected = (
            self.detect_humans(
                current_video[-1]
            )
        )

        if not human_detected:

            return 0.0, 0.0, False

        current_embedding = (
            self.compute_video_embedding(
                current_video
            )
        )

        positive_similarity = (
            self.compare_database(
                current_embedding,
                self.positive_database
            )
        )

        negative_similarity = (
            self.compare_database(
                current_embedding,
                self.negative_database
            )
        )

        return (
            positive_similarity,
            negative_similarity,
            True
        )

    ##################################################
    # Build Graph State
    ##################################################

    def build_state_graph(
            self,
            laser_scan
    ):

        ##################################################
        # Robot Node
        ##################################################

        robot_node = {

            "position": [
                self.robot_pose.position.x,
                self.robot_pose.position.y,
                self.robot_pose.position.z
            ],

            "orientation": self.yaw,

            "linear_velocity":
                self.linear_velocity,

            "angular_velocity":
                self.angular_velocity

        }

        ##################################################
        # Goal Node
        ##################################################

        target_node = {

            "position": [
                self.goal_pose.position.x,
                self.goal_pose.position.y,
                self.goal_pose.position.z
            ]

        }

        ##################################################
        # Obstacle Nodes
        ##################################################

        obstacle_nodes = []

        angle = laser_scan.angle_min

        min_obstacle_distance = float(
            "inf"
        )

        for distance in laser_scan.ranges:

            if (
                np.isinf(distance) or
                np.isnan(distance)
            ):

                angle += (
                    laser_scan.angle_increment
                )

                continue

            obstacle_nodes.append({

                "position": [
                    self.robot_pose.position.x +
                    distance *
                    math.cos(
                        self.yaw + angle
                    ),

                    self.robot_pose.position.y +
                    distance *
                    math.sin(
                        self.yaw + angle
                    ),

                    self.robot_pose.position.z
                ],

                "distance": distance,

                "orientation": (
                    self.yaw + angle
                )

            })

            min_obstacle_distance = min(
                min_obstacle_distance,
                distance
            )

            angle += (
                laser_scan.angle_increment
            )

        ##################################################
        # Human Nodes
        ##################################################

        human_nodes = []

        human_detected = False

        if self.current_frame is not None:

            results = self.human_pose_detector(
                self.current_frame,
                verbose=False
            )

            for result in results:

                if result.keypoints is None:
                    continue

                if result.keypoints.xy is None:
                    continue

                keypoints = (
                    result.keypoints.xy
                    .cpu()
                    .numpy()
                )

                if len(keypoints) == 0:
                    continue

                if result.boxes is not None:

                    confidences = (
                        result.boxes.conf
                        .cpu()
                        .numpy()
                    )

                    person_id = int(
                        np.argmax(
                            confidences
                        )
                    )

                else:

                    person_id = 0

                person = keypoints[
                    person_id
                ]

                human_detected = True

                cx = float(
                    np.mean(
                        person[:, 0]
                    )
                )

                cy = float(
                    np.mean(
                        person[:, 1]
                    )
                )

                imu_features = (
                    self.predict_human_imu()
                )

                motionbert_features = (
                    self.predict_motionbert()
                )

                human_features = np.concatenate(
                    [

                        np.array(
                            [
                                cx,
                                cy,
                                1.0
                            ],
                            dtype=np.float32
                        ),

                        imu_features,

                        motionbert_features

                    ]
                ).astype(
                    np.float32
                )

                human_nodes.append({

                    "position": [
                        cx,
                        cy,
                        1.0
                    ],

                    "features":
                        human_features.tolist(),

                    "imu":
                        imu_features.tolist(),

                    "motionbert":
                        motionbert_features.tolist()

                })

        ##################################################
        # Graph Builder
        ##################################################

        graph_state = build_graph(

            robot_node,

            target_node,

            obstacle_nodes,

            human_nodes

        )

        ##################################################
        # Graph Builder compatibility
        ##################################################

        if isinstance(
            graph_state,
            dict
        ):

            graph_state[
                "human_detected"
            ] = human_detected

        return (
            graph_state,
            min_obstacle_distance
        )

    ##################################################
    # Get State
    ##################################################

    def getState(self, scan):

        done = False
        arrive = False

        (
            graph_state,
            min_obstacle_distance
        ) = self.build_state_graph(
            scan
        )

        current_distance = math.hypot(

            self.goal_pose.position.x -
            self.robot_pose.position.x,

            self.goal_pose.position.y -
            self.robot_pose.position.y

        )

        if min_obstacle_distance < 0.20:

            done = True

        if current_distance < self.goal_threshold:

            arrive = True

        timeout = False

        if self.current_step >= self.max_steps:

            timeout = True
            done = True

        return (

            graph_state,

            current_distance,

            min_obstacle_distance,

            timeout,

            done,

            arrive

        )

    ##################################################
    # Reward
    ##################################################

    def compute_reward(
            self,
            current_distance,
            target_reached,
            timeout,
            current_v,
            current_w,
            positive_similarity,
            negative_similarity,
            human_detected
    ):

        r_goal = (
            self.Rg
            if target_reached
            else 0.0
        )

        r_progress = (
            self.lambda_p *
            (
                self.previous_distance -
                current_distance
            )
        )

        r_failure = (
            -self.Rf
            if timeout
            else 0.0
        )

        r_damage = (
            -self.lambda_d *
            (
                abs(
                    current_v -
                    self.previous_linear_velocity
                )
                +
                abs(
                    current_w -
                    self.previous_angular_velocity
                )
            )
        )

        r_social = (
            self.lambda_s *
            (
                positive_similarity -
                negative_similarity
            )
        )

        delta = (
            1
            if human_detected
            else 0
        )

        reward = (

            r_goal +
            r_progress +
            r_failure +
            r_damage +
            delta * r_social

        )

        self.previous_distance = (
            current_distance
        )

        self.previous_linear_velocity = (
            current_v
        )

        self.previous_angular_velocity = (
            current_w
        )

        return reward

    ##################################################
    # Generate Goal
    ##################################################

    def generate_goal(self):

        try:

            self.delete_goal(
                "Target"
            )

        except:

            pass

        if len(
            self.training_targets
        ) == 0:

            for block_x in [
                -8, -4, 0, 4
            ]:

                for block_y in [
                    -8, -4, 0, 4
                ]:

                    self.training_targets.extend([

                        (
                            block_x + 1.0,
                            block_y + 1.0
                        ),

                        (
                            block_x + 3.0,
                            block_y + 1.0
                        ),

                        (
                            block_x + 1.0,
                            block_y + 3.0
                        ),

                        (
                            block_x + 3.0,
                            block_y + 3.0
                        )

                    ])

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

                        target[0] -
                        self.last_target[0],

                        target[1] -
                        self.last_target[1]

                    )

                    if d >= MIN_DISTANCE:

                        candidates.append(
                            target
                        )

                if len(candidates) == 0:

                    candidates = (
                        self.training_targets
                    )

                self.last_target = random.choice(
                    candidates
                )

            x = self.last_target[0]
            y = self.last_target[1]

        else:

            while True:

                x = random.uniform(
                    -8.0,
                    8.0
                )

                y = random.uniform(
                    -8.0,
                    8.0
                )

                d = math.hypot(

                    x -
                    self.robot_pose.position.x,

                    y -
                    self.robot_pose.position.y

                )

                if d >= 3.0:

                    break

        self.goal_pose.position.x = x
        self.goal_pose.position.y = y
        self.goal_pose.position.z = 0.0

        with open(
            goal_model_dir,
            "r"
        ) as f:

            goal_xml = f.read()

        self.spawn_goal(

            "Target",

            goal_xml,

            "",

            self.goal_pose,

            "world"

        )

    ##################################################
    # Step
    ##################################################

    def step(self, action):

        self.linear_velocity = float(
            action[0]
        )

        self.angular_velocity = float(
            action[1]
        )

        cmd = Twist()

        cmd.linear.x = (
            self.linear_velocity
        )

        cmd.angular.z = (
            self.angular_velocity
        )

        self.cmd_pub.publish(
            cmd
        )

        rospy.sleep(0.10)

        self.current_step += 1

        scan = rospy.wait_for_message(
            "/scan",
            LaserScan
        )

        (
            graph_state,
            current_distance,
            min_obstacle_distance,
            timeout,
            done,
            arrive
        ) = self.getState(
            scan
        )

        (
            positive_similarity,
            negative_similarity,
            human_detected
        ) = self.compute_social_similarity()

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

        if done:

            stop = Twist()

            self.cmd_pub.publish(
                stop
            )

        info = {

            "goal_distance":
                current_distance,

            "closest_obstacle":
                min_obstacle_distance,

            "positive_similarity":
                positive_similarity,

            "negative_similarity":
                negative_similarity,

            "human_detected":
                human_detected,

            "target_reached":
                arrive,

            "timeout":
                timeout,

            "step":
                self.current_step

        }

        return (

            graph_state,

            reward,

            done,

            arrive,

            info

        )

    ##################################################
    # Reset
    ##################################################

    def reset(self):

        stop = Twist()

        self.cmd_pub.publish(
            stop
        )

        rospy.sleep(0.5)

        self.reset_world()

        self.current_step = 0

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.previous_linear_velocity = 0.0
        self.previous_angular_velocity = 0.0

        self.previous_distance = 0.0

        self.video_buffer.clear()

        self.generate_goal()

        while self.current_frame is None:

            rospy.sleep(0.05)

        scan = rospy.wait_for_message(
            "/scan",
            LaserScan
        )

        (
            graph_state,
            current_distance,
            _,
            _,
            _,
            _
        ) = self.getState(
            scan
        )

        self.previous_distance = (
            current_distance
        )

        return graph_state
