#!/usr/bin/env python3
"""
train_timesformer_imu_refined.py

Train a 6D IMU regression head on top of a frozen pretrained TimeSformer.

Research pipeline
-----------------
Independent human-motion experiment:
    synchronized human video + on-body IMU measurements
                |
                v
       2-second / 20-frame scene
                |
                v
      uniform temporal sampling
                |
                v
          16 frames
                |
                v
    pretrained TimeSformer (FROZEN)
                |
                v
        video embedding
                |
                v
       regression head
                |
                v
       [ax, ay, az, wx, wy, wz]

Dataset structure
-----------------
The recorded video and IMU files are stored separately.

Example:

    human_motion_dataset/
    |
    +-- videos/
    |     +-- participant01_run01.mp4
    |     +-- participant01_run02.mp4
    |     +-- participant02_run01.mp4
    |
    +-- imu/
          +-- participant01_run01.csv
          +-- participant01_run02.csv
          +-- participant02_run01.csv

Video and IMU filenames must have the same stem.

Each IMU CSV must contain:
    timestamp, ax, ay, az, wx, wy, wz

The timestamp is assumed to be in seconds relative to the beginning
of the corresponding video recording.

IMPORTANT
---------
SCENE_FRAMES = 20 represents the complete observation window.

The pretrained TimeSformer checkpoint determines its native temporal
input size. For the selected checkpoint, the script reads that value
automatically and uniformly samples the 20-frame scene to that number
of frames. Thus the complete 20-frame scene is NOT discarded; it is
the observation window from which the pretrained model input is sampled.

Only the regression head is trained. The pretrained TimeSformer is frozen.
"""


import os
import glob
import random

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

from transformers import (
    AutoImageProcessor,
    TimesformerModel,
)


# ============================================================
# 1. ACTUAL TRAINING DATA PATHS
# ============================================================

# ------------------------------------------------------------
# PATH TO THE RECORDED HUMAN VIDEOS
# ------------------------------------------------------------
#
# Replace this with the real path on your machine.
#
VIDEO_DIR = r"D:\LISSI\human_motion_dataset\videos"


# ------------------------------------------------------------
# PATH TO THE RECORDED IMU MEASUREMENTS
# ------------------------------------------------------------
#
# Replace this with the real path on your machine.
#
IMU_DIR = r"D:\LISSI\human_motion_dataset\imu"


# ============================================================
# 2. PRETRAINED TIMESFORMER
# ============================================================

MODEL_ID = "facebook/timesformer-base-finetuned-k400"


# ============================================================
# 3. SCENE CONFIGURATION
# ============================================================

# The agent/human-motion parser observes a 2-second scene.
#
# According to our experimental setup:
#       2 seconds -> 20 recorded frames
#
SCENE_FRAMES = 20


# ============================================================
# 4. IMU TARGET
# ============================================================

# The IMU provides:
#
# Linear acceleration:
#       ax, ay, az
#
# Angular velocity:
#       wx, wy, wz
#
# Therefore the regression target has 6 elements.

TARGET_COLUMNS = [
    "ax",
    "ay",
    "az",
    "wx",
    "wy",
    "wz",
]

TARGET_DIM = 6


# ============================================================
# 5. TRAINING HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 4

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.20

NUM_WORKERS = 0

SEED = 42

CHECKPOINT_PATH = "timesformer_imu_regressor.pt"


# ============================================================
# 6. REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 7. VIDEO + IMU DATASET
# ============================================================

class VideoIMUDataset(Dataset):

    def __init__(
        self,
        samples,
        processor,
        scene_frames=SCENE_FRAMES,
        model_frames=None,
    ):

        self.samples = samples

        self.processor = processor

        self.scene_frames = scene_frames

        self.model_frames = model_frames

    def __len__(self):

        return len(self.samples)

    @staticmethod
    def read_scene(
        video_path,
        start_frame,
        num_frames,
    ):

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video:\n"
                f"{video_path}"
            )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        frames = []

        for _ in range(num_frames):

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

        cap.release()

        if len(frames) != num_frames:

            raise RuntimeError(
                f"Could not read "
                f"{num_frames} frames from:\n"
                f"{video_path}\n"
                f"Starting frame: {start_frame}\n"
                f"Only {len(frames)} frames were read."
            )

        return frames

    @staticmethod
    def sample_frames(
        frames,
        target_count,
    ):

        if len(frames) == target_count:

            return frames

        indices = np.linspace(
            0,
            len(frames) - 1,
            target_count,
            dtype=np.int64,
        )

        return [
            frames[i]
            for i in indices
        ]

    def __getitem__(self, index):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Read the complete 20-frame scene
        # ----------------------------------------------------

        scene = self.read_scene(
            sample["video_path"],
            sample["start_frame"],
            self.scene_frames,
        )

        # ----------------------------------------------------
        # Uniformly sample the number of frames expected by
        # the pretrained TimeSformer.
        #
        # Example:
        #       20-frame scene -> 16 model frames
        # ----------------------------------------------------

        model_input_frames = (
            self.sample_frames(
                scene,
                self.model_frames,
            )
        )

        # ----------------------------------------------------
        # TimeSformer preprocessing
        # ----------------------------------------------------

        inputs = self.processor(
            model_input_frames,
            return_tensors="pt",
        )

        pixel_values = (
            inputs["pixel_values"]
            .squeeze(0)
        )

        # ----------------------------------------------------
        # Ground-truth 6D IMU target
        # ----------------------------------------------------

        target = torch.tensor(
            sample["target"],
            dtype=torch.float32,
        )

        return (
            pixel_values,
            target,
        )


# ============================================================
# 8. BUILD SYNCHRONIZED VIDEO + IMU SAMPLES
# ============================================================

def build_samples():

    # --------------------------------------------------------
    # Find recorded videos
    # --------------------------------------------------------

    video_paths = sorted(
        glob.glob(
            os.path.join(
                VIDEO_DIR,
                "*.mp4",
            )
        )
    )

    # Also support AVI/MOV if needed.
    video_paths += sorted(
        glob.glob(
            os.path.join(
                VIDEO_DIR,
                "*.avi",
            )
        )
    )

    video_paths += sorted(
        glob.glob(
            os.path.join(
                VIDEO_DIR,
                "*.mov",
            )
        )
    )

    if not video_paths:

        raise FileNotFoundError(
            "No recorded videos were found.\n\n"
            f"VIDEO_DIR:\n{VIDEO_DIR}"
        )

    print(
        f"\nFound {len(video_paths)} recorded videos."
    )

    samples = []

    # --------------------------------------------------------
    # Process every recorded video
    # --------------------------------------------------------

    for video_path in video_paths:

        video_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        # ----------------------------------------------------
        # Locate corresponding IMU recording
        # ----------------------------------------------------

        imu_path = os.path.join(
            IMU_DIR,
            video_name + ".csv"
        )

        if not os.path.exists(
            imu_path
        ):

            print(
                "\nWARNING: no corresponding IMU file:"
            )

            print(
                f"Video: {video_path}"
            )

            print(
                f"Expected IMU: {imu_path}"
            )

            continue

        print(
            "\n------------------------------------------"
        )

        print(
            f"Video:\n{video_path}"
        )

        print(
            f"IMU:\n{imu_path}"
        )

        # ----------------------------------------------------
        # Load IMU data
        # ----------------------------------------------------

        imu_df = pd.read_csv(
            imu_path
        )

        required_columns = (
            {"timestamp"}
            | set(TARGET_COLUMNS)
        )

        missing_columns = (
            required_columns
            - set(imu_df.columns)
        )

        if missing_columns:

            raise ValueError(
                f"\nIMU file is missing columns:\n"
                f"{sorted(missing_columns)}\n\n"
                f"File:\n{imu_path}"
            )

        # ----------------------------------------------------
        # Ensure timestamps are numeric
        # ----------------------------------------------------

        imu_df["timestamp"] = pd.to_numeric(
            imu_df["timestamp"],
            errors="coerce",
        )

        imu_df = imu_df.dropna(
            subset=["timestamp"]
            + TARGET_COLUMNS
        )

        # ----------------------------------------------------
        # Video properties
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            video_path
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        cap.release()

        if fps <= 0:

            raise ValueError(
                f"Invalid FPS for:\n"
                f"{video_path}"
            )

        # ----------------------------------------------------
        # Create 20-frame observation windows
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We keep the full 20-frame scene.
        #
        # The pretrained TimeSformer later samples its
        # required temporal length from these 20 frames.
        #
        # ----------------------------------------------------

        for start_frame in range(
            0,
            total_frames
            - SCENE_FRAMES
            + 1,
            SCENE_FRAMES,
        ):

            # ------------------------------------------------
            # Convert frame positions into time
            # ------------------------------------------------

            start_time = (
                start_frame / fps
            )

            end_time = (
                (
                    start_frame
                    + SCENE_FRAMES
                )
                / fps
            )

            # ------------------------------------------------
            # Select IMU measurements synchronized with
            # this exact video observation window.
            # ------------------------------------------------

            imu_mask = (
                (imu_df["timestamp"]
                 >= start_time)
                &
                (imu_df["timestamp"]
                 < end_time)
            )

            scene_imu = imu_df.loc[
                imu_mask,
                TARGET_COLUMNS,
            ]

            # No IMU measurement for this scene.
            if len(scene_imu) == 0:

                continue

            # ------------------------------------------------
            # Ground-truth target
            # ------------------------------------------------
            #
            # One 6D IMU target is associated with this
            # 2-second video observation.
            #
            # The current implementation uses the mean
            # measured motion over the scene.
            #
            # [ax, ay, az, wx, wy, wz]
            # ------------------------------------------------

            target = (
                scene_imu
                .mean()
                .values
                .astype(np.float32)
            )

            samples.append(
                {
                    "video_path":
                        video_path,

                    "imu_path":
                        imu_path,

                    "start_frame":
                        start_frame,

                    "target":
                        target,
                }
            )

    # --------------------------------------------------------
    # Dataset check
    # --------------------------------------------------------

    if not samples:

        raise RuntimeError(
            "\nNo synchronized video/IMU samples "
            "were created.\n\n"
            "Check:\n"
            "1. VIDEO_DIR\n"
            "2. IMU_DIR\n"
            "3. matching video/IMU filenames\n"
            "4. IMU timestamp units\n"
            "5. required IMU columns"
        )

    print(
        "\n=========================================="
    )

    print(
        f"Total synchronized samples: "
        f"{len(samples)}"
    )

    print(
        "=========================================="
    )

    return samples


# ============================================================
# 9. TIMESFORMER + REGRESSION HEAD
# ============================================================

class TimeSformerIMURegressor(
    nn.Module
):

    def __init__(
        self,
        model_id=MODEL_ID,
        output_dim=TARGET_DIM,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Load pretrained TimeSformer
        # ----------------------------------------------------

        self.timesformer = (
            TimesformerModel
            .from_pretrained(
                model_id
            )
        )

        # ----------------------------------------------------
        # FREEZE TimeSformer
        # ----------------------------------------------------

        for parameter in (
            self.timesformer.parameters()
        ):

            parameter.requires_grad = False

        # ----------------------------------------------------
        # TimeSformer embedding dimension
        # ----------------------------------------------------

        hidden_size = (
            self.timesformer
            .config
            .hidden_size
        )

        # ----------------------------------------------------
        # TRAINABLE REGRESSION HEAD
        # ----------------------------------------------------

        self.regression_head = nn.Sequential(

            nn.Linear(
                hidden_size,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                output_dim,
            ),
        )

    def forward(
        self,
        pixel_values,
    ):

        # ----------------------------------------------------
        # Frozen TimeSformer
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = (
                self.timesformer(
                    pixel_values=pixel_values
                )
            )

        # ----------------------------------------------------
        # Video embedding
        # ----------------------------------------------------

        embedding = (
            outputs
            .last_hidden_state
            .mean(dim=1)
        )

        # ----------------------------------------------------
        # 6D IMU prediction
        # ----------------------------------------------------

        prediction = (
            self.regression_head(
                embedding
            )
        )

        return prediction


# ============================================================
# 10. TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    for pixel_values, targets in loader:

        pixel_values = (
            pixel_values.to(device)
        )

        targets = (
            targets.to(device)
        )

        optimizer.zero_grad()

        predictions = (
            model(pixel_values)
        )

        loss = criterion(
            predictions,
            targets,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * pixel_values.size(0)
        )

    return (
        total_loss
        / len(loader.dataset)
    )


# ============================================================
# 11. VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    for pixel_values, targets in loader:

        pixel_values = (
            pixel_values.to(device)
        )

        targets = (
            targets.to(device)
        )

        predictions = (
            model(pixel_values)
        )

        loss = criterion(
            predictions,
            targets,
        )

        total_loss += (
            loss.item()
            * pixel_values.size(0)
        )

    return (
        total_loss
        / len(loader.dataset)
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    set_seed()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n=========================================="
    )

    print(
        "TimeSformer + IMU Training"
    )

    print(
        "=========================================="
    )

    print(
        f"\nVIDEO_DIR:\n{VIDEO_DIR}"
    )

    print(
        f"\nIMU_DIR:\n{IMU_DIR}"
    )

    print(
        f"\nScene observation: "
        f"{SCENE_FRAMES} frames"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Pretrained TimeSformer processor
    # --------------------------------------------------------

    processor = (
        AutoImageProcessor
        .from_pretrained(
            MODEL_ID
        )
    )

    # --------------------------------------------------------
    # Determine native frame count of the
    # pretrained TimeSformer
    # --------------------------------------------------------

    pretrained_model = (
        TimesformerModel
        .from_pretrained(
            MODEL_ID
        )
    )

    model_frames = (
        pretrained_model
        .config
        .num_frames
    )

    del pretrained_model

    print(
        "\nPretrained TimeSformer "
        f"native frame count: {model_frames}"
    )

    print(
        f"Full observation window: "
        f"{SCENE_FRAMES} frames"
    )

    print(
        f"Model input: "
        f"{model_frames} uniformly sampled frames"
    )

    # --------------------------------------------------------
    # Build synchronized dataset
    # --------------------------------------------------------

    samples = build_samples()

    dataset = (
        VideoIMUDataset(
            samples=samples,
            processor=processor,
            scene_frames=SCENE_FRAMES,
            model_frames=model_frames,
        )
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    val_size = max(
        1,
        int(
            len(dataset)
            * VAL_RATIO
        )
    )

    train_size = (
        len(dataset)
        - val_size
    )

    if train_size < 1:

        raise ValueError(
            "Dataset is too small "
            "for a train/validation split."
        )

    train_dataset, val_dataset = (
        random_split(
            dataset,
            [
                train_size,
                val_size,
            ],
            generator=(
                torch.Generator()
                .manual_seed(SEED)
            ),
        )
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        TimeSformerIMURegressor(
            model_id=MODEL_ID,
            output_dim=TARGET_DIM,
        )
        .to(device)
    )

    # --------------------------------------------------------
    # ONLY regression-head parameters are trainable
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Regression loss
    # --------------------------------------------------------

    criterion = nn.SmoothL1Loss()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )
        )

        val_loss = (
            evaluate(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS} | "
            f"Train Loss: "
            f"{train_loss:.6f} | "
            f"Val Loss: "
            f"{val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "model_id":
                        MODEL_ID,

                    "scene_frames":
                        SCENE_FRAMES,

                    "model_frames":
                        model_frames,

                    "target_dim":
                        TARGET_DIM,

                    "target_names":
                        TARGET_COLUMNS,

                    "best_val_loss":
                        best_val_loss,
                },
                CHECKPOINT_PATH,
            )

            print(
                f"Saved best checkpoint -> "
                f"{CHECKPOINT_PATH}"
            )

    print(
        "\n=========================================="
    )

    print(
        "Training completed."
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        "=========================================="
    )


# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":

    main()
