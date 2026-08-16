#!/usr/bin/env python3
"""
train_timesformer_imu.py

Train an IMU regression head on top of a frozen pretrained TimeSformer.

Pipeline:
    20-frame / 2-second human-motion scene
        -> TimeSformer
        -> video embedding
        -> regression head
        -> [ax, ay, az, wx, wy, wz]

Dataset manifest format (CSV):
    video_path,start_frame,ax,ay,az,wx,wy,wz

Each CSV row represents ONE training sample:
- video_path: path to the source video
- start_frame: first frame of the 20-frame scene window (0-based)
- ax..az: synchronized ground-truth linear acceleration
- wx..wz: synchronized ground-truth angular velocity

Important:
The 20 frames define the 2-second scene window. Standard pretrained
TimeSformer checkpoints commonly use a native temporal length of 8 frames.
Therefore, by default, the script uniformly samples the 20-frame scene down
to the checkpoint's native number of frames before inference. This preserves
the pretrained temporal architecture instead of randomly initializing temporal
weights. If a 20-frame pretrained checkpoint is selected later, MODEL_FRAMES
can be set accordingly.
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoImageProcessor, TimesformerModel


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "facebook/timesformer-base-finetuned-k400"

SCENE_FRAMES = 20          # 2-second scene window
TARGET_DIM = 6

BATCH_SIZE = 100
EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.30
NUM_WORKERS = 0
SEED = 42

CHECKPOINT_PATH = "timesformer_imu_regressor.pt"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class VideoIMUDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        processor,
        scene_frames=SCENE_FRAMES,
        model_frames=None,
    ):
        self.processor = processor
        self.scene_frames = scene_frames
        self.model_frames = model_frames
        self.samples = []

        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            """ video_path → where the video file is located
                start_frame → which 20-frame scene to use
                ax, ay, az → linear acceleration
                wx, wy, wz → angular velocity """" 

            required = {
                "video_path",
                "start_frame",
                "ax",
                "ay",
                "az",
                "wx",
                "wy",
                "wz",
            }

            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Manifest is missing columns: {sorted(missing)}"
                )

            for row in reader:
                self.samples.append(row)

        if not self.samples:
            raise ValueError("The manifest contains no samples.")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_scene(video_path, start_frame, num_frames):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []

        for _ in range(num_frames):
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) != num_frames:
            raise RuntimeError(
                f"Could not read {num_frames} frames from "
                f"{video_path} starting at frame {start_frame}. "
                f"Only {len(frames)} frames were read."
            )

        return frames

    @staticmethod
    def _sample_frames(frames, target_count):
        if len(frames) == target_count:
            return frames

        indices = np.linspace(
            0,
            len(frames) - 1,
            target_count,
            dtype=np.int64,
        )

        return [frames[i] for i in indices]

    def __getitem__(self, index):
        row = self.samples[index]

        video_path = row["video_path"]

        if not os.path.isabs(video_path):
            video_path = os.path.abspath(video_path)

        start_frame = int(row["start_frame"])

        scene = self._read_scene(
            video_path,
            start_frame,
            self.scene_frames,
        )

        # The full 20-frame scene is the observation window.
        # If the pretrained TimeSformer expects a different number
        # of frames, sample representative frames uniformly.
        model_input_frames = self._sample_frames(
            scene,
            self.model_frames,
        )

        inputs = self.processor(
            model_input_frames,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].squeeze(0)

        target = torch.tensor(
            [
                float(row["ax"]),
                float(row["ay"]),
                float(row["az"]),
                float(row["wx"]),
                float(row["wy"]),
                float(row["wz"]),
            ],
            dtype=torch.float32,
        )

        return pixel_values, target


# ============================================================
# Regression Model
# ============================================================

class TimeSformerIMURegressor(nn.Module):

    def __init__(self, model_id=MODEL_ID, output_dim=TARGET_DIM):
        super().__init__()

        self.timesformer = TimesformerModel.from_pretrained(model_id)

        # Freeze the pretrained TimeSformer.
        for parameter in self.timesformer.parameters():
            parameter.requires_grad = False

        hidden_size = self.timesformer.config.hidden_size

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, pixel_values):

        with torch.no_grad():
            outputs = self.timesformer(
                pixel_values=pixel_values
            )

        # Mean-pool the TimeSformer token representations.
        embedding = outputs.last_hidden_state.mean(dim=1)

        prediction = self.regression_head(embedding)

        return prediction


# ============================================================
# Training / Validation
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0.0

    for pixel_values, targets in loader:

        pixel_values = pixel_values.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions = model(pixel_values)

        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pixel_values.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0

    for pixel_values, targets in loader:

        pixel_values = pixel_values.to(device)
        targets = targets.to(device)

        predictions = model(pixel_values)

        loss = criterion(predictions, targets)

        total_loss += loss.item() * pixel_values.size(0)

    return total_loss / len(loader.dataset)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="CSV manifest containing video clips and 6D IMU targets.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=CHECKPOINT_PATH,
        help="Output checkpoint path.",
    )

    args = parser.parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Scene window: {SCENE_FRAMES} frames")

    # --------------------------------------------------------
    # Pretrained TimeSformer
    # --------------------------------------------------------

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID
    )

    pretrained_model = TimesformerModel.from_pretrained(
        MODEL_ID
    )

    model_frames = pretrained_model.config.num_frames

    del pretrained_model

    print(
        f"Pretrained TimeSformer native frame count: "
        f"{model_frames}"
    )

    if model_frames != SCENE_FRAMES:
        print(
            f"Using the full {SCENE_FRAMES}-frame scene window, "
            f"then uniformly sampling {model_frames} frames for "
            f"the pretrained TimeSformer."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = VideoIMUDataset(
        manifest_path=args.manifest,
        processor=processor,
        scene_frames=SCENE_FRAMES,
        model_frames=model_frames,
    )

    val_size = max(1, int(len(dataset) * VAL_RATIO))
    train_size = len(dataset) - val_size

    if train_size < 1:
        raise ValueError("Dataset is too small for a train/validation split.")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

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

    model = TimeSformerIMURegressor(
        model_id=MODEL_ID,
        output_dim=TARGET_DIM,
    ).to(device)

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

    criterion = nn.SmoothL1Loss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_id": MODEL_ID,
                    "scene_frames": SCENE_FRAMES,
                    "model_frames": model_frames,
                    "target_dim": TARGET_DIM,
                    "target_names": [
                        "ax",
                        "ay",
                        "az",
                        "wx",
                        "wy",
                        "wz",
                    ],
                    "best_val_loss": best_val_loss,
                },
                args.output,
            )

            print(
                f"Saved best checkpoint -> {args.output}"
            )

    print("\nTraining completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
