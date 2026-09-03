"""
ROBOT ORCHESTRATION — HISTORICAL GEMINI 1.5 FLASH (2024) + LoRA + EWC
======================================================================

This script implements the methodology requested by the project:

BLOCK 1
-------
Post-training on old tasks:
    old prompt + typical robot architecture models/actions
    -> Gemini 1.5 Flash + LoRA
    -> adapted LoRA parameters theta_1*

BLOCK 2
-------
Post-training on new orchestration tasks:
    prompt + Neural Process (NP) technical description
    -> chronological processing pipeline

Continual learning:
    L_total = L_task + L_EWC

    L_EWC = lambda / 2 * sum_i F_i (theta_i - theta_1_i*)^2

Only LoRA parameters are trainable.
The pretrained Gemini parameters remain frozen.

BLOCK 3
-------
Final adapted model:
    Gemini 1.5 Flash + final LoRA parameters
    -> runtime orchestration.

IMPORTANT
---------
This file is intentionally written as the 2024-style experimental
implementation requested by the project. The historical Gemini tuning API
supported managed tuning of Gemini 1.5 Flash and exposed the tuned model
resource, training data and tuning hyperparameters.

The EWC part requires the LoRA adapter parameter tensors and their gradients.
The code therefore isolates that interface in GeminiLoRAAdapterBackend.
If your old 2024 environment already has the adapter-weight export/retrieval
hook you used, connect that hook in the marked methods.

No external dataset is required: representative data are defined below,
following the same philosophy as the NP implementation.

The script NEVER treats frozen Gemini base weights as EWC parameters.
EWC is applied only to the trainable LoRA parameters.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# OPTIONAL HISTORICAL GEMINI SDK
# ============================================================

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY", "")

# Historical Gemini 1.5 Flash tuning resource used by the old API.
BASE_MODEL = "models/gemini-1.5-flash-001-tuning"

# These are the training settings used by the historical managed tuning API.
BLOCK1_EPOCHS = 5
BLOCK1_BATCH_SIZE = 4
BLOCK1_LEARNING_RATE = 1e-3

BLOCK2_EPOCHS = 5
BLOCK2_BATCH_SIZE = 4
BLOCK2_LEARNING_RATE = 1e-3

# EWC coefficient.
EWC_LAMBDA = 0.4

# Fisher estimation.
FISHER_MAX_EXAMPLES = None

# Local artifact directory.
ARTIFACT_DIR = Path("robot_orchestration_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK1_METADATA = ARTIFACT_DIR / "block1_metadata.json"
BLOCK2_METADATA = ARTIFACT_DIR / "block2_metadata.json"
BLOCK3_METADATA = ARTIFACT_DIR / "block3_metadata.json"

BLOCK1_LORA_WEIGHTS = ARTIFACT_DIR / "block1_lora_state.pt"
BLOCK1_FISHER = ARTIFACT_DIR / "block1_fisher.pt"
BLOCK1_REFERENCE = ARTIFACT_DIR / "block1_lora_reference.pt"

BLOCK2_LORA_WEIGHTS = ARTIFACT_DIR / "block2_lora_state.pt"


# ============================================================
# ROBOT GLOBAL ARCHITECTURE
# ============================================================

ROBOT_ARCHITECTURE = {
    "speech": {
        "speech_recognition": "Speech Recognition",
    },
    "vision": {
        "visual_feature_extraction": "ResNet",
        "face_recognition": "Dlib",
        "object_detection": "YOLO",
        "depth_estimation": "MiDaS",
        "object_classification": "SVM",
    },
    "knowledge_reasoning": {
        "knowledge_graph": "Knowledge Graph",
        "llm": "Gemini 1.5 Flash",
    },
    "robot_control": {
        "coordinate_transformation": "Coordinate Transformation",
        "inverse_kinematics": "Inverse Kinematics",
        "robot": "Pepper",
    },
    "speech_output": {
        "tts": "TTS",
    },
}


# ============================================================
# BLOCK 1 — OLD TASK DATA
# ============================================================
#
# Old data:
#     prompt + output
#
# The output identifies the normal technical models used in the
# robot environment.
#
# Purpose:
#     familiarize Gemini with the robot's usual duties and technical
#     vocabulary before introducing the orchestration problem.
# ============================================================

OLD_TASKS: List[Dict[str, Any]] = [
    {
        "id": "old_001",
        "prompt": "Recognize the person in front of the robot.",
        "output": {
            "models": ["ResNet", "Dlib"]
            
        },
    },
    {
        "id": "old_002",
        "prompt": "Detect the object visible in the camera image.",
        "output": {
            "models": ["YOLO"]
           
        },
    },
    {
        "id": "old_003",
        "prompt": "Understand the user's spoken request.",
        "output": {
            "models": [
                "Speech Recognition",
                "Gemini 1.5 Flash",
            ]
            ],
        },
    },
    {
        "id": "old_004",
        "prompt": "Estimate the depth of a detected object.",
        "output": {
            "models": ["MiDaS"]
            
        },
    },
    {
        "id": "old_005",
        "prompt": "Convert an image target coordinate into a robot coordinate.",
        "output": {
            "models": ["Coordinate Transformation"]
            ],
        },
    },
    {
        "id": "old_006",
        "prompt": "Compute the robot arm configuration required to reach a target.",
        "output": {
            "models": ["Inverse Kinematics"]
            ],
        },
    },
    {
        "id": "old_007",
        "prompt": "Use object knowledge to determine the appropriate action.",
        "output": {
            "models": [
                "Knowledge Graph",
                "Gemini 1.5 Flash",
            ]
        },
    },
    {
        "id": "old_008",
        "prompt": "Give the user a verbal answer.",
        "output": {
            "models": [
                "Gemini 1.5 Flash",
                "TTS",
            ]
        },
    },
]


# ============================================================
# BLOCK 2 — NEW ORCHESTRATION DATA
# ============================================================
#
# New data:
#
#     prompt
#       +
#     NP technical description
#       ->
#     chronological processing pipeline
#
# NP predicts WHAT models/actions are needed.
# Gemini learns HOW and IN WHAT ORDER they should be activated.
# ============================================================

NEW_ORCHESTRATION_TASKS: List[Dict[str, Any]] = [
    {
        "id": "new_001",
        "prompt": "Find the red cup and grasp it.",
        "np_description": {
            "models": [
                "Speech Recognition",
                "Gemini 1.5 Flash",
                "YOLO",
                "MiDaS",
                "Coordinate Transformation",
                "Inverse Kinematics",
            ],
            "actions": [
                "speech-to-text",
                "intent interpretation",
                "object detection",
                "depth estimation",
                "coordinate conversion",
                "grasping",
            ],
        },
        "output": {
            "pipeline": [
                {
                    "step": 1,
                    "module": "Speech Recognition",
                    "input": "user speech",
                    "output": "text request",
                    "purpose": "convert speech to text",
                },
                {
                    "step": 2,
                    "module": "Gemini 1.5 Flash",
                    "input": "text request",
                    "output": "target object and task intent",
                    "purpose": "interpret the request",
                },
                {
                    "step": 3,
                    "module": "YOLO",
                    "input": "camera image",
                    "output": "red cup detection and pixel coordinates",
                    "purpose": "locate the target object",
                },
                {
                    "step": 4,
                    "module": "MiDaS",
                    "input": "camera image and detected target",
                    "output": "target depth",
                    "purpose": "estimate target distance",
                },
                {
                    "step": 5,
                    "module": "Coordinate Transformation",
                    "input": "pixel coordinates and depth",
                    "output": "robot-frame target coordinates",
                    "purpose": "convert visual coordinates to robot coordinates",
                },
                {
                    "step": 6,
                    "module": "Inverse Kinematics",
                    "input": "robot-frame target coordinates",
                    "output": "arm joint configuration",
                    "purpose": "compute the grasp configuration",
                },
                {
                    "step": 7,
                    "module": "Pepper",
                    "input": "arm joint configuration",
                    "output": "object grasped",
                    "purpose": "execute the action",
                },
            ]
        },
    },
    {
        "id": "new_002",
        "prompt": "Recognize the person standing in front of the robot.",
        "np_description": {
            "models": [
                "ResNet",
                "Dlib",
            ],
            "actions": [
                "visual feature extraction",
                "face recognition",
            ],
        },
        "output": {
            "pipeline": [
                {
                    "step": 1,
                    "module": "ResNet",
                    "input": "camera image",
                    "output": "visual features",
                    "purpose": "extract visual representation",
                },
                {
                    "step": 2,
                    "module": "Dlib",
                    "input": "face region and visual features",
                    "output": "face identity",
                    "purpose": "recognize the person",
                },
            ]
        },
    },
    {
        "id": "new_003",
        "prompt": "Tell me what object is on the table and answer verbally.",
        "np_description": {
            "models": [
                "YOLO",
                "Gemini 1.5 Flash",
                "TTS",
            ],
            "actions": [
                "object detection",
                "object labeling",
                "language reasoning",
                "text-to-speech",
            ],
        },
        "output": {
            "pipeline": [
                {
                    "step": 1,
                    "module": "YOLO",
                    "input": "camera image",
                    "output": "detected object and label",
                    "purpose": "identify the object",
                },
                {
                    "step": 2,
                    "module": "Gemini 1.5 Flash",
                    "input": "detected object label",
                    "output": "natural-language answer",
                    "purpose": "reason and formulate the answer",
                },
                {
                    "step": 3,
                    "module": "TTS",
                    "input": "generated answer",
                    "output": "speech",
                    "purpose": "produce verbal response",
                },
                {
                    "step": 4,
                    "module": "Pepper",
                    "input": "speech signal",
                    "output": "verbal response to user",
                    "purpose": "communicate the answer",
                },
            ]
        },
    },
    {
        "id": "new_004",
        "prompt": "Pick up the object requested by the user.",
        "np_description": {
            "models": [
                "Speech Recognition",
                "Gemini 1.5 Flash",
                "Knowledge Graph",
                "YOLO",
                "MiDaS",
                "Coordinate Transformation",
                "Inverse Kinematics",
            ],
            "actions": [
                "speech-to-text",
                "intent interpretation",
                "knowledge retrieval",
                "object detection",
                "depth estimation",
                "coordinate conversion",
                "grasping",
            ],
        },
        "output": {
            "pipeline": [
                {
                    "step": 1,
                    "module": "Speech Recognition",
                    "input": "user speech",
                    "output": "text request",
                    "purpose": "convert speech to text",
                },
                {
                    "step": 2,
                    "module": "Gemini 1.5 Flash",
                    "input": "text request",
                    "output": "requested object and intent",
                    "purpose": "understand the task",
                },
                {
                    "step": 3,
                    "module": "Knowledge Graph",
                    "input": "requested object",
                    "output": "object/action class",
                    "purpose": "retrieve semantic knowledge",
                },
                {
                    "step": 4,
                    "module": "YOLO",
                    "input": "camera image",
                    "output": "object detection and pixel coordinates",
                    "purpose": "locate the requested object",
                },
                {
                    "step": 5,
                    "module": "MiDaS",
                    "input": "camera image and detected object",
                    "output": "object depth",
                    "purpose": "estimate 3D position information",
                },
                {
                    "step": 6,
                    "module": "Coordinate Transformation",
                    "input": "pixel coordinates and depth",
                    "output": "robot-frame coordinates",
                    "purpose": "obtain the target robot coordinates",
                },
                {
                    "step": 7,
                    "module": "Inverse Kinematics",
                    "input": "robot-frame coordinates",
                    "output": "arm joint configuration",
                    "purpose": "compute the grasp configuration",
                },
                {
                    "step": 8,
                    "module": "Pepper",
                    "input": "arm joint configuration",
                    "output": "object grasped",
                    "purpose": "execute the grasp",
                },
            ]
        },
    },
]


# ============================================================
# DATA FORMATTING
# ============================================================

def build_block1_examples(
    tasks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Create supervised examples for Block 1."""
    examples = []

    for task in tasks:
        text_input = (
            "You are operating inside a Pepper robot architecture.\n\n"
            f"Task:\n{task['prompt']}\n\n"
            "Identify the technical models/modules and actions normally "
            "used to solve this task.\n"
        )

        output = json.dumps(
            task["output"],
            ensure_ascii=False,
        )

        examples.append(
            {
                "text_input": text_input,
                "output": output,
            }
        )

    return examples


def build_block2_examples(
    tasks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Create supervised orchestration examples for Block 2."""
    examples = []

    for task in tasks:
        text_input = (
            "You are the reasoning and orchestration layer of a Pepper robot.\n\n"
            f"User task:\n{task['prompt']}\n\n"
            "Neural Process technical prediction:\n"
            f"{json.dumps(task['np_description'], ensure_ascii=False, indent=2)}\n\n"
            "Determine a feasible chronological processing pipeline.\n"
            "The Neural Process determines WHAT models/actions are needed.\n"
            "You determine HOW and IN WHAT ORDER they are activated.\n"
            "Respect dependencies between perception, reasoning, "
            "coordinate transformation and robot execution.\n"
            "Use only the technical modules provided by the architecture.\n"
        )

        output = json.dumps(
            task["output"],
            ensure_ascii=False,
        )

        examples.append(
            {
                "text_input": text_input,
                "output": output,
            }
        )

    return examples


# ============================================================
# GEMINI API — HISTORICAL 2024 INTERFACE
# ============================================================

class HistoricalGeminiAPI:
    """
    Wrapper around the old Gemini API tuning workflow.

    The API-side tuning job is responsible for producing the adapted
    Gemini model. The adapter parameter interface is deliberately separated
    from the API request itself because EWC needs actual LoRA tensors.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

        if genai is None:
            raise ImportError(
                "google-generativeai is not installed. "
                "This script targets the historical 2024 Gemini API SDK."
            )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )

        genai.configure(api_key=api_key)

    @staticmethod
    def create_tuned_model(
        display_name: str,
        examples: List[Dict[str, str]],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        source_model: str = BASE_MODEL,
    ) -> Any:
        """
        Historical 2024 Gemini API tuning call.

        The training data are supervised pairs:
            text_input -> output
        """
        operation = genai.create_tuned_model(
            source_model=source_model,
            training_data=examples,
            id=display_name,
            epoch_count=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        return operation

    @staticmethod
    def wait_for_operation(
        operation: Any,
        poll_seconds: int = 10,
    ) -> Any:
        """Wait for a historical tuning operation."""
        while True:
            try:
                if operation.done():
                    return operation
            except Exception:
                pass

            time.sleep(poll_seconds)

    @staticmethod
    def extract_tuned_model_name(operation: Any) -> str:
        """
        Extract the tuned model resource name from the completed operation.

        The historical SDK exposed this through the operation result/metadata
        depending on the SDK revision.
        """
        possible_objects = []

        for attr in ("result", "metadata"):
            try:
                value = getattr(operation, attr)
                if value is not None:
                    possible_objects.append(value)
            except Exception:
                pass

        try:
            result = operation.result()
            possible_objects.append(result)
        except Exception:
            pass

        for obj in possible_objects:
            if isinstance(obj, dict):
                for key in (
                    "tunedModel",
                    "tuned_model",
                    "name",
                ):
                    if obj.get(key):
                        return obj[key]

            for attr in (
                "tunedModel",
                "tuned_model",
                "name",
            ):
                try:
                    value = getattr(obj, attr)
                    if value:
                        return value
                except Exception:
                    pass

        raise RuntimeError(
            "Could not extract the tuned model resource name. "
            "Inspect the completed historical operation."
        )

    @staticmethod
    def generate(
        model_name: str,
        prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate from a tuned Gemini model."""
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
            },
        )

        return response.text


# ============================================================
# LORA ADAPTER BACKEND
# ============================================================

class GeminiLoRAAdapterBackend:
    """
    Interface representing the LoRA parameters trained on the user's data.

    IMPORTANT:
        base Gemini parameters are NEVER stored here.

    Only:
        A_lora, B_lora, scaling, etc.
    belong to this backend.

    The historical environment used by the project can implement the
    three adapter-specific methods:

        export_lora_state()
        import_lora_state()
        compute_log_likelihood_loss()

    The EWC implementation below then operates exclusively on these
    trainable LoRA parameters.

    The fallback implementation is a strict, explicit failure rather than
    silently using Gemini base weights or fabricated Fisher values.
    """

    def __init__(self):
        self._state: Optional[Dict[str, torch.Tensor]] = None

    def export_lora_state(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve the LoRA weights produced by the tuning stage.

        Replace this method with the adapter retrieval mechanism from the
        original 2024 environment.

        Expected output:
            {
                "layer_name.lora_A": tensor(...),
                "layer_name.lora_B": tensor(...),
                ...
            }

        Frozen Gemini base weights must NOT appear here.
        """
        if self._state is None:
            raise NotImplementedError(
                "Connect this method to the 2024 LoRA adapter-weight "
                "retrieval used by your original Gemini setup."
            )

        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self._state.items()
        }

    def import_lora_state(
        self,
        state: Dict[str, torch.Tensor],
    ) -> None:
        """
        Load only LoRA adapter parameters.
        """
        if not state:
            raise ValueError("Empty LoRA state.")

        self._state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in state.items()
        }

    def named_trainable_parameters(
        self,
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Return only trainable LoRA parameters.

        The connected implementation should return tensors with:
            requires_grad=True
        """
        if self._state is None:
            raise RuntimeError(
                "No LoRA state has been loaded."
            )

        for name, tensor in self._state.items():
            yield name, tensor

    def compute_log_likelihood_loss(
        self,
        text_input: str,
        target_output: str,
    ) -> torch.Tensor:
        """
        Adapter-specific loss hook.

        A true Fisher estimate requires gradients of the likelihood with
        respect to each trainable LoRA parameter.

        Connect this method to the old training runtime so that the returned
        tensor is differentiable with respect to the LoRA parameters.
        """
        raise NotImplementedError(
            "Connect the historical Gemini LoRA training runtime here. "
            "EWC requires differentiable likelihood loss with respect to "
            "the trainable LoRA parameters."
        )


# ============================================================
# EWC
# ============================================================

@dataclass
class EWCConfig:
    lambda_ewc: float
    formula: str
    trainable_parameter_scope: str
    frozen_parameter_scope: str


class ElasticWeightConsolidation:
    """
    Exact EWC implementation over LoRA parameters only.
    """

    def __init__(self, config: EWCConfig):
        self.config = config

        # theta_1*: Block-1 LoRA reference parameters.
        self.theta_old: Dict[str, torch.Tensor] = {}

        # F_i: Fisher information for Block-1 LoRA parameters.
        self.fisher: Dict[str, torch.Tensor] = {}

    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------

    def save_reference(
        self,
        path: Path,
    ) -> None:
        torch.save(self.theta_old, path)

    def save_fisher(
        self,
        path: Path,
    ) -> None:
        torch.save(self.fisher, path)

    def load_reference(
        self,
        path: Path,
    ) -> None:
        self.theta_old = torch.load(
            path,
            map_location="cpu",
        )

    def load_fisher(
        self,
        path: Path,
    ) -> None:
        self.fisher = torch.load(
            path,
            map_location="cpu",
        )

    # --------------------------------------------------------
    # BLOCK 1 REFERENCE
    # --------------------------------------------------------

    def capture_block1_lora_parameters(
        self,
        lora_state: Dict[str, torch.Tensor],
    ) -> None:
        """
        theta_old = theta_1*
        """
        if not lora_state:
            raise ValueError(
                "Block-1 LoRA state is empty."
            )

        self.theta_old = {
            name: tensor.detach().cpu().clone()
            for name, tensor in lora_state.items()
        }

    # --------------------------------------------------------
    # FISHER INFORMATION
    # --------------------------------------------------------

    def estimate_fisher(
        self,
        backend: GeminiLoRAAdapterBackend,
        old_examples: List[Dict[str, str]],
        max_examples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher information:

            F_i ~= 1/N sum_n
                   (d log p(y_n | x_n, theta) / d theta_i)^2

        Since:
            L = -log p(y|x)

        the squared gradient of L gives the same squared quantity.

        CRITICAL:
            Gradients are taken ONLY with respect to LoRA parameters.
        """
        if not self.theta_old:
            raise RuntimeError(
                "Block-1 LoRA reference parameters have not been captured."
            )

        examples = old_examples
        if max_examples is not None:
            examples = examples[:max_examples]

        fisher = {
            name: torch.zeros_like(value, dtype=torch.float32)
            for name, value in self.theta_old.items()
        }

        processed = 0

        for example in examples:
            loss = backend.compute_log_likelihood_loss(
                example["text_input"],
                example["output"],
            )

            # Clear existing gradients if the connected backend supports it.
            if hasattr(backend, "zero_grad"):
                backend.zero_grad()

            loss.backward()

            found_gradient = False

            for name, parameter in backend.named_trainable_parameters():
                if name not in fisher:
                    continue

                if parameter.grad is None:
                    continue

                fisher[name] += (
                    parameter.grad.detach().cpu().float() ** 2
                )

                found_gradient = True

            if not found_gradient:
                raise RuntimeError(
                    "No LoRA gradients were found during Fisher estimation."
                )

            processed += 1

        if processed == 0:
            raise RuntimeError(
                "No old-task examples were processed."
            )

        for name in fisher:
            fisher[name] /= float(processed)

        self.fisher = fisher

        return fisher

    # --------------------------------------------------------
    # EWC PENALTY
    # --------------------------------------------------------

    def penalty_from_state(
        self,
        current_lora_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute:

            L_EWC =
                lambda/2 *
                sum_i F_i (theta_i - theta_old_i)^2
        """
        if not self.theta_old:
            raise RuntimeError(
                "theta_old is not initialized."
            )

        if not self.fisher:
            raise RuntimeError(
                "Fisher information is not initialized."
            )

        penalty = torch.tensor(
            0.0,
            dtype=torch.float32,
        )

        for name, theta_current in current_lora_state.items():
            if name not in self.theta_old:
                continue

            if name not in self.fisher:
                continue

            theta_old = self.theta_old[name].to(
                theta_current.device
            )

            fisher = self.fisher[name].to(
                theta_current.device
            )

            penalty = penalty + (
                0.5
                * self.config.lambda_ewc
                * torch.sum(
                    fisher
                    * (theta_current - theta_old) ** 2
                )
            )

        return penalty

    # --------------------------------------------------------
    # TOTAL LOSS
    # --------------------------------------------------------

    def total_loss(
        self,
        task_loss: torch.Tensor,
        current_lora_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        L_total = L_task + L_EWC
        """
        ewc_loss = self.penalty_from_state(
            current_lora_state
        )

        total = task_loss + ewc_loss

        return total, ewc_loss


# ============================================================
# BLOCK 1
# ============================================================

def run_block1_api() -> str:
    """
    Block 1:
        old task data
        ->
        historical Gemini 1.5 Flash managed SFT
        ->
        tuned Gemini model
    """
    api = HistoricalGeminiAPI(API_KEY)

    examples = build_block1_examples(
        OLD_TASKS
    )

    print("\n" + "=" * 72)
    print("BLOCK 1 — POST-TRAINING ON OLD TASKS")
    print("=" * 72)

    operation = api.create_tuned_model(
        display_name="pepper_old_tasks_gemini_15_flash",
        examples=examples,
        epochs=BLOCK1_EPOCHS,
        batch_size=BLOCK1_BATCH_SIZE,
        learning_rate=BLOCK1_LEARNING_RATE,
    )

    operation = api.wait_for_operation(
        operation
    )

    tuned_model = api.extract_tuned_model_name(
        operation
    )

    metadata = {
        "block": 1,
        "base_model": BASE_MODEL,
        "method": "SFT",
        "adaptation": "LoRA",
        "frozen_parameters": "pretrained Gemini 1.5 Flash parameters",
        "trainable_parameters": "LoRA adapter parameters",
        "purpose": (
            "Familiarize the LLM with the robot environment, "
            "usual duties and technical architecture vocabulary."
        ),
        "examples": examples,
        "epochs": BLOCK1_EPOCHS,
        "batch_size": BLOCK1_BATCH_SIZE,
        "learning_rate": BLOCK1_LEARNING_RATE,
        "tuned_model": tuned_model,
    }

    BLOCK1_METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return tuned_model


# ============================================================
# BLOCK 1 — CAPTURE LoRA AND FISHER
# ============================================================

def prepare_block1_ewc(
    backend: GeminiLoRAAdapterBackend,
) -> ElasticWeightConsolidation:
    """
    After Block 1 tuning:

        1. retrieve Block-1 LoRA parameters theta_1*
        2. store them
        3. estimate Fisher F_i using old data
    """
    print("\n" + "=" * 72)
    print("BLOCK 1 — CAPTURE LoRA PARAMETERS + FISHER")
    print("=" * 72)

    lora_state = backend.export_lora_state()

    if not lora_state:
        raise RuntimeError(
            "No LoRA weights were returned by the adapter backend."
        )

    # Verify that this is adapter state, not a frozen Gemini state dump.
    if any(
        "lora" not in name.lower()
        for name in lora_state
    ):
        raise ValueError(
            "The retrieved state contains parameters that do not look like "
            "LoRA parameters. EWC must operate only on LoRA weights."
        )

    torch.save(
        lora_state,
        BLOCK1_LORA_WEIGHTS,
    )

    ewc = ElasticWeightConsolidation(
        EWCConfig(
            lambda_ewc=EWC_LAMBDA,
            formula=(
                "L_total = L_task + "
                "(lambda/2) * sum_i F_i * "
                "(theta_i - theta_1_i*)^2"
            ),
            trainable_parameter_scope=(
                "LoRA adapter parameters learned from old-task data"
            ),
            frozen_parameter_scope=(
                "Pretrained Gemini 1.5 Flash base parameters"
            ),
        )
    )

    ewc.capture_block1_lora_parameters(
        lora_state
    )

    ewc.save_reference(
        BLOCK1_REFERENCE
    )

    # Fisher is estimated from old-task likelihood gradients.
    ewc.estimate_fisher(
        backend=backend,
        old_examples=build_block1_examples(
            OLD_TASKS
        ),
        max_examples=FISHER_MAX_EXAMPLES,
    )

    ewc.save_fisher(
        BLOCK1_FISHER
    )

    print(
        f"Saved Block-1 LoRA parameters: "
        f"{BLOCK1_LORA_WEIGHTS}"
    )

    print(
        f"Saved Fisher information: "
        f"{BLOCK1_FISHER}"
    )

    return ewc


# ============================================================
# BLOCK 2 — NEW ORCHESTRATION POST-TRAINING
# ============================================================

def train_block2_with_ewc(
    backend: GeminiLoRAAdapterBackend,
    ewc: ElasticWeightConsolidation,
) -> Dict[str, torch.Tensor]:
    """
    True Block-2 continual-learning loop.

    Starting point:
        theta = theta_1*

    For each new orchestration example:

        L_task = SFT loss

        L_EWC =
            lambda/2 *
            sum_i F_i(theta_i - theta_1_i*)^2

        L_total = L_task + L_EWC

    Only LoRA parameters are updated.
    """
    print("\n" + "=" * 72)
    print("BLOCK 2 — POST-TRAINING ON NEW ORCHESTRATION TASKS + EWC")
    print("=" * 72)

    if not ewc.theta_old:
        raise RuntimeError(
            "Missing Block-1 LoRA reference parameters."
        )

    if not ewc.fisher:
        raise RuntimeError(
            "Missing Block-1 Fisher information."
        )

    # Start Block 2 from Block-1 LoRA parameters.
    backend.import_lora_state(
        ewc.theta_old
    )

    # The connected backend must expose actual trainable tensors.
    named_params = list(
        backend.named_trainable_parameters()
    )

    if not named_params:
        raise RuntimeError(
            "No trainable LoRA parameters are exposed."
        )

    # The optimizer operates ONLY on LoRA parameters.
    optimizer = torch.optim.AdamW(
        [
            parameter
            for _, parameter in named_params
            if parameter.requires_grad
        ],
        lr=BLOCK2_LEARNING_RATE,
    )

    examples = build_block2_examples(
        NEW_ORCHESTRATION_TASKS
    )

    for epoch in range(BLOCK2_EPOCHS):
        epoch_task_loss = 0.0
        epoch_ewc_loss = 0.0

        for example in examples:
            optimizer.zero_grad()

            # Connected Gemini-LoRA runtime computes SFT loss.
            task_loss = backend.compute_log_likelihood_loss(
                example["text_input"],
                example["output"],
            )

            current_state = {
                name: parameter
                for name, parameter
                in backend.named_trainable_parameters()
            }

            total_loss, ewc_loss = ewc.total_loss(
                task_loss,
                current_state,
            )

            total_loss.backward()

            optimizer.step()

            epoch_task_loss += float(
                task_loss.detach().cpu()
            )

            epoch_ewc_loss += float(
                ewc_loss.detach().cpu()
            )

        n = max(len(examples), 1)

        print(
            f"Epoch {epoch + 1}/{BLOCK2_EPOCHS} | "
            f"L_task={epoch_task_loss / n:.6f} | "
            f"L_EWC={epoch_ewc_loss / n:.6f}"
        )

    final_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter
        in backend.named_trainable_parameters()
    }

    torch.save(
        final_state,
        BLOCK2_LORA_WEIGHTS,
    )

    return final_state


# ============================================================
# BLOCK 2 — API MODEL REPRESENTATION
# ============================================================

def register_block2_metadata(
    block1_model: str,
) -> None:
    """
    Record the relationship between the historical Gemini API model and the
    externally applied LoRA/EWC training loop.
    """
    examples = build_block2_examples(
        NEW_ORCHESTRATION_TASKS
    )

    metadata = {
        "block": 2,
        "starting_model": block1_model,
        "input": (
            "usual task prompt + NP technical description"
        ),
        "target": (
            "chronological task orchestration / processing pipeline"
        ),
        "method": "SFT + EWC",
        "trainable_parameters": (
            "LoRA parameters only"
        ),
        "frozen_parameters": (
            "pretrained Gemini 1.5 Flash parameters"
        ),
        "ewc_lambda": EWC_LAMBDA,
        "ewc_formula": (
            "L_total = L_task + "
            "(lambda/2) * sum_i F_i * "
            "(theta_i - theta_1_i*)^2"
        ),
        "examples": examples,
        "block1_model": block1_model,
        "block1_lora_reference": str(
            BLOCK1_REFERENCE
        ),
        "block1_fisher": str(
            BLOCK1_FISHER
        ),
        "block2_lora_output": str(
            BLOCK2_LORA_WEIGHTS
        ),
    }

    BLOCK2_METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# BLOCK 3
# ============================================================

def build_final_model_record(
    block1_model: str,
) -> Dict[str, Any]:
    """
    Final adapted model representation.

    Conceptually:
        frozen Gemini base
              +
        Block-2 LoRA parameters
              =
        final orchestration model
    """
    record = {
        "block": 3,
        "base_model": BASE_MODEL,
        "historical_block1_model": block1_model,
        "final_trainable_parameters": (
            "Block-2 LoRA parameters after SFT + EWC"
        ),
        "frozen_parameters": (
            "pretrained Gemini 1.5 Flash parameters"
        ),
        "adapter_weights": str(
            BLOCK2_LORA_WEIGHTS
        ),
        "role": (
            "Reasoning and orchestration layer of the Pepper robot."
        ),
        "NP_role": (
            "Predict WHAT technical models/modules/actions are required."
        ),
        "LLM_role": (
            "Determine HOW and IN WHAT ORDER the predicted models/actions "
            "are activated to form a feasible chronological pipeline."
        ),
        "architecture": ROBOT_ARCHITECTURE,
    }

    BLOCK3_METADATA.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return record


# ============================================================
# RUNTIME ORCHESTRATION
# ============================================================

def build_runtime_prompt(
    user_task: str,
    np_prediction: Dict[str, Any],
) -> str:
    """
    Runtime input to the final adapted Gemini model.
    """
    return f"""
You are the orchestration and reasoning layer of a Pepper robot.

User task:
{user_task}

Neural Process technical prediction:
{json.dumps(np_prediction, ensure_ascii=False, indent=2)}

Available robot architecture:
{json.dumps(ROBOT_ARCHITECTURE, ensure_ascii=False, indent=2)}

Your role:
- The Neural Process determines WHAT technical models/modules/actions
  are required.
- You determine HOW and IN WHAT ORDER those components must execute.
- Resolve dependencies between perception, reasoning, coordinate
  transformation and robot control.
- Produce a feasible chronological processing pipeline.
- Do not invent models or modules outside the given architecture.

Return:
{{
  "pipeline": [
    {{
      "step": 1,
      "module": "...",
      "input": "...",
      "output": "...",
      "purpose": "..."
    }}
  ]
}}
""".strip()


# ============================================================
# COMPLETE EXPERIMENT
# ============================================================

def main() -> None:
    """
    Full methodology:

        BLOCK 1
            old data
              |
              v
        Gemini 1.5 Flash 2024 + LoRA SFT
              |
              v
        theta_1*
              |
              +----> Fisher F_i
              |
              v
        BLOCK 2
            new orchestration data
              |
              v
        SFT loss + EWC
              |
              v
        theta_2*
              |
              v
        BLOCK 3
            final Gemini + LoRA
              |
              v
        orchestration pipeline
    """

    print("=" * 72)
    print("ROBOT ORCHESTRATION — GEMINI 1.5 FLASH 2024")
    print("LoRA + SFT + EWC CONTINUAL LEARNING")
    print("=" * 72)

    print("\nMethodology:")
    print("  Block 1: old tasks -> SFT -> LoRA theta_1*")
    print("  Fisher:  old-task gradients -> F_i")
    print("  Block 2: new tasks -> SFT + EWC")
    print("  Block 3: final Gemini + updated LoRA")
    print("\nEWC:")
    print(
        "  L_total = L_task + "
        "(lambda/2) * sum_i F_i(theta_i - theta_1_i*)^2"
    )

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    block1_model = run_block1_api()

    print(
        f"\nBlock 1 tuned Gemini model: {block1_model}"
    )

    # --------------------------------------------------------
    # LOADING THE LoRA ADAPTER
    # --------------------------------------------------------
    #
    # This is the project-specific bridge to the old 2024 LoRA environment.
    #
    # The API tuning job creates the adapted model.
    # The adapter backend must retrieve the LoRA tensors learned from the
    # user's data. The frozen Gemini weights are NOT retrieved or modified.
    # --------------------------------------------------------

    backend = GeminiLoRAAdapterBackend()

    # This call intentionally requires the project's old LoRA retrieval hook.
    #
    # Once connected:
    #     lora_state = backend.export_lora_state()
    #
    # will return only theta_1*.

    try:
        ewc = prepare_block1_ewc(
            backend
        )
    except NotImplementedError as exc:
        print("\nLoRA adapter bridge required:")
        print(exc)
        print(
            "\nThe Gemini API tuning stage is complete, but the EWC stage "
            "needs the actual LoRA tensors from the old adapter runtime."
        )
        return

    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    final_lora_state = train_block2_with_ewc(
        backend,
        ewc,
    )

    register_block2_metadata(
        block1_model
    )

    # --------------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------------

    final_record = build_final_model_record(
        block1_model
    )

    print("\n" + "=" * 72)
    print("BLOCK 3 — FINAL ADAPTED MODEL")
    print("=" * 72)

    print(
        json.dumps(
            final_record,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nFinal LoRA parameters saved to:"
        f"\n{BLOCK2_LORA_WEIGHTS}"
    )


if __name__ == "__main__":
    main()
