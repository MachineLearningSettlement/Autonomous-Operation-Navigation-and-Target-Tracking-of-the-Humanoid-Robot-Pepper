# ============================================================
# ADVANCED NEURAL PROCESS META-LEARNING
#
# META-TRAINING:
#
#   meta_training_tasks
#        │
#        ├── perception
#        │      ├── task_1
#        │      ├── task_2
#        │      └── ...
#        │
#        ├── navigation
#        │      ├── task_1
#        │      ├── task_2
#        │      └── ...
#        │
#        ├── grasping
#        │      ├── task_1
#        │      ├── task_2
#        │      └── ...
#        │
#        ├── human_robot_interaction
#        │      ├── task_1
#        │      ├── task_2
#        │      └── ...
#        │ .
#        │ .
#        │ .
#        │ .
#        │
#        └── additional task natures ( we used more than 15 task natures)
#               ├── task_1
#               ├── task_2
#               └── ...
#     
#
#   Each task:
#       prompt + models + actions
#              ↓
#         BERT encoder
#              ↓
#              X
#              ↓
#         NP Encoder
#              ↓
#             r_i
#              ↓
#        Aggregation
#              ↓
#              z
#              ↓
#         NP Decoder
#              ↓
#       models + actions
#
#
# META-EVALUATION:
#
#   New unseen composed task
#              ↓
#         BERT encoder
#              ↓
#        new task embedding
#              +
#        learned z
#              ↓
#         NP Decoder
#              ↓
#       models + actions
#
#   Execution order / chronology is NOT predicted by NP.
# ============================================================


import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModel
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. META-TRAINING TASKS
# ============================================================

# meta_training_tasks:
#
# Dictionary organized according to task nature.
#
# Each key represents a task nature:
#
#   perception
#   navigation
#   grasping
#   human_robot_interaction
#   ...
#
# Each value is another dictionary containing MANY
# previously observed tasks belonging to that task nature.
#
# Each task contains:
#
#   prompt  -> natural-language task description
#   models  -> required models / functional modules
#   actions -> required actions
#
# These are the SEEN tasks used during meta-training.
#
# The new composed task used during evaluation is NOT
# contained in this dictionary.

meta_training_tasks = {

    "perception": {

        "task_1": {
            "prompt": "Detect objects in front of the robot",
            "models": [
                "YOLO"
            ],
            "actions": [
                "Capture image",
                "Detect objects"
            ]
        },

        "task_2": {
            "prompt": "Recognize the person in front of the robot",
            "models": [
                "Dlib"
            ],
            "actions": [
                "Capture image",
                "Detect face",
                "Recognize person"
            ]
        },

        "task_3": {
            "prompt": "Identify the bottle on the table",
            "models": [
                "YOLO"
            ],
            "actions": [
                "Detect bottle",
                "Recognize bottle"
            ]
        }

        # ... many more perception tasks
    },


    "navigation": {

        "task_1": {
            "prompt": "Move to the table",
            "models": [
                "MiDaS",
                "Coordinate Transformation",
                "Inverse Kinematics"
            ],
            "actions": [
                "Locate table",
                "Estimate depth",
                "Transfer coordinates",
                "Solve inverse kinematics",
                "Navigate to table"
            ]
        },

        "task_2": {
            "prompt": "Go to the person",
            "models": [
                "YOLO",
                "MiDaS",
                "Coordinate Transformation",
                "Inverse Kinematics"
            ],
            "actions": [
                "Detect person",
                "Estimate position",
                "Estimate depth",
                "Transfer coordinates",
                "Solve inverse kinematics",
                "Navigate to person"
            ]
        },

        "task_3": {
            "prompt": "Navigate to the object",
            "models": [
                "YOLO",
                "MiDaS",
                "Coordinate Transformation",
                "Inverse Kinematics"
            ],
            "actions": [
                "Detect object",
                "Locate object",
                "Estimate depth",
                "Transfer coordinates",
                "Solve inverse kinematics",
                "Navigate to object"
            ]
        }

        # ... many more navigation tasks
    },


    "grasping": {

        "task_1": {
            "prompt": "Pick up the bottle",
            "models": [
                "YOLO",
                "MiDaS",
                "Coordinate Transformation",
                "Inverse Kinematics"
            ],
            "actions": [
                "Locate bottle",
                "Estimate depth",
                "Transfer coordinates",
                "Solve inverse kinematics",
                "Grasp bottle"
            ]
        },

        "task_2": {
            "prompt": "Pick up the object from the table",
            "models": [
                "YOLO",
                "MiDaS",
                "Coordinate Transformation",
                "Inverse Kinematics"
            ],
            "actions": [
                "Detect object",
                "Locate object",
                "Estimate depth",
                "Transfer coordinates",
                "Solve inverse kinematics",
                "Grasp object"
            ]
        }

        # ... many more grasping tasks
    },


    "human_robot_interaction": {

        "task_1": {
            "prompt": "Answer the user's question",
            "models": [
                "Speech Recognition",
                "Gemini 1.5 Flash",
                "TTS"
            ],
            "actions": [
                "Understand request",
                "Generate response",
                "Speak"
            ]
        },

        "task_2": {
            "prompt": "Have a conversation with the user",
            "models": [
                "Speech Recognition",
                "Gemini 1.5 Flash",
                "TTS"
            ],
            "actions": [
                "Understand speech",
                "Generate response",
                "Speak"
            ]
        }

        # ... many more human-robot interaction tasks
    }

    # ... additional task natures ( we used more than 15 task natures)
}


# ============================================================
# 3. FLATTEN TASK HIERARCHY
# ============================================================

# Convert the hierarchical dictionary into a single collection
# for NP meta-training while preserving the task information.

meta_training_data = {

    f"{task_nature}_{task_id}": task

    for task_nature, tasks_of_nature
    in meta_training_tasks.items()

    for task_id, task
    in tasks_of_nature.items()
}


# ============================================================
# 4. OUTPUT VOCABULARIES
# ============================================================

# The NP decoder operates in a fixed output space learned
# from the meta-training tasks.
#
# The output space contains:
#
#       required models
#       +
#       required actions
#
# New tasks must be decoded into this same output space.

MODEL_VOCAB = sorted({

    model

    for task in meta_training_data.values()

    for model in task["models"]
})


ACTION_VOCAB = sorted({

    action

    for task in meta_training_data.values()

    for action in task["actions"]
})


model_to_id = {
    model: i
    for i, model in enumerate(MODEL_VOCAB)
}


action_to_id = {
    action: i
    for i, action in enumerate(ACTION_VOCAB)
}


num_models = len(MODEL_VOCAB)
num_actions = len(ACTION_VOCAB)

output_dim = (
    num_models +
    num_actions
)


# ============================================================
# 5. BERT
# ============================================================

# BERT independently converts every natural-language
# task prompt into a semantic numerical representation.
#
# BERT is NOT the Neural Process.
#
# BERT:
#
#       prompt → semantic embedding
#
# NP:
#
#       semantic embedding + task output
#                         ↓
#                       r_i
#                         ↓
#                         z

BERT_MODEL_NAME = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(
    BERT_MODEL_NAME
)


bert = AutoModel.from_pretrained(
    BERT_MODEL_NAME
).to(device)


# ============================================================
# 6. BERT PROMPT ENCODING
# ============================================================

def encode_prompts(prompts):

    """
    Convert natural-language prompts into BERT embeddings.

    Input:
        prompts = list[str]

    Output:
        embeddings:
            [N_tasks, bert_hidden_dim]
    """

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    outputs = bert(
        **encoded
    )

    hidden_states = (
        outputs.last_hidden_state
    )

    attention_mask = (
        encoded["attention_mask"]
        .unsqueeze(-1)
        .float()
    )

    masked_hidden_states = (
        hidden_states *
        attention_mask
    )

    embeddings = (
        masked_hidden_states.sum(dim=1)
        /
        attention_mask.sum(dim=1)
        .clamp(min=1.0)
    )

    return embeddings


# ============================================================
# 7. OUTPUT ENCODING
# ============================================================

def encode_outputs(tasks):

    """
    Convert symbolic task outputs into deterministic
    multi-label vectors.

    Vector structure:

        [models | actions]

    A value of 1 means that the model/action is required
    for that task.
    """

    encoded_outputs = []

    for task in tasks:

        vector = torch.zeros(
            output_dim,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        for model_name in task["models"]:

            model_id = model_to_id[
                model_name
            ]

            vector[
                model_id
            ] = 1.0

        # ----------------------------------------------------
        # Actions
        # ----------------------------------------------------

        for action_name in task["actions"]:

            action_id = action_to_id[
                action_name
            ]

            vector[
                num_models + action_id
            ] = 1.0

        encoded_outputs.append(
            vector
        )

    return torch.stack(
        encoded_outputs
    ).to(device)


# ============================================================
# 8. PREPARE META-TRAINING SET
# ============================================================

training_prompts = [
    task["prompt"]
    for task in meta_training_data.values()
]


Y = encode_outputs(
    list(meta_training_data.values())
)


# ============================================================
# 9. NP ENCODER
# ============================================================

class NPEncoder(nn.Module):

    def __init__(
        self,
        prompt_dim,
        output_dim,
        representation_dim=256
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                prompt_dim + output_dim,
                512
            ),

            nn.LayerNorm(512),

            nn.GELU(),

            nn.Dropout(0.10),

            nn.Linear(
                512,
                256
            ),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Linear(
                256,
                representation_dim
            )
        )


    def forward(
        self,
        x,
        y
    ):

        # x = BERT embedding of task prompt
        #
        # y = known models + actions for that task
        #
        # output = individual task representation r_i

        encoder_input = torch.cat(
            [x, y],
            dim=-1
        )

        r = self.network(
            encoder_input
        )

        return r


# ============================================================
# 10. NP AGGREGATOR
# ============================================================

class NPAggregator(nn.Module):

    def __init__(
        self,
        representation_dim=256,
        latent_dim=128
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                representation_dim,
                256
            ),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Linear(
                256,
                latent_dim
            )
        )


    def forward(
        self,
        r
    ):

        # r:
        #
        # [r_1, r_2, ..., r_N]
        #
        # The aggregation must be permutation-invariant.
        #
        # Therefore the order in which training tasks
        # are presented does not change the resulting z.

        aggregated = r.mean(
            dim=0,
            keepdim=True
        )

        z = self.network(
            aggregated
        )

        return z


# ============================================================
# 11. NP DECODER
# ============================================================

class NPDecoder(nn.Module):

    def __init__(
        self,
        prompt_dim,
        latent_dim,
        output_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                prompt_dim + latent_dim,
                512
            ),

            nn.LayerNorm(512),

            nn.GELU(),

            nn.Dropout(0.10),

            nn.Linear(
                512,
                256
            ),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Linear(
                256,
                output_dim
            )
        )


    def forward(
        self,
        x,
        z
    ):

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Same learned global z is used with the task prompt.

        z = z.expand(
            x.size(0),
            -1
        )

        decoder_input = torch.cat(
            [x, z],
            dim=-1
        )

        logits = self.network(
            decoder_input
        )

        return logits


# ============================================================
# 12. COMPLETE NP META-LEARNER
# ============================================================

class NeuralProcessMetaLearner(nn.Module):

    def __init__(
        self,
        prompt_dim,
        output_dim,
        representation_dim=256,
        latent_dim=128
    ):

        super().__init__()

        self.encoder = NPEncoder(
            prompt_dim=prompt_dim,
            output_dim=output_dim,
            representation_dim=representation_dim
        )

        self.aggregator = NPAggregator(
            representation_dim=representation_dim,
            latent_dim=latent_dim
        )

        self.decoder = NPDecoder(
            prompt_dim=prompt_dim,
            latent_dim=latent_dim,
            output_dim=output_dim
        )


    def infer_z(
        self,
        x,
        y
    ):

        # Encode individual seen tasks.

        r = self.encoder(
            x,
            y
        )

        # Aggregate all task representations.

        z = self.aggregator(
            r
        )

        return z


    def forward(
        self,
        x,
        y
    ):

        z = self.infer_z(
            x,
            y
        )

        prediction = self.decoder(
            x,
            z
        )

        return prediction, z


# ============================================================
# 13. INITIALIZE NP
# ============================================================

with torch.no_grad():

    sample_embedding = encode_prompts(
        [training_prompts[0]]
    )


bert_hidden_dim = (
    sample_embedding.shape[-1]
)


model = NeuralProcessMetaLearner(

    prompt_dim=bert_hidden_dim,

    output_dim=output_dim,

    representation_dim=256,

    latent_dim=128

).to(device)


# ============================================================
# 14. OPTIMIZER
# ============================================================

# Joint optimization:
#
#   BERT parameters
#       +
#   NP parameters
#
# BERT receives a smaller learning rate because it is
# pretrained, while the NP layers are learned from scratch.

optimizer = torch.optim.AdamW(

    [
        {
            "params": bert.parameters(),
            "lr": 2e-5
        },

        {
            "params": model.parameters(),
            "lr": 1e-3
        }
    ],

    weight_decay=1e-4
)


# ============================================================
# 15. META-TRAINING
# ============================================================

num_epochs = 1000


for epoch in range(
    num_epochs
):

    model.train()
    bert.train()

    optimizer.zero_grad()


    # --------------------------------------------------------
    # STEP 1
    # BERT:
    #
    # task prompt → semantic task embedding
    # --------------------------------------------------------

    X = encode_prompts(
        training_prompts
    )


    # --------------------------------------------------------
    # STEP 2
    # NP ENCODER:
    #
    # (prompt embedding + known models/actions)
    #                         ↓
    #                        r_i
    # --------------------------------------------------------

    # --------------------------------------------------------
    # STEP 3
    # AGGREGATOR:
    #
    # r_1 ... r_N
    #      ↓
    #      z
    # --------------------------------------------------------

    # --------------------------------------------------------
    # STEP 4
    # DECODER:
    #
    # prompt embedding + z
    #      ↓
    # models + actions
    # --------------------------------------------------------

    predictions, z = model(
        X,
        Y
    )


    # --------------------------------------------------------
    # STEP 5
    # TRAINING LOSS
    # --------------------------------------------------------

    loss = F.binary_cross_entropy_with_logits(
        predictions,
        Y
    )


    # --------------------------------------------------------
    # STEP 6
    # BACKPROPAGATION
    # --------------------------------------------------------

    loss.backward()


    # Gradient clipping for stable optimization.

    torch.nn.utils.clip_grad_norm_(
        list(bert.parameters())
        +
        list(model.parameters()),
        max_norm=1.0
    )


    optimizer.step()


    if epoch % 100 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# 16. EXTRACT LEARNED z
# ============================================================

model.eval()
bert.eval()


with torch.no_grad():

    X = encode_prompts(
        training_prompts
    )

    learned_z = model.infer_z(
        X,
        Y
    )

    learned_z = learned_z.detach()


print(
    "\nLearned latent z shape:",
    learned_z.shape
)


# ============================================================
# 17. NEW UNSEEN COMPOSED TASK
# ============================================================

# new_task_prompt:
#
# A completely NEW composed task.
#
# It was NOT included in meta_training_tasks.
#
# It can combine capabilities from different task natures.

new_task_prompt = (
    "new unseen composed task prompt"
)


# ============================================================
# 18. ENCODE NEW TASK WITH BERT
# ============================================================

with torch.no_grad():

    new_task_embedding = encode_prompts(
        [new_task_prompt]
    )


# ============================================================
# 19. NP DECODER
# ============================================================

with torch.no_grad():

    logits = model.decoder(
        new_task_embedding,
        learned_z
    )

    probabilities = torch.sigmoid(
        logits
    )[0]


# ============================================================
# 20. DECODE MODELS
# ============================================================

MODEL_THRESHOLD = 0.5


predicted_models = []


for i, probability in enumerate(
    probabilities[:num_models]
):

    if probability >= MODEL_THRESHOLD:

        predicted_models.append(
            {
                "model": MODEL_VOCAB[i],
                "probability": float(
                    probability
                )
            }
        )


# ============================================================
# 21. DECODE ACTIONS
# ============================================================

ACTION_THRESHOLD = 0.5


predicted_actions = []


for i, probability in enumerate(
    probabilities[num_models:]
):

    if probability >= ACTION_THRESHOLD:

        predicted_actions.append(
            {
                "action": ACTION_VOCAB[i],
                "probability": float(
                    probability
                )
            }
        )


# ============================================================
# 22. FINAL NP OUTPUT
# ============================================================

np_output = {

    "models": predicted_models,

    "actions": predicted_actions

}


print(
    "\n========================================"
)

print(
    "UNSEEN COMPOSED TASK"
)

print(
    "========================================"
)

print(
    new_task_prompt
)


print(
    "\nPredicted Models:"
)

for item in predicted_models:

    print(
        f"  {item['model']}: "
        f"{item['probability']:.4f}"
    )


print(
    "\nPredicted Actions:"
)

for item in predicted_actions:

    print(
        f"  {item['action']}: "
        f"{item['probability']:.4f}"
    )


print(
    "\nFinal NP output:"
)

print(
    np_output
)
