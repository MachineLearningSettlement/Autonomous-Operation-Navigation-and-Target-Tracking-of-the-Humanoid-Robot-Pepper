#!/usr/bin/env python3

##################################################
# train.py
##################################################

import os
import random
import numpy as np
import torch
import rospy

##################################################
# Project Modules
##################################################

from environment import PepperEnvironment
from graph_builder import GraphBuilder
from gnn import GraphSAGE
from ppo import PPO
from ctal import CTAL

##################################################
# Random Seed
##################################################

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

##################################################
# Device
##################################################

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

##################################################
# Hyperparameters
##################################################

NUM_CYCLES = 50

NUM_BATCHES = 8

EPISODES_PER_BATCH = 8

PPO_EPOCHS = 4

GAMMA = 0.99

LAMBDA = 0.95

THRESHOLD = 0.15

M = 10

K = 100

##################################################
# Weights
##################################################

PRETRAINED_GNN_DIRECTORY = "./weights/pretrained_gnn.pth"

PPO_WEIGHTS_DIRECTORY = "./weights/ppo"

os.makedirs(

    PPO_WEIGHTS_DIRECTORY,

    exist_ok=True

)

##################################################
# ROS
##################################################

rospy.init_node(

    "ppo_training"

)

##################################################
# Environment
##################################################

env = PepperEnvironment(

    training=True

)

##################################################
# Graph Builder
##################################################

graph_builder = GraphBuilder()

##################################################
# Load Pretrained GNN
##################################################

gnn = GraphSAGE().to(

    DEVICE

)

gnn.load_pretrained(

    PRETRAINED_GNN_DIRECTORY,

    device=DEVICE

)

##################################################
# PPO
##################################################

ppo = PPO(

    graph_dim=128,

    social_dim=2,

    gamma=GAMMA,

    lam=LAMBDA,

    epochs=PPO_EPOCHS

)

##################################################
# CTAL
##################################################

ctal = CTAL(

    actor=ppo.actor,

    threshold=THRESHOLD,

    M=M,

    K=K

)


##################################################
# Training
##################################################

for cycle in range(NUM_CYCLES):

    print("\n========================================")

    print(

        "Training Cycle :",

        cycle + 1

    )

    print("========================================")

    ##################################################
    # 8 Batches
    ##################################################

    for batch in range(NUM_BATCHES):

        print(

            "\nBatch :",

            batch + 1,

            "/",

            NUM_BATCHES

        )

        batch_uncertainties = []

        ##################################################
        # 8 Episodes
        ##################################################

        for episode in range(EPISODES_PER_BATCH):

            ##################################################
            # Reset Environment
            ##################################################

            graph_state = env.reset()

            done = False

            episode_uncertainties = []

            ##################################################
            # One Episode = One Trajectory
            ##################################################

            while not done:

                ##################################################
                # Graph
                ##################################################

                graph = graph_builder.build_graph(

                    graph_state

                ).to(

                    DEVICE

                )

                ##################################################
                # Graph Embedding
                ##################################################

                with torch.no_grad():

                    graph_embedding = gnn(

                        graph

                    )

                ##################################################
                # Social Features
                ##################################################

                social_features = torch.tensor(

                    [[

                        env.positive_similarity,

                        env.negative_similarity

                    ]],

                    dtype=torch.float32,

                    device=DEVICE

                )

                ##################################################
                # Bayesian PPO Actor
                ##################################################

                (

                    action,

                    log_prob,

                    entropy,

                    mu,

                    sigma,

                    predictive_variance,

                    U_epi

                ) = ppo.actor(

                    graph_embedding,

                    social_features

                )

                ##################################################
                # Critic
                ##################################################

                value = ppo.critic(

                    graph_embedding,

                    social_features

                )

                ##################################################
                # Execute Action
                ##################################################

                (

                    next_graph_state,

                    reward,

                    done,

                    info

                ) = env.step(

                    action.squeeze(0)

                    .detach()

                    .cpu()

                    .numpy()

                )

                ##################################################
                # Store Transition
                ##################################################

                ppo.memory.store(

                    graph_embedding,

                    action,

                    log_prob,

                    reward,

                    value,

                    done,

                    social_features,

                    U_epi

                )

                ##################################################
                # Store Uncertainty
                ##################################################

                episode_uncertainties.append(

                    U_epi.detach()

                )

                ##################################################
                # Next State
                ##################################################

                graph_state = next_graph_state

            ##################################################
            # End of Episode
            ##################################################

            episode_uncertainty = torch.stack(

                episode_uncertainties

            ).mean()

            batch_uncertainties.append(

                episode_uncertainty

            )

            ##################################################
            # Stage 1 Monitoring
            ##################################################

            if ctal.get_stage() == 1:

                triggered = ctal.update_episode_uncertainty(

                    episode_uncertainty

                )

                if triggered:

                    print(

                        "\n>>> CTAL Stage 2 Activated <<<"

                    )

        ##################################################
        # Stage 2 : Counterfactual Active Learning
        ##################################################

        if ctal.get_stage() == 2:

            ##################################################
            # Compute PPO Advantages
            ##################################################

            advantages, returns = ppo.compute_GAE(

                next_value=0.0

            )

            ##################################################
            # Counterfactual Analysis
            ##################################################

            for i in range(len(ppo.memory.states)):

                if ctal.trigger_counterfactual(

                    ppo.memory.uncertainties[i]

                ):

                    ctal.counterfactual_policy_analysis(

                        graph_embedding=ppo.memory.states[i],

                        social_features=ppo.memory.social_features[i],

                        advantages=advantages[i].unsqueeze(0)

                    )

        ##################################################
        # PPO Update
        ##################################################

        statistics = ppo.update(

            next_value=0.0

        )

        ##################################################
        # Batch Statistics
        ##################################################

        average_batch_uncertainty = torch.stack(

            batch_uncertainties

        ).mean()

        ##################################################
        # Display Training Statistics
        ##################################################

        print(

            "Average Batch Uncertainty :",

            average_batch_uncertainty.item()

        )

        print(

            "Actor Loss :",

            statistics["actor_loss"]

        )

        print(

            "Critic Loss :",

            statistics["critic_loss"]

        )

        print(

            "Current Stage :",

            ctal.get_stage()

        )

        ##################################################
        # Save Models Every 10 Cycles
        ##################################################

        if (

            cycle + 1

        ) % 10 == 0:

            ppo.save(

                PPO_WEIGHTS_DIRECTORY

            )

            print(

                "\nModels Saved."

            )

##################################################
# Final Save
##################################################

ppo.save(

    PPO_WEIGHTS_DIRECTORY

)

print(

    "\nTraining Finished Successfully."

)
