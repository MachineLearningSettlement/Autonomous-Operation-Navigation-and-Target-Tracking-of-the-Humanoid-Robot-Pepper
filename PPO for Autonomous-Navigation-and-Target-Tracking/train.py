#!/usr/bin/env python3

##################################################
# train.py
##################################################

import os
import rospy
import torch
import numpy as np

from environment import PepperEnvironment
from gnn import GraphSAGE
from ppo import PPO
from ctal import CTAL


##################################################
# Hyperparameters
##################################################

NUM_CYCLES = 50

NUM_BATCHES = 8

EPISODES_PER_BATCH = 8

PPO_EPOCHS = 4

MAX_EPISODES = NUM_BATCHES * EPISODES_PER_BATCH

GAMMA = 0.99

LAMBDA = 0.95

THRESHOLD = 0.15

M = 10

K = 100

SAVE_PATH = "./weights/"


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
# GNN
##################################################

gnn = GraphSAGE()


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
# Create Saving Folder
##################################################

os.makedirs(
    SAVE_PATH,
    exist_ok=True
)

##################################################
# Training
##################################################

for cycle in range(NUM_CYCLES):

    print("\n========================================")
    print("Training Cycle :", cycle + 1)
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

        ##################################################
        # Episode Average Uncertainty
        ##################################################

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

            episode_uncertainty = []

            ##################################################
            # One Episode = One Trajectory
            ##################################################

            while not done:

                ##################################################
                # Graph Embedding
                ##################################################

                graph_embedding = gnn(
                    graph_state
                )

                ##################################################
                # Social Features
                ##################################################

                social_features = torch.tensor(
                    [[
                        env.positive_similarity,
                        env.negative_similarity
                    ]],
                    dtype=torch.float32
                )

                ##################################################
                # Actor
                ##################################################

                (
                    action,
                    log_prob,
                    entropy,
                    mu,
                    sigma,
                    variance,
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
                    arrive
                ) = env.step(
                    action.squeeze(0).detach().cpu().numpy()
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
                # Episode Uncertainty
                ##################################################

                episode_uncertainty.append(
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
                episode_uncertainty
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
        # Bootstrap Value
        ##################################################

        next_value = torch.zeros(
            1,
            1
        )

        ##################################################
        # PPO Update
        ##################################################

        statistics = ppo.update(
            next_value
        )

        ##################################################
        # Average Batch Uncertainty
        ##################################################

        avg_batch_uncertainty = torch.stack(
            batch_uncertainties
        ).mean()

        print(
            "Average Batch Uncertainty :",
            avg_batch_uncertainty.item()
        )

        print(
            "Actor Loss :", statistics["actor_loss"]
        )

        print(
            "Critic Loss :", statistics["critic_loss"]
        )

        print(
            "Current Stage :", ctal.get_stage()
        )

        ##################################################
        # Stage 2 : Counterfactual Active Learning
        ##################################################

        if ctal.get_stage() == 2:

            print(
                "\nRunning Counterfactual Trajectories Active Learning..."
            )

            ##################################################
            # Loop Through Stored Trajectories
            ##################################################

            for i in range(len(ppo.memory.states)):

                ##################################################
                # State Uncertainty
                ##################################################

                state_uncertainty = ppo.memory.uncertainties[i]

                ##################################################
                # Trigger Counterfactual Analysis
                ##################################################

                if ctal.trigger_counterfactual(
                        state_uncertainty
                ):

                    result = ctal.counterfactual_policy_analysis(
                        graph_embedding=ppo.memory.states[i],
                        social_features=ppo.memory.social_features[i],
                        advantages=torch.tensor([[1.0]])
                    )

                    ##################################################
                    # Replace Action
                    ##################################################

                    ppo.memory.actions[i] = result["best_action"]

        ##################################################
        # Save Models
        ##################################################

        if (cycle + 1) % 10 == 0:

            ppo.save(
                SAVE_PATH
            )

            print(
                "\nModels saved."
            )

##################################################
# Final Save
##################################################

ppo.save(
    SAVE_PATH
)

print(
    "\nTraining Finished."
)
