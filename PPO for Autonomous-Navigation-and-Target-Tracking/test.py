#!/usr/bin/env python3

##################################################
# test.py
##################################################

import random
import numpy as np
import torch
import rospy

from environment import PepperEnvironment
from graph_builder import GraphBuilder
from gnn import GraphSAGE
from ppo import PPO

##################################################
# Random Seed
##################################################

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
# ROS
##################################################

rospy.init_node(

    "pepper_navigation_test",

    anonymous=True

)

##################################################
# Test Configuration
##################################################

NUM_TEST_TARGETS = 80

##################################################
# Weights
##################################################

PRETRAINED_GNN_DIRECTORY = "./weights/pretrained_gnn.pth"

PPO_WEIGHTS_DIRECTORY = "./weights/ppo"

##################################################
# Environment
##################################################

environment = PepperEnvironment(

    training=False

)

##################################################
# 80 Unseen Targets (5 per 4×4 square)
##################################################

test_targets = [

    (0.5,0.5),(1.5,1.5),(2.5,2.5),(3.2,0.8),(0.8,3.2),
    (4.5,0.5),(5.5,1.5),(6.5,2.5),(7.2,0.8),(4.8,3.2),
    (8.5,0.5),(9.5,1.5),(10.5,2.5),(11.2,0.8),(8.8,3.2),
    (12.5,0.5),(13.5,1.5),(14.5,2.5),(15.2,0.8),(12.8,3.2),

    (0.5,4.5),(1.5,5.5),(2.5,6.5),(3.2,4.8),(0.8,7.2),
    (4.5,4.5),(5.5,5.5),(6.5,6.5),(7.2,4.8),(4.8,7.2),
    (8.5,4.5),(9.5,5.5),(10.5,6.5),(11.2,4.8),(8.8,7.2),
    (12.5,4.5),(13.5,5.5),(14.5,6.5),(15.2,4.8),(12.8,7.2),

    (0.5,8.5),(1.5,9.5),(2.5,10.5),(3.2,8.8),(0.8,11.2),
    (4.5,8.5),(5.5,9.5),(6.5,10.5),(7.2,8.8),(4.8,11.2),
    (8.5,8.5),(9.5,9.5),(10.5,10.5),(11.2,8.8),(8.8,11.2),
    (12.5,8.5),(13.5,9.5),(14.5,10.5),(11.2,12.8),(8.8,11.2),

    (0.5,12.5),(1.5,13.5),(2.5,14.5),(3.2,12.8),(0.8,15.2),
    (4.5,12.5),(5.5,13.5),(6.5,14.5),(7.2,12.8),(4.8,15.2),
    (8.5,12.5),(9.5,13.5),(10.5,14.5),(11.2,12.8),(8.8,15.2),
    (12.5,12.5),(13.5,13.5),(14.5,14.5),(15.2,12.8),(12.8,15.2)

]

##################################################
# Graph Builder
##################################################

graph_builder = GraphBuilder()

##################################################
# Load GNN
##################################################

gnn = GraphSAGE().to(DEVICE)

gnn.load_pretrained(device=DEVICE)

##################################################
# Load PPO
##################################################

ppo = PPO(

    graph_dim=128,

    social_dim=2

)

ppo.load(PPO_WEIGHTS_DIRECTORY)

ppo.actor.eval()

ppo.critic.eval()

##################################################
# Test Metrics
##################################################

success_count = 0

collision_count = 0

timeout_count = 0

navigation_times = []

trajectory_lengths = []

##################################################
# Test Loop
##################################################

for episode in range(NUM_TEST_TARGETS):

    ##################################################
    # Reset Environment
    ##################################################

    state = environment.reset(

        target=test_targets[episode]

    )

    done = False

    navigation_time = 0.0

    trajectory_length = 0.0

    previous_position = np.array(

        state[:2]

    )

    ##################################################
    # One Episode = One Trajectory
    ##################################################

    while not done:

        ##################################################
        # Graph Construction
        ##################################################

        graph = graph_builder.build_graph(

            state

        )

        ##################################################
        # Graph Embedding
        ##################################################

        graph_embedding = gnn(

            graph.to(DEVICE)

        )

        ##################################################
        # Social Feature
        ##################################################

        social_feature = environment.get_social_feature()

        ##################################################
        # Actor Inference
        ##################################################

        with torch.no_grad():

            action, _, _, _, _, _, _ = ppo.actor(

                graph_embedding,

                social_feature.unsqueeze(0)

            )

        ##################################################
        # Execute Action
        ##################################################

        state, reward, done, info = environment.step(

            action.squeeze(0).cpu().numpy()

        )

        ##################################################
        # Navigation Time
        ##################################################

        navigation_time += environment.dt

        ##################################################
        # Trajectory Length
        ##################################################

        current_position = np.array(

            state[:2]

        )

        trajectory_length += np.linalg.norm(

            current_position -

            previous_position

        )

        previous_position = current_position

    ##################################################
    # Episode Statistics
    ##################################################

    if info["success"]:

        success_count += 1

    elif info["collision"]:

        collision_count += 1

    elif info["timeout"]:

        timeout_count += 1

    ##################################################
    # Save Metrics
    ##################################################

    navigation_times.append(

        navigation_time

    )

    trajectory_lengths.append(

        trajectory_length

    )

##################################################
# Final Metrics
##################################################

navigation_success_rate = (

    success_count / NUM_TEST_TARGETS

) * 100.0

collision_rate = (

    collision_count / NUM_TEST_TARGETS

) * 100.0

timeout_rate = (

    timeout_count / NUM_TEST_TARGETS

) * 100.0

average_navigation_time = np.mean(

    navigation_times

)

average_trajectory_length = np.mean(

    trajectory_lengths

)

##################################################
# Display Results
##################################################

print("\n========================================")

print("TEST RESULTS")

print("========================================")

print(

    f"Navigation Success Rate : {navigation_success_rate:.2f}%"

)

print(

    f"Collision Rate          : {collision_rate:.2f}%"

)

print(

    f"Timeout Rate            : {timeout_rate:.2f}%"

)

print(

    f"Average Navigation Time : {average_navigation_time:.2f} s"

)

print(

    f"Average Trajectory Length : {average_trajectory_length:.2f} m"

)

print("========================================")
