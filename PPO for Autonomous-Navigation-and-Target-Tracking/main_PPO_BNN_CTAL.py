#!/usr/bin/env python3

##################################################
# main.py
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
from train import Trainer

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
# ROS Initialization
##################################################

rospy.init_node(

    "pepper_ctal_navigation",

    anonymous=True

)

##################################################
# Project Hyperparameters
##################################################

NUM_TRAINING_CYCLES = 50

NUM_BATCHES = 8

EPISODES_PER_BATCH = 8

PPO_OPTIMIZATION_EPOCHS = 4

M = 10

K = 100

UNCERTAINTY_THRESHOLD = 0.15

##################################################
# Saving Directories
##################################################

MODEL_DIRECTORY = "./weights"

LOG_DIRECTORY = "./logs"

os.makedirs(

    MODEL_DIRECTORY,

    exist_ok=True

)

os.makedirs(

    LOG_DIRECTORY,

    exist_ok=True

  

)

##################################################
# Environment
##################################################

environment = PepperEnvironment(

    training=True

)

##################################################
# Graph Builder
##################################################

graph_builder = GraphBuilder()

##################################################
# Graph Neural Network
##################################################

gnn = GraphSAGE().to(

    DEVICE

)

##################################################
# Bayesian PPO
##################################################

ppo = PPO(

    graph_dim=128,

    social_dim=2,

    gamma=0.99,

    lam=0.95,

    clip=0.2,

    entropy_coef=0.01,

    critic_coef=0.5,

    epochs=PPO_OPTIMIZATION_EPOCHS

)

##################################################
# Counterfactual Trajectories Active Learning
##################################################

ctal = CTAL(

    actor=ppo.actor,

    threshold=UNCERTAINTY_THRESHOLD,

    M=M,

    K=K

)

##################################################
# Trainer
##################################################

trainer = Trainer(

    env=environment,

    graph_builder=graph_builder,

    gnn=gnn,

    ppo=ppo,

    ctal=ctal,

    num_cycles=NUM_TRAINING_CYCLES,

    num_batches=NUM_BATCHES,

    episodes_per_batch=EPISODES_PER_BATCH,

    device=DEVICE,

    model_directory=MODEL_DIRECTORY,

    log_directory=LOG_DIRECTORY

)

##################################################
# Main
##################################################

if __name__ == "__main__":

    try:

        print("\n===================================")
        print(" Bayesian PPO + CTAL ")
        print(" Pepper Robot Navigation ")
        print("===================================\n")

        ##################################################
        # Start Training
        ##################################################

        trainer.train()

        ##################################################
        # Save Final Models
        ##################################################

        ppo.save(

            MODEL_DIRECTORY

        )

        print(

            "\nTraining Finished Successfully."

        )

    ##################################################
    # ROS Exception
    ##################################################

    except rospy.ROSInterruptException:

        print(

            "\nROS Interrupted."

        )

    ##################################################
    # Keyboard Interrupt
    ##################################################

    except KeyboardInterrupt:

        print(

            "\nTraining Interrupted."

        )

        ppo.save(

            MODEL_DIRECTORY

        )

    ##################################################
    # Unexpected Exception
    ##################################################

    except Exception as e:

        print(

            "\nUnexpected Error:",

            str(e)

        )

        ppo.save(

            MODEL_DIRECTORY

        )

        raise
