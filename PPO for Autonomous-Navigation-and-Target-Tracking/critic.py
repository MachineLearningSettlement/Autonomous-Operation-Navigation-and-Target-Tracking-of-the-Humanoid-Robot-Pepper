#!/usr/bin/env python3

##################################################
# critic.py
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################
# PPO Critic Network
##################################################

class Critic(nn.Module):

    def __init__(

            self,

            graph_dim=128,

            social_dim=2,

            hidden_dim=256

    ):

        super(Critic, self).__init__()

        ##################################################
        # Input Dimension
        ##################################################

        input_dim = graph_dim + social_dim

        ##################################################
        # Hidden Layers
        ##################################################

        self.fc1 = nn.Linear(

            input_dim,

            hidden_dim

        )

        self.fc2 = nn.Linear(

            hidden_dim,

            hidden_dim

        )

        self.fc3 = nn.Linear(

            hidden_dim,

            hidden_dim

        )

        ##################################################
        # Value Head
        ##################################################

        self.value = nn.Linear(

            hidden_dim,

            1

        )

        ##################################################
        # Initialization
        ##################################################

        nn.init.xavier_uniform_(

            self.fc1.weight

        )

        nn.init.zeros_(

            self.fc1.bias

        )

        nn.init.xavier_uniform_(

            self.fc2.weight

        )

        nn.init.zeros_(

            self.fc2.bias

        )

        nn.init.xavier_uniform_(

            self.fc3.weight

        )

        nn.init.zeros_(

            self.fc3.bias

        )

        nn.init.xavier_uniform_(

            self.value.weight

        )

        nn.init.zeros_(

            self.value.bias

        )

    ##################################################
    # Forward
    ##################################################

    def forward(

            self,

            graph_embedding,

            social_features

    ):

        ##################################################
        # Fusion
        ##################################################

        x = torch.cat(

            [

                graph_embedding,

                social_features

            ],

            dim=1

        )

        ##################################################
        # Hidden Layer 1
        ##################################################

        x = self.fc1(x)

        x = F.relu(x)

        ##################################################
        # Hidden Layer 2
        ##################################################

        x = self.fc2(x)

        x = F.relu(x)

        ##################################################
        # Hidden Layer 3
        ##################################################

        x = self.fc3(x)

        x = F.relu(x)

        ##################################################
        # State Value
        ##################################################

        value = self.value(x)

        return value
