#!/usr/bin/env python3

##################################################
# gnn.py
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GINEConv,
    global_mean_pool
)


##################################################
# Pretrained GNN Path
##################################################

PRETRAINED_GNN_PATH = "./weights/pretrained_gnn.pth"


##################################################
# Graph Neural Network
##################################################

class GraphSAGE(nn.Module):

    ##################################################
    # Initialization
    ##################################################

    def __init__(

            self,

            hidden_dim=128,

            embedding_dim=128,

            edge_dim=128,

            dropout=0.20

    ):

        super(GraphSAGE, self).__init__()

        ##################################################
        # Parameters
        ##################################################

        self.hidden_dim = hidden_dim

        self.embedding_dim = embedding_dim

        self.edge_dim = edge_dim

        self.dropout = dropout

        ##################################################
        # Robot MLP
        #
        # 6 → 32 → 64 → 128
        ##################################################

        self.robot_encoder = nn.Sequential(

            nn.Linear(
                6,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                128
            ),

            nn.ReLU()

        )

        ##################################################
        # Goal MLP
        #
        # 3 → 32 → 64 → 128
        ##################################################

        self.goal_encoder = nn.Sequential(

            nn.Linear(
                3,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                128
            ),

            nn.ReLU()

        )

        ##################################################
        # Obstacle MLP
        #
        # 2 → 16 → 32 → 128
        ##################################################

        self.obstacle_encoder = nn.Sequential(

            nn.Linear(
                2,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                128
            ),

            nn.ReLU()

        )

        ##################################################
        # Human MLP
        #
        # 1029 → 512 → 256 → 128
        ##################################################

        self.human_encoder = nn.Sequential(

            nn.Linear(
                1029,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU()

        )

        ##################################################
        # Edge MLP
        #
        # Distance:
        #
        # 1 → 32 → 64 → 128
        ##################################################

        self.edge_encoder = nn.Sequential(

            nn.Linear(
                1,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                128
            ),

            nn.ReLU()

        )

        ##################################################
        # Edge-Aware GNN Layer 1
        ##################################################

        self.conv1 = GINEConv(

            nn.Sequential(

                nn.Linear(
                    128,
                    128
                ),

                nn.ReLU(),

                nn.Linear(
                    128,
                    128
                )

            ),

            edge_dim=128

        )

        ##################################################
        # Edge-Aware GNN Layer 2
        ##################################################

        self.conv2 = GINEConv(

            nn.Sequential(

                nn.Linear(
                    128,
                    128
                ),

                nn.ReLU(),

                nn.Linear(
                    128,
                    128
                )

            ),

            edge_dim=128

        )

        ##################################################
        # Edge-Aware GNN Layer 3
        ##################################################

        self.conv3 = GINEConv(

            nn.Sequential(

                nn.Linear(
                    128,
                    128
                ),

                nn.ReLU(),

                nn.Linear(
                    128,
                    128
                )

            ),

            edge_dim=128

        )

        ##################################################
        # Batch Normalization
        ##################################################

        self.bn1 = nn.BatchNorm1d(
            128
        )

        self.bn2 = nn.BatchNorm1d(
            128
        )

        self.bn3 = nn.BatchNorm1d(
            128
        )

        ##################################################
        # Dropout
        ##################################################

        self.drop = nn.Dropout(
            p=dropout
        )

        ##################################################
        # Graph Embedding Head
        ##################################################

        self.graph_head = nn.Sequential(

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                128,
                128
            )

        )

    ##################################################
    # Encode Nodes
    ##################################################

    def encode_nodes(

            self,

            data

    ):

        ##################################################
        # Robot
        ##################################################

        robot_x = self.robot_encoder(

            data["robot"].x

        )

        ##################################################
        # Goal
        ##################################################

        goal_x = self.goal_encoder(

            data["goal"].x

        )

        ##################################################
        # Obstacles
        ##################################################

        obstacle_x = self.obstacle_encoder(

            data["obstacle"].x

        )

        ##################################################
        # Humans
        ##################################################

        human_x = self.human_encoder(

            data["human"].x

        )

        ##################################################
        # Combine All Node Embeddings
        ##################################################

        x = torch.cat(

            [

                robot_x,

                goal_x,

                obstacle_x,

                human_x

            ],

            dim=0

        )

        return x

    ##################################################
    # Encode Edges
    ##################################################

    def encode_edges(

            self,

            data

    ):

        ##################################################
        # Raw Edge Distance
        ##################################################

        edge_attr = data.edge_attr

        ##################################################
        # Edge MLP
        ##################################################

        edge_attr = self.edge_encoder(

            edge_attr

        )

        return edge_attr

    ##################################################
    # Forward
    ##################################################

    def forward(

            self,

            data

    ):

        ##################################################
        # Node Embeddings
        ##################################################

        x = self.encode_nodes(

            data

        )

        ##################################################
        # Edge Embeddings
        ##################################################

        edge_attr = self.encode_edges(

            data

        )

        ##################################################
        # Edge Connections
        ##################################################

        edge_index = data.edge_index

        ##################################################
        # First GNN Layer
        ##################################################

        x = self.conv1(

            x,

            edge_index,

            edge_attr

        )

        x = self.bn1(

            x

        )

        x = F.relu(

            x

        )

        x = self.drop(

            x

        )

        ##################################################
        # Second GNN Layer
        ##################################################

        x = self.conv2(

            x,

            edge_index,

            edge_attr

        )

        x = self.bn2(

            x

        )

        x = F.relu(

            x

        )

        x = self.drop(

            x

        )

        ##################################################
        # Third GNN Layer
        ##################################################

        x = self.conv3(

            x,

            edge_index,

            edge_attr

        )

        x = self.bn3(

            x

        )

        x = F.relu(

            x

        )

        ##################################################
        # Global Mean Pooling
        ##################################################

        graph_embedding = global_mean_pool(

            x,

            data.batch

        )

        ##################################################
        # Graph Embedding Head
        ##################################################

        graph_embedding = self.graph_head(

            graph_embedding

        )

        ##################################################
        # L2 Normalization
        ##################################################

        graph_embedding = F.normalize(

            graph_embedding,

            p=2,

            dim=1

        )

        ##################################################
        # Output
        ##################################################

        return graph_embedding

    ##################################################
    # Load Pretrained GNN
    ##################################################

    def load_pretrained(

            self,

            device="cpu"

    ):

        ##################################################
        # Load Checkpoint
        ##################################################

        checkpoint = torch.load(

            PRETRAINED_GNN_PATH,

            map_location=device

        )

        ##################################################
        # Load Parameters
        ##################################################

        self.load_state_dict(

            checkpoint

        )

        ##################################################
        # Evaluation Mode
        ##################################################

        self.eval()

        ##################################################
        # Freeze Parameters
        ##################################################

        for parameter in self.parameters():

            parameter.requires_grad = False

        print(

            "\nPretrained GNN loaded successfully."

        )
