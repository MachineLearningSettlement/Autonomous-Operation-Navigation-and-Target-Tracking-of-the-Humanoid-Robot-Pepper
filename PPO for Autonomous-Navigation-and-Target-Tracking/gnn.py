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


PRETRAINED_GNN_PATH = "./weights/pretrained_gnn.pth"


##################################################
# Graph Neural Network
##################################################

class GraphSAGE(nn.Module):

    def __init__(
            self,
            hidden_dim=128,
            embedding_dim=128,
            edge_dim=128,
            dropout=0.20
    ):

        super(GraphSAGE, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.edge_dim = edge_dim
        self.dropout = dropout

        ##################################################
        # Robot Encoder
        # 6 -> 32 -> 64 -> 128
        ##################################################

        self.robot_encoder = nn.Sequential(

            nn.Linear(6, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU()

        )

        ##################################################
        # Goal Encoder
        # 3 -> 32 -> 64 -> 128
        ##################################################

        self.goal_encoder = nn.Sequential(

            nn.Linear(3, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU()

        )

        ##################################################
        # Obstacle Encoder
        # 2 -> 16 -> 32 -> 128
        ##################################################

        self.obstacle_encoder = nn.Sequential(

            nn.Linear(2, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 128),
            nn.ReLU()

        )

        ##################################################
        # Human Encoder
        # 1029 -> 512 -> 256 -> 128
        ##################################################

        self.human_encoder = nn.Sequential(

            nn.Linear(1029, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU()

        )

        ##################################################
        # Edge Encoder
        # 1 -> 32 -> 64 -> 128
        ##################################################

        self.edge_encoder = nn.Sequential(

            nn.Linear(1, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU()

        )

        ##################################################
        # GINE Layer 1
        ##################################################

        self.conv1 = GINEConv(

            nn.Sequential(

                nn.Linear(128, 128),
                nn.ReLU(),

                nn.Linear(128, 128)

            ),

            edge_dim=128

        )

        ##################################################
        # GINE Layer 2
        ##################################################

        self.conv2 = GINEConv(

            nn.Sequential(

                nn.Linear(128, 128),
                nn.ReLU(),

                nn.Linear(128, 128)

            ),

            edge_dim=128

        )

        ##################################################
        # GINE Layer 3
        ##################################################

        self.conv3 = GINEConv(

            nn.Sequential(

                nn.Linear(128, 128),
                nn.ReLU(),

                nn.Linear(128, 128)

            ),

            edge_dim=128

        )

        ##################################################
        # Batch Normalization
        ##################################################

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

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

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(64, 128)

        )

    ##################################################
    # Encode Nodes
    ##################################################

    def encode_nodes(self, data):

        robot_x = self.robot_encoder(
            data.robot_x
        )

        goal_x = self.goal_encoder(
            data.goal_x
        )

        obstacle_x = self.obstacle_encoder(
            data.obstacle_x
        )

        human_x = self.human_encoder(
            data.human_x
        )

        ##################################################
        # Same ordering as graph_builder.py
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

    def encode_edges(self, data):

        return self.edge_encoder(
            data.edge_attr
        )

    ##################################################
    # Forward
    ##################################################

    def forward(self, data):

        x = self.encode_nodes(data)

        edge_attr = self.encode_edges(data)

        edge_index = data.edge_index

        ##################################################
        # GINE Layer 1
        ##################################################

        x = self.conv1(
            x,
            edge_index,
            edge_attr
        )

        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)

        ##################################################
        # GINE Layer 2
        ##################################################

        x = self.conv2(
            x,
            edge_index,
            edge_attr
        )

        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop(x)

        ##################################################
        # GINE Layer 3
        ##################################################

        x = self.conv3(
            x,
            edge_index,
            edge_attr
        )

        x = self.bn3(x)
        x = F.relu(x)

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
        # Final Output
        ##################################################

        return graph_embedding

    ##################################################
    # Load Pretrained GNN
    ##################################################

    def load_pretrained(self, device="cpu"):

        checkpoint = torch.load(
            PRETRAINED_GNN_PATH,
            map_location=device
        )

        self.load_state_dict(
            checkpoint
        )

        self.eval()

        for parameter in self.parameters():
            parameter.requires_grad = False

        print(
            "\nPretrained GNN loaded successfully."
        )
