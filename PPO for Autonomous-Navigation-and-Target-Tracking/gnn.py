#!/usr/bin/env python3

##################################################
# gnn.py
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    SAGEConv,
    global_mean_pool
)

PRETRAINED_GNN_PATH = "./weights/pretrained_gnn.pth"


##################################################
# GraphSAGE Network
##################################################

class GraphSAGE(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        embedding_dim=128,
        dropout=0.20
    ):

        super(GraphSAGE, self).__init__()

        ##################################################
        # Node-Specific MLP Encoders
        ##################################################

        self.robot_encoder = nn.Sequential(

            nn.Linear(6, 32),
            nn.ReLU(),
            
            nn.Linear(32, 86),
            nn.ReLU(),

            nn.Linear(86, 128),
            nn.ReLU()

        )

        self.goal_encoder = nn.Sequential(

            nn.Linear(3, 32),
            nn.ReLU(),

            nn.Linear(32, 86),
            nn.ReLU()

            nn.Linear(86, 128),
            nn.ReLU()

        )

        self.obstacle_encoder = nn.Sequential(

            nn.Linear(2, 32),
            nn.ReLU(),
            
            nn.Linear(32, 86),
            nn.ReLU()

            nn.Linear(86, 128),
            nn.ReLU()

        )

        self.human_encoder = nn.Sequential(

            nn.Linear(1029, 464),
            nn.ReLU(),

            nn.Linear(464, 256),
            nn.ReLU()

            nn.Linear(256, 128),
            nn.ReLU()

        )

        ##################################################
        # GraphSAGE Layers
        ##################################################

        self.conv1 = SAGEConv(
            128,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.conv3 = SAGEConv(
            hidden_dim,
            embedding_dim
        )

        ##################################################
        # Batch Normalization
        ##################################################

        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.bn3 = nn.BatchNorm1d(embedding_dim)

        ##################################################
        # Dropout
        ##################################################

        self.drop = nn.Dropout(dropout)

        ##################################################
        # Graph Embedding Head
        ##################################################

        self.graph_head = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                embedding_dim,
                embedding_dim
            )

        )

    ##################################################
    # Encode Nodes
    ##################################################

    def encode_nodes(self, data):

        robot_x = self.robot_encoder(
            data["robot"].x
        )

        goal_x = self.goal_encoder(
            data["goal"].x
        )

        obstacle_x = self.obstacle_encoder(
            data["obstacle"].x
        )

        human_x = self.human_encoder(
            data["human"].x
        )

        return (
            robot_x,
            goal_x,
            obstacle_x,
            human_x
        )

    ##################################################
    # Forward
    ##################################################

    def forward(self, data):

        ##################################################
        # Node Encoding
        ##################################################

        (
            robot_x,
            goal_x,
            obstacle_x,
            human_x
        ) = self.encode_nodes(data)

        ##################################################
        # Combine Encoded Nodes
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

        ##################################################
        # Graph Edges
        ##################################################

        edge_index = data.edge_index

        ##################################################
        # Batch
        ##################################################

        batch = data.batch

        ##################################################
        # GraphSAGE Layer 1
        ##################################################

        x = self.conv1(
            x,
            edge_index
        )

        x = self.bn1(x)

        x = F.relu(x)

        x = self.drop(x)

        ##################################################
        # GraphSAGE Layer 2
        ##################################################

        x = self.conv2(
            x,
            edge_index
        )

        x = self.bn2(x)

        x = F.relu(x)

        x = self.drop(x)

        ##################################################
        # GraphSAGE Layer 3
        ##################################################

        x = self.conv3(
            x,
            edge_index
        )

        x = self.bn3(x)

        x = F.relu(x)

        ##################################################
        # Global Mean Pooling
        ##################################################

        graph_embedding = global_mean_pool(
            x,
            batch
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

        return graph_embedding

    ##################################################
    # Load Pretrained GraphSAGE
    ##################################################

    def load_pretrained(
        self,
        device="cpu"
    ):

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
            "\nPretrained GraphSAGE loaded successfully."
        )
