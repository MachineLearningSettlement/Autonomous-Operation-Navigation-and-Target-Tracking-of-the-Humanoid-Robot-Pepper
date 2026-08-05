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

##################################################
# GraphSAGE Network
##################################################

class GraphSAGE(nn.Module):

    ##################################################
    # Initialization
    ##################################################

    def __init__(
            self,
            input_dim=6,
            hidden_dim=128,
            embedding_dim=128,
            dropout=0.20):

        super(GraphSAGE, self).__init__()

        ##################################################
        # Parameters
        ##################################################

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        ##################################################
        # GraphSAGE Layers
        ##################################################

        self.conv1 = SAGEConv(
            input_dim,
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

        self.bn1 = nn.BatchNorm1d(
            hidden_dim
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_dim
        )

        self.bn3 = nn.BatchNorm1d(
            embedding_dim
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
                embedding_dim,
                embedding_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embedding_dim,
                embedding_dim
            )

        )

    ##################################################
    # Forward
    ##################################################

    def forward(self, data):

        ##################################################
        # Inputs
        ##################################################

        x = data.x

        edge_index = data.edge_index

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
        # Graph Head
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
