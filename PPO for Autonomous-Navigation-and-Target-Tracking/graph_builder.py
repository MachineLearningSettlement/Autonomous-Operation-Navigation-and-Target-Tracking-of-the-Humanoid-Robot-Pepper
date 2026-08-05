#!/usr/bin/env python3

##################################################
# graph_builder.py
##################################################

import torch

from torch_geometric.data import Data


##################################################
# Node Feature Encoder
##################################################

def encode_node(node_type, node):

    """
    Node features:

    Robot:
    [
        x,
        y,
        theta,
        linear_velocity,
        angular_velocity,
        node_type
    ]

    Goal:
    [
        x,
        y,
        0,
        0,
        0,
        node_type
    ]

    Obstacle:
    [
        x,
        y,
        distance,
        0,
        0,
        node_type
    ]

    Human:
    [
        x,
        y,
        0,
        0,
        0,
        node_type
    ]

    Node type:
    Robot     = 0
    Goal      = 1
    Obstacle  = 2
    Human     = 3
    """

    if node_type == "robot":

        feature = [

            node["position"][0],

            node["position"][1],

            node["orientation"],

            node["linear_velocity"],

            node["angular_velocity"],

            0.0

        ]


    elif node_type == "goal":

        feature = [

            node["position"][0],

            node["position"][1],

            0.0,

            0.0,

            0.0,

            1.0

        ]


    elif node_type == "obstacle":

        feature = [

            node["position"][0],

            node["position"][1],

            node["distance"],

            0.0,

            0.0,

            2.0

        ]


    elif node_type == "human":

        feature = [

            node["position"][0],

            node["position"][1],

            0.0,

            0.0,

            0.0,

            3.0

        ]


    else:

        raise ValueError(

            "Unknown node type"

        )


    return feature
