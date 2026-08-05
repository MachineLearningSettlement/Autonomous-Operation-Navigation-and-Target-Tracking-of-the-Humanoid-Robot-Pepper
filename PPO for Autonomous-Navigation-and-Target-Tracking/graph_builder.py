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

##################################################
# Build Graph
##################################################

def build_graph(

        robot_node,

        target_node,

        obstacle_nodes,

        human_nodes

):

    ##################################################
    # Nodes
    ##################################################

    node_features = []

    node_positions = []

    ##################################################
    # Robot
    ##################################################

    node_features.append(

        encode_node(

            "robot",

            robot_node

        )

    )

    node_positions.append(

        robot_node["position"]

    )

    robot_index = 0

    ##################################################
    # Goal
    ##################################################

    node_features.append(

        encode_node(

            "goal",

            target_node

        )

    )

    node_positions.append(

        target_node["position"]

    )

    goal_index = 1

    ##################################################
    # Obstacles
    ##################################################

    obstacle_indices = []

    for obstacle in obstacle_nodes:

        node_features.append(

            encode_node(

                "obstacle",

                obstacle

            )

        )

        node_positions.append(

            obstacle["position"]

        )

        obstacle_indices.append(

            len(node_features) - 1

        )

    ##################################################
    # Humans
    ##################################################

    human_indices = []

    for human in human_nodes:

        node_features.append(

            encode_node(

                "human",

                human

            )

        )

        node_positions.append(

            human["position"]

        )

        human_indices.append(

            len(node_features) - 1

        )

    ##################################################
    # Edges
    ##################################################

    edge_index = []

    edge_attr = []

    ##################################################
    # Robot <-> Goal
    ##################################################

    edge_index.append(

        [robot_index, goal_index]

    )

    edge_index.append(

        [goal_index, robot_index]

    )

    d = ((

        robot_node["position"][0] -

        target_node["position"][0]

    ) ** 2 + (

        robot_node["position"][1] -

        target_node["position"][1]

    ) ** 2) ** 0.5

    edge_attr.append([d])

    edge_attr.append([d])

    ##################################################
    # Robot <-> Obstacles
    ##################################################

    for obstacle_index in obstacle_indices:

        d = ((

            node_positions[robot_index][0] -

            node_positions[obstacle_index][0]

        ) ** 2 + (

            node_positions[robot_index][1] -

            node_positions[obstacle_index][1]

        ) ** 2) ** 0.5

        edge_index.append(

            [robot_index, obstacle_index]

        )

        edge_index.append(

            [obstacle_index, robot_index]

        )

        edge_attr.append([d])

        edge_attr.append([d])

    ##################################################
    # Robot <-> Humans
    ##################################################

    for human_index in human_indices:

        d = ((

            node_positions[robot_index][0] -

            node_positions[human_index][0]

        ) ** 2 + (

            node_positions[robot_index][1] -

            node_positions[human_index][1]

        ) ** 2) ** 0.5

        edge_index.append(

            [robot_index, human_index]

        )

        edge_index.append(

            [human_index, robot_index]

        )

        edge_attr.append([d])

        edge_attr.append([d])

    ##################################################
    # Human <-> Human
    ##################################################

    for i in range(len(human_indices)):

        for j in range(i + 1, len(human_indices)):

            h1 = human_indices[i]

            h2 = human_indices[j]

            d = ((

                node_positions[h1][0] -

                node_positions[h2][0]

            ) ** 2 + (

                node_positions[h1][1] -

                node_positions[h2][1]

            ) ** 2) ** 0.5

            edge_index.append(

                [h1, h2]

            )

            edge_index.append(

                [h2, h1]

            )

            edge_attr.append([d])

            edge_attr.append([d])

    ##################################################
    # Tensor Conversion
    ##################################################

    x = torch.tensor(

        node_features,

        dtype=torch.float

    )

    edge_index = torch.tensor(

        edge_index,

        dtype=torch.long

    ).t().contiguous()

    edge_attr = torch.tensor(

        edge_attr,

        dtype=torch.float

    )

    ##################################################
    # PyTorch Geometric Graph
    ##################################################

    graph = Data(

        x=x,

        edge_index=edge_index,

        edge_attr=edge_attr

    )

    return graph
