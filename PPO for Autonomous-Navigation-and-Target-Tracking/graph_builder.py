#!/usr/bin/env python3

##################################################
# graph_builder.py
##################################################

import math
import torch

from torch_geometric.data import Data


##################################################
# Node Feature Encoder
##################################################

def encode_node(node_type, node):

    ##################################################
    # Robot
    #
    # [x, y, z, orientation,
    #  linear_velocity, angular_velocity]
    #
    # 6 features
    ##################################################

    if node_type == "robot":

        return [
            node["position"][0],
            node["position"][1],
            node["position"][2],
            node["orientation"],
            node["linear_velocity"],
            node["angular_velocity"]
        ]

    ##################################################
    # Goal
    #
    # [x, y, z]
    #
    # 3 features
    ##################################################

    elif node_type == "goal":

        return [
            node["position"][0],
            node["position"][1],
            node["position"][2]
        ]

    ##################################################
    # Obstacle
    #
    # [distance, orientation]
    #
    # 2 features
    ##################################################

    elif node_type == "obstacle":

        return [
            node["distance"],
            node["orientation"]
        ]

    ##################################################
    # Human
    #
    # [x, y, z]
    # + 6 IMU
    # + 1020 MotionBERT
    #
    # 1029 features
    ##################################################

    elif node_type == "human":

        position = node["position"]
        imu = node["imu"]
        motionbert = node["motionbert"]

        if len(position) != 3:
            raise ValueError(
                "Human position must contain 3 values."
            )

        if len(imu) != 6:
            raise ValueError(
                "Human IMU must contain 6 values."
            )

        if len(motionbert) != 1020:
            raise ValueError(
                "MotionBERT must contain 1020 values."
            )

        return position + imu + motionbert

    else:

        raise ValueError(
            f"Unknown node type: {node_type}"
        )


##################################################
# 3D Euclidean Distance
##################################################

def compute_distance(position_1, position_2):

    return math.sqrt(

        (position_1[0] - position_2[0]) ** 2 +

        (position_1[1] - position_2[1]) ** 2 +

        (position_1[2] - position_2[2]) ** 2

    )


##################################################
# Build Graph
##################################################

def build_graph(

        robot_node,
        target_node,
        obstacle_nodes,
        human_nodes

):

    node_features = []

    node_types = []

    node_positions = []

    ##################################################
    # Robot
    ##################################################

    robot_index = 0

    node_features.append(
        encode_node(
            "robot",
            robot_node
        )
    )

    node_types.append("robot")

    node_positions.append(
        robot_node["position"]
    )

    ##################################################
    # Goal
    ##################################################

    goal_index = 1

    node_features.append(
        encode_node(
            "goal",
            target_node
        )
    )

    node_types.append("goal")

    node_positions.append(
        target_node["position"]
    )

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

        node_types.append("obstacle")

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

        node_types.append("human")

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

    distance = compute_distance(
        robot_node["position"],
        target_node["position"]
    )

    edge_index.extend([
        [robot_index, goal_index],
        [goal_index, robot_index]
    ])

    edge_attr.extend([
        [distance],
        [distance]
    ])

    ##################################################
    # Robot <-> Obstacles
    ##################################################

    for i, obstacle in enumerate(obstacle_nodes):

        obstacle_index = obstacle_indices[i]

        distance = compute_distance(
            robot_node["position"],
            obstacle["position"]
        )

        edge_index.extend([
            [robot_index, obstacle_index],
            [obstacle_index, robot_index]
        ])

        edge_attr.extend([
            [distance],
            [distance]
        ])

    ##################################################
    # Robot <-> Humans
    ##################################################

    for i, human in enumerate(human_nodes):

        human_index = human_indices[i]

        distance = compute_distance(
            robot_node["position"],
            human["position"]
        )

        edge_index.extend([
            [robot_index, human_index],
            [human_index, robot_index]
        ])

        edge_attr.extend([
            [distance],
            [distance]
        ])

    ##################################################
    # Human <-> Human
    ##################################################

    for i in range(len(human_indices)):

        for j in range(i + 1, len(human_indices)):

            h1 = human_indices[i]

            h2 = human_indices[j]

            distance = compute_distance(
                human_nodes[i]["position"],
                human_nodes[j]["position"]
            )

            edge_index.extend([
                [h1, h2],
                [h2, h1]
            ])

            edge_attr.extend([
                [distance],
                [distance]
            ])

    ##################################################
    # Create Graph
    ##################################################

    graph = Data()

    ##################################################
    # Raw Node Features
    #
    # These names MUST match gnn.py
    ##################################################

    graph.robot_x = torch.tensor(
        [node_features[robot_index]],
        dtype=torch.float32
    )

    graph.goal_x = torch.tensor(
        [node_features[goal_index]],
        dtype=torch.float32
    )

    if obstacle_indices:

        graph.obstacle_x = torch.tensor(
            [
                node_features[i]
                for i in obstacle_indices
            ],
            dtype=torch.float32
        )

    else:

        graph.obstacle_x = torch.empty(
            (0, 2),
            dtype=torch.float32
        )

    if human_indices:

        graph.human_x = torch.tensor(
            [
                node_features[i]
                for i in human_indices
            ],
            dtype=torch.float32
        )

    else:

        graph.human_x = torch.empty(
            (0, 1029),
            dtype=torch.float32
        )

    ##################################################
    # Edge Index
    ##################################################

    if edge_index:

        graph.edge_index = torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous()

        graph.edge_attr = torch.tensor(
            edge_attr,
            dtype=torch.float32
        )

    else:

        graph.edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        graph.edge_attr = torch.empty(
            (0, 1),
            dtype=torch.float32
        )

    ##################################################
    # Metadata
    ##################################################

    graph.node_types = node_types

    graph.num_nodes = len(node_features)

    ##################################################
    # Batch
    ##################################################

    graph.batch = torch.zeros(
        graph.num_nodes,
        dtype=torch.long
    )

    return graph
