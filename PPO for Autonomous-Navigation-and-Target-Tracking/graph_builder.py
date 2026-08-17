#!/usr/bin/env python3

##################################################
# graph_builder.py
##################################################

import math
import torch

from torch_geometric.data import HeteroData


##################################################
# Node Feature Encoder
##################################################

def encode_node(node_type, node):

    ##################################################
    # Robot
    #
    # [x, y, z,
    #  orientation,
    #  linear_velocity,
    #  angular_velocity]
    #
    # Dimension = 6
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
    # Dimension = 3
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
    # Dimension = 2
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
    # + 6 IMU values
    # + 1020 MotionBERT values
    #
    # Dimension = 1029
    ##################################################

    elif node_type == "human":

        position = node["position"]
        imu = node["imu"]
        motionbert = node["motionbert"]

        if len(position) != 3:
            raise ValueError(
                "Human position must contain exactly 3 values."
            )

        if len(imu) != 6:
            raise ValueError(
                "Human IMU must contain exactly 6 values."
            )

        if len(motionbert) != 1020:
            raise ValueError(
                "MotionBERT representation must contain exactly 1020 values."
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

        (position_1[0] - position_2[0]) ** 2

        + (position_1[1] - position_2[1]) ** 2

        + (position_1[2] - position_2[2]) ** 2

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

    ##################################################
    # Heterogeneous Graph
    ##################################################

    graph = HeteroData()

    ##################################################
    # Robot
    ##################################################

    robot_features = encode_node(
        "robot",
        robot_node
    )

    graph["robot"].x = torch.tensor(
        [robot_features],
        dtype=torch.float32
    )

    ##################################################
    # Goal
    ##################################################

    goal_features = encode_node(
        "goal",
        target_node
    )

    graph["goal"].x = torch.tensor(
        [goal_features],
        dtype=torch.float32
    )

    ##################################################
    # Obstacles
    ##################################################

    obstacle_features = [

        encode_node(
            "obstacle",
            obstacle
        )

        for obstacle in obstacle_nodes

    ]

    if obstacle_features:

        graph["obstacle"].x = torch.tensor(
            obstacle_features,
            dtype=torch.float32
        )

    else:

        graph["obstacle"].x = torch.empty(
            (0, 2),
            dtype=torch.float32
        )

    ##################################################
    # Humans
    ##################################################

    human_features = [

        encode_node(
            "human",
            human
        )

        for human in human_nodes

    ]

    if human_features:

        graph["human"].x = torch.tensor(
            human_features,
            dtype=torch.float32
        )

    else:

        graph["human"].x = torch.empty(
            (0, 1029),
            dtype=torch.float32
        )

    ##################################################
    # Global Node Indices
    ##################################################

    robot_index = 0

    goal_index = 1

    obstacle_indices = [

        2 + i

        for i in range(
            len(obstacle_nodes)
        )

    ]

    human_start = (

        2

        + len(obstacle_nodes)

    )

    human_indices = [

        human_start + i

        for i in range(
            len(human_nodes)
        )

    ]

    ##################################################
    # Edges
    ##################################################

    edge_index = []

    edge_distances = []

    ##################################################
    # Robot ↔ Goal
    ##################################################

    distance = compute_distance(
        robot_node["position"],
        target_node["position"]
    )

    edge_index.extend([
        [robot_index, goal_index],
        [goal_index, robot_index]
    ])

    edge_distances.extend([
        [distance],
        [distance]
    ])

    ##################################################
    # Robot ↔ Obstacles
    ##################################################

    for i, obstacle in enumerate(
        obstacle_nodes
    ):

        obstacle_index = obstacle_indices[i]

        distance = compute_distance(
            robot_node["position"],
            obstacle["position"]
        )

        edge_index.extend([
            [robot_index, obstacle_index],
            [obstacle_index, robot_index]
        ])

        edge_distances.extend([
            [distance],
            [distance]
        ])

    ##################################################
    # Robot ↔ Humans
    ##################################################

    for i, human in enumerate(
        human_nodes
    ):

        human_index = human_indices[i]

        distance = compute_distance(
            robot_node["position"],
            human["position"]
        )

        edge_index.extend([
            [robot_index, human_index],
            [human_index, robot_index]
        ])

        edge_distances.extend([
            [distance],
            [distance]
        ])

    ##################################################
    # Human ↔ Human
    ##################################################

    for i in range(
        len(human_indices)
    ):

        for j in range(
            i + 1,
            len(human_indices)
        ):

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

            edge_distances.extend([
                [distance],
                [distance]
            ])

    ##################################################
    # Edge Tensors
    ##################################################

    if edge_index:

        graph.edge_index = torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous()

        graph.edge_attr = torch.tensor(
            edge_distances,
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
    # Batch
    ##################################################

    total_nodes = (

        2

        + len(obstacle_nodes)

        + len(human_nodes)

    )

    graph.batch = torch.zeros(
        total_nodes,
        dtype=torch.long
    )

    ##################################################
    # Return
    ##################################################

    return graph
