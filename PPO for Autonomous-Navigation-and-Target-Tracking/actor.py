#!/usr/bin/env python3

##################################################
# actor.py
##################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal


##################################################
# Bayesian Linear Layer
##################################################

class BayesianLinear(nn.Module):

    def __init__(

            self,

            in_features,

            out_features

    ):

        super(BayesianLinear, self).__init__()

        ##################################################
        # Dimensions
        ##################################################

        self.in_features = in_features
        self.out_features = out_features

        ##################################################
        # Mean Parameters
        ##################################################

        self.weight_mu = nn.Parameter(

            torch.empty(

                out_features,

                in_features

            )

        )

        self.bias_mu = nn.Parameter(

            torch.empty(

                out_features

            )

        )

        ##################################################
        # Rho Parameters
        ##################################################

        self.weight_rho = nn.Parameter(

            torch.empty(

                out_features,

                in_features

            )

        )

        self.bias_rho = nn.Parameter(

            torch.empty(

                out_features

            )

        )

        self.reset_parameters()

    ##################################################
    # Initialization
    ##################################################

    def reset_parameters(self):

        nn.init.xavier_uniform_(

            self.weight_mu

        )

        nn.init.zeros_(

            self.bias_mu

        )

        self.weight_rho.data.fill_(

            -3.0

        )

        self.bias_rho.data.fill_(

            -3.0

        )

    ##################################################
    # Forward
    ##################################################

    def forward(self, x):

        weight_sigma = torch.log1p(

            torch.exp(

                self.weight_rho

            )

        )

        bias_sigma = torch.log1p(

            torch.exp(

                self.bias_rho

            )

        )

        epsilon_w = torch.randn_like(

            self.weight_mu

        )

        epsilon_b = torch.randn_like(

            self.bias_mu

        )

        weight = self.weight_mu + weight_sigma * epsilon_w

        bias = self.bias_mu + bias_sigma * epsilon_b

        return F.linear(

            x,

            weight,

            bias

        )


##################################################
# Bayesian PPO Actor
##################################################

class BayesianActor(nn.Module):

    def __init__(

            self,

            graph_dim=128,

            social_dim=2,

            hidden_dim=256,

            action_dim=2,

            N=20

    ):

        super(BayesianActor, self).__init__()

        ##################################################
        # Monte Carlo Forward Passes
        ##################################################

        self.N = N

        self.action_dim = action_dim

        ##################################################
        # Input Dimension
        ##################################################

        input_dim = graph_dim + social_dim

        ##################################################
        # Bayesian Layers
        ##################################################

        self.fc1 = BayesianLinear(

            input_dim,

            hidden_dim

        )

        self.fc2 = BayesianLinear(

            hidden_dim,

            hidden_dim

        )

        ##################################################
        # Mean Head
        ##################################################

        self.mu_head = BayesianLinear(

            hidden_dim,

            action_dim

        )

        ##################################################
        # Std Head
        ##################################################

        self.sigma_head = BayesianLinear(

            hidden_dim,

            action_dim

        )

    ##################################################
    # One Bayesian Forward Pass
    ##################################################

    def single_forward(

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
        # Mean Prediction
        ##################################################

        mu = self.mu_head(x)

        ##################################################
        # Standard Deviation Prediction
        ##################################################

        sigma = self.sigma_head(x)

        sigma = F.softplus(

            sigma

        ) + 1e-6

        return mu, sigma

    ##################################################
    # Monte Carlo Bayesian Inference
    ##################################################

    def forward(

            self,

            graph_embedding,

            social_features

    ):

        ##################################################
        # Containers
        ##################################################

        mu_predictions = []

        sigma_predictions = []

        ##################################################
        # N Forward Passes
        ##################################################

        for _ in range(self.N):

            mu_i, sigma_i = self.single_forward(

                graph_embedding,

                social_features

            )

            mu_predictions.append(

                mu_i

            )

            sigma_predictions.append(

                sigma_i

            )

        ##################################################
        # Stack Predictions
        ##################################################

        mu_predictions = torch.stack(

            mu_predictions,

            dim=0

        )

        sigma_predictions = torch.stack(

            sigma_predictions,

            dim=0

        )

        ##################################################
        # Final Mean
        ##################################################

        mu_final = torch.mean(

            mu_predictions,

            dim=0

        )

        ##################################################
        # Aleatoric Variance
        ##################################################

        aleatoric_variance = torch.mean(

            sigma_predictions ** 2,

            dim=0

        )

        ##################################################
        # Epistemic Variance
        ##################################################

        epistemic_variance = torch.mean(

            (

                mu_predictions -

                mu_final.unsqueeze(0)

            ) ** 2,

            dim=0

        )

        ##################################################
        # Predictive Variance
        ##################################################

        predictive_variance = (

            aleatoric_variance +

            epistemic_variance

        )

        sigma_final = torch.sqrt(

            predictive_variance +

            1e-8

        )

        ##################################################
        # CTAL Uncertainty Metric
        ##################################################

        U_epi = torch.mean(

            epistemic_variance,

            dim=1,

            keepdim=True

        )

        ##################################################
        # Action Distribution
        ##################################################

        distribution = Normal(

            mu_final,

            sigma_final

        )

        action = distribution.rsample()

        log_prob = distribution.log_prob(

            action

        ).sum(

            dim=1,

            keepdim=True

        )

        entropy = distribution.entropy().sum(

            dim=1,

            keepdim=True

        )

        ##################################################
        # Return
        ##################################################

        return (

            action,

            log_prob,

            entropy,

            mu_final,

            sigma_final,

            predictive_variance,

            U_epi

        )
