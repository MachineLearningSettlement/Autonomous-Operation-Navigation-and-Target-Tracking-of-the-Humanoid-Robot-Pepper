#!/usr/bin/env python3

##################################################
# ppo.py
##################################################

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from actor import BayesianActor
from critic import Critic


##################################################
# PPO Memory
##################################################

class PPOMemory:

    def __init__(self):

        self.states = []

        self.actions = []

        self.log_probs = []

        self.rewards = []

        self.values = []

        self.dones = []

        self.social_features = []

        self.uncertainties = []

    ##################################################
    # Store Transition
    ##################################################

    def store(

            self,

            state,

            action,

            log_prob,

            reward,

            value,

            done,

            social_feature,

            uncertainty

    ):

        self.states.append(state)

        self.actions.append(action)

        self.log_probs.append(log_prob)

        self.rewards.append(reward)

        self.values.append(value)

        self.dones.append(done)

        self.social_features.append(

            social_feature

        )

        self.uncertainties.append(

            uncertainty

        )

    ##################################################
    # Clear Memory
    ##################################################

    def clear(self):

        self.states.clear()

        self.actions.clear()

        self.log_probs.clear()

        self.rewards.clear()

        self.values.clear()

        self.dones.clear()

        self.social_features.clear()

        self.uncertainties.clear()


##################################################
# PPO
##################################################

class PPO:

    ##################################################
    # Initialization
    ##################################################

    def __init__(

            self,

            graph_dim=128,

            social_dim=2,

            actor_lr=3e-4,

            critic_lr=1e-3,

            gamma=0.99,

            lam=0.95,

            clip=0.2,

            entropy_coef=0.01,

            critic_coef=0.5,

            epochs=5,

            batch_size=64,

            threshold_C=0.1

    ):

        ##################################################
        # Hyperparameters
        ##################################################

        self.gamma = gamma

        self.lam = lam

        self.clip = clip

        self.entropy_coef = entropy_coef

        self.critic_coef = critic_coef

        self.epochs = epochs

        self.batch_size = batch_size

        self.C = threshold_C

        ##################################################
        # Networks
        ##################################################

        self.actor = BayesianActor(

            graph_dim,

            social_dim

        )

        self.critic = Critic(

            graph_dim,

            social_dim

        )

        ##################################################
        # Old Policy
        ##################################################

        self.old_actor = copy.deepcopy(

            self.actor

        )

        ##################################################
        # Optimizers
        ##################################################

        self.actor_optimizer = optim.Adam(

            self.actor.parameters(),

            lr=actor_lr

        )

        self.critic_optimizer = optim.Adam(

            self.critic.parameters(),

            lr=critic_lr

        )

        ##################################################
        # Memory
        ##################################################

        self.memory = PPOMemory()

    ##################################################
    # Generalized Advantage Estimation (GAE)
    ##################################################

    def compute_GAE(self, next_value):

        rewards = self.memory.rewards
        values = self.memory.values
        dones = self.memory.dones

        advantages = []

        gae = 0.0

        ##################################################
        # Bootstrap
        ##################################################

        values = values + [next_value]

        ##################################################
        # Backward Computation
        ##################################################

        for t in reversed(range(len(rewards))):

            delta = (

                rewards[t]

                +

                self.gamma

                * values[t + 1]

                * (1 - dones[t])

                -

                values[t]

            )

            gae = (

                delta

                +

                self.gamma

                * self.lam

                * (1 - dones[t])

                * gae

            )

            advantages.insert(

                0,

                gae

            )

        ##################################################
        # Tensor Conversion
        ##################################################

        advantages = torch.tensor(

            advantages,

            dtype=torch.float32

        )

        values = torch.tensor(

            values[:-1],

            dtype=torch.float32

        )

        ##################################################
        # Returns
        ##################################################

        returns = advantages + values

        ##################################################
        # Normalize Advantages
        ##################################################

        advantages = (

            advantages -

            advantages.mean()

        ) / (

            advantages.std() + 1e-8

        )

        return advantages, returns

    ##################################################
    # PPO Update
    ##################################################

    def update(self, next_value):

        ##################################################
        # Compute GAE
        ##################################################

        advantages, returns = self.compute_GAE(

            next_value

        )

        ##################################################
        # Old Data
        ##################################################

        old_actions = torch.stack(

            self.memory.actions

        )

        old_log_probs = torch.stack(

            self.memory.log_probs

        ).detach()

        old_values = torch.stack(

            self.memory.values

        ).detach()

        states = self.memory.states

        social_features = torch.stack(

            self.memory.social_features

        )

        ##################################################
        # PPO Optimization
        ##################################################

        for epoch in range(self.epochs):

            ##################################################
            # Loop Over Collected Episodes
            ##################################################

            for i in range(len(states)):

                ##################################################
                # Graph Embedding
                ##################################################

                graph_embedding = states[i]

                social = social_features[i].unsqueeze(0)

                ##################################################
                # Actor Forward
                ##################################################

                action, \
                log_prob, \
                entropy, \
                mu_final, \
                sigma_final, \
                predictive_variance, \
                U_epi = self.actor(

                    graph_embedding,

                    social

                )

                ##################################################
                # Critic Forward
                ##################################################

                value = self.critic(

                    graph_embedding,

                    social

                )

                ##################################################
                # PPO Ratio
                ##################################################

                ratio = torch.exp(

                    log_prob -

                    old_log_probs[i]

                )

                ##################################################
                # Clipped Objective
                ##################################################

                surr1 = ratio * advantages[i]

                surr2 = torch.clamp(

                    ratio,

                    1.0 - self.clip,

                    1.0 + self.clip

                ) * advantages[i]

                actor_loss = -torch.min(

                    surr1,

                    surr2

                )

                ##################################################
                # Critic Loss
                ##################################################

                critic_loss = F.mse_loss(

                    value,

                    returns[i].unsqueeze(0)

                )

                ##################################################
                # Total Loss
                ##################################################

                loss = (

                    actor_loss

                    +

                    self.critic_coef * critic_loss

                    -

                    self.entropy_coef * entropy

                )

                ##################################################
                # Optimize Actor
                ##################################################

                self.actor_optimizer.zero_grad()

                ##################################################
                # Optimize Critic
                ##################################################

                self.critic_optimizer.zero_grad()

                ##################################################
                # Backpropagation
                ##################################################

                loss.backward()

                ##################################################
                # Gradient Clipping
                ##################################################

                torch.nn.utils.clip_grad_norm_(

                    self.actor.parameters(),

                    0.5

                )

                torch.nn.utils.clip_grad_norm_(

                    self.critic.parameters(),

                    0.5

                )

                ##################################################
                # Update Parameters
                ##################################################

                self.actor_optimizer.step()

                self.critic_optimizer.step()

        ##################################################
        # Synchronize Old Policy
        ##################################################

        self.old_actor.load_state_dict(

            self.actor.state_dict()

        )

        ##################################################
        # Average Epistemic Uncertainty
        ##################################################

        uncertainties = torch.stack(

            self.memory.uncertainties

        )

        average_uncertainty = torch.mean(

            uncertainties

        )

        ##################################################
        # CTAL Trigger
        ##################################################

        ctal_trigger = False

        if average_uncertainty < self.C:

            ctal_trigger = True

        ##################################################
        # Clear Memory
        ##################################################

        self.memory.clear()

        ##################################################
        # Return Statistics
        ##################################################

        statistics = {

            "actor_loss": actor_loss.detach().cpu().item(),

            "critic_loss": critic_loss.detach().cpu().item(),

            "average_uncertainty": average_uncertainty.detach().cpu().item(),

            "ctal_trigger": ctal_trigger

        }

        return statistics

    ##################################################
    # Save Models
    ##################################################

    def save(self, directory):

        torch.save(

            self.actor.state_dict(),

            directory + "/actor.pth"

        )

        torch.save(

            self.critic.state_dict(),

            directory + "/critic.pth"

        )

    ##################################################
    # Load Models
    ##################################################

    def load(self, directory):

        self.actor.load_state_dict(

            torch.load(

                directory + "/actor.pth"

            )

        )

        self.critic.load_state_dict(

            torch.load(

                directory + "/critic.pth"

            )

        )

        self.old_actor.load_state_dict(

            self.actor.state_dict()

        )
