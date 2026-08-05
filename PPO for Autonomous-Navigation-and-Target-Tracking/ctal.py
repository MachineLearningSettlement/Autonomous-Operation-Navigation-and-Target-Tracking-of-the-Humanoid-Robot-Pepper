#!/usr/bin/env python3

##################################################
# ctal.py
##################################################

import copy
import torch
import torch.nn.functional as F


##################################################
# Counterfactual Trajectories Active Learning
##################################################

class CTAL:

    ##################################################
    # Initialization
    ##################################################

    def __init__(
            self,
            actor,
            threshold,
            M=10,
            K=100
    ):
        ##################################################
        # Bayesian PPO Actor
        ##################################################

        self.actor = actor

        ##################################################
        # Stage 1 → Stage 2 Threshold
        ##################################################

        self.threshold = threshold

        ##################################################
        # Number of consecutive episodes
        ##################################################

        self.M = M

        ##################################################
        # Number of counterfactual actions
        ##################################################

        self.K = K

        ##################################################
        # Stage Flag
        ##################################################

        self.stage = 1

        ##################################################
        # Episode uncertainty history
        ##################################################

        self.episode_uncertainties = []

    ##################################################
    # Stage 1 Monitor
    ##################################################

    def update_episode_uncertainty(
            self,
            episode_uncertainty
    ):
        ##################################################
        # Store average uncertainty
        ##################################################

        self.episode_uncertainties.append(
            episode_uncertainty
        )

        ##################################################
        # Keep only last M episodes
        ##################################################

        if len(self.episode_uncertainties) > self.M:
            self.episode_uncertainties.pop(0)

        ##################################################
        # Stage 2 Trigger
        ##################################################

        if (
            len(self.episode_uncertainties) == self.M
            and max(self.episode_uncertainties) < self.threshold
        ):
            self.stage = 2
            return True

        return False

    ##################################################
    # Current Stage
    ##################################################

    def get_stage(self):
        return self.stage

    ##################################################
    # Should Counterfactual Analysis be Triggered ?
    ##################################################

    def trigger_counterfactual(
            self,
            state_uncertainty
    ):
        if self.stage == 1:
            return False

        return state_uncertainty > self.threshold

    ##################################################
    # Generate K Counterfactual Actions
    ##################################################

    def generate_counterfactual_actions(
            self,
            graph_embedding,
            social_features
    ):
        ##################################################
        # Bayesian Forward
        ##################################################

        with torch.no_grad():
            (
                _,
                _,
                _,
                mu_final,
                sigma_final,
                _,
                _
            ) = self.actor(
                graph_embedding,
                social_features
            )

        ##################################################
        # Old Policy Distribution
        ##################################################

        distribution = torch.distributions.Normal(
            mu_final,
            sigma_final
        )

        ##################################################
        # Sample K Candidate Actions
        ##################################################

        candidate_actions = []
        candidate_log_probs = []

        for _ in range(self.K):
            action = distribution.rsample()

            log_prob = distribution.log_prob(
                action
            ).sum(
                dim=1,
                keepdim=True
            )

            candidate_actions.append(
                action
            )

            candidate_log_probs.append(
                log_prob
            )

        return (
            candidate_actions,
            candidate_log_probs
        )

    ##################################################
    # Evaluate Counterfactual PPO Objective
    ##################################################

    def evaluate_counterfactuals(
            self,
            graph_embedding,
            social_features,
            candidate_actions,
            advantages
    ):
        scores = []

        ##################################################
        # Evaluate every candidate action
        ##################################################

        for action in candidate_actions:
            (
                _,
                _,
                _,
                mu_new,
                sigma_new,
                _,
                _
            ) = self.actor(
                graph_embedding,
                social_features
            )

            distribution_new = torch.distributions.Normal(
                mu_new,
                sigma_new
            )

            log_prob_new = distribution_new.log_prob(
                action
            ).sum(
                dim=1,
                keepdim=True
            )

            score = (
                log_prob_new *
                advantages
            )

            scores.append(
                score
            )

        return scores

    ##################################################
    # Select Best Counterfactual Action
    ##################################################

    def select_best_action(
            self,
            candidate_actions,
            candidate_scores
    ):
        ##################################################
        # Stack Scores
        ##################################################

        scores = torch.cat(
            candidate_scores,
            dim=0
        )

        ##################################################
        # Best Candidate
        ##################################################

        best_index = torch.argmax(
            scores
        ).item()

        best_action = candidate_actions[best_index]
        best_score = candidate_scores[best_index]

        return (
            best_action,
            best_score,
            best_index
        )

    ##################################################
    # Counterfactual Policy Analysis
    ##################################################

    def counterfactual_policy_analysis(
            self,
            graph_embedding,
            social_features,
            advantages
    ):
        ##################################################
        # Generate K Counterfactual Actions
        ##################################################

        (
            candidate_actions,
            candidate_log_probs
        ) = self.generate_counterfactual_actions(
            graph_embedding,
            social_features
        )

        ##################################################
        # Evaluate PPO Objective
        ##################################################

        candidate_scores = self.evaluate_counterfactuals(
            graph_embedding,
            social_features,
            candidate_actions,
            advantages
        )

        ##################################################
        # Select Best Action
        ##################################################

        (
            best_action,
            best_score,
            best_index
        ) = self.select_best_action(
            candidate_actions,
            candidate_scores
        )

        ##################################################
        # Return Best Counterfactual Action
        ##################################################

        return {
            "best_action": best_action,
            "best_score": best_score,
            "best_index": best_index,
            "candidate_actions": candidate_actions,
            "candidate_scores": candidate_scores,
            "candidate_log_probs": candidate_log_probs
        }
