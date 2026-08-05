#!/usr/bin/env python3

##################################################
# ctal.py
##################################################

import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

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

            actor_lr=3e-4,

            clip=0.2,

            M=10,

            K=100

    ):

        ##################################################
        # Current Bayesian PPO Actor
        ##################################################

        self.actor = actor

        ##################################################
        # Frozen Old Policy
        ##################################################

        self.old_actor = copy.deepcopy(

            actor

        )

        ##################################################
        # PPO Optimizer
        ##################################################

        self.optimizer = optim.Adam(

            self.actor.parameters(),

            lr=actor_lr

        )

        ##################################################
        # PPO Clip
        ##################################################

        self.clip = clip

        ##################################################
        # CTAL Threshold
        ##################################################

        self.threshold = threshold

        ##################################################
        # Consecutive Episodes
        ##################################################

        self.M = M

        ##################################################
        # Counterfactual Candidates
        ##################################################

        self.K = K

        ##################################################
        # Stage
        ##################################################

        self.stage = 1

        ##################################################
        # Episode Uncertainty History
        ##################################################

        self.episode_uncertainties = []

    ##################################################
    # Stage 1 Monitor
    ##################################################

    def update_episode_uncertainty(

            self,

            episode_uncertainty

    ):

        self.episode_uncertainties.append(

            episode_uncertainty

        )

        if len(self.episode_uncertainties) > self.M:

            self.episode_uncertainties.pop(0)

        ##################################################
        # Trigger Stage 2
        ##################################################

        if (

            len(self.episode_uncertainties) == self.M

            and

            max(self.episode_uncertainties) < self.threshold

        ):

            self.stage = 2

            ##################################################
            # Synchronize Old Policy
            ##################################################

            self.old_actor.load_state_dict(

                self.actor.state_dict()

            )

            return True

        return False

    ##################################################
    # Current Stage
    ##################################################

    def get_stage(self):

        return self.stage

    ##################################################
    # Trigger Counterfactual Analysis
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
        # Old Policy Forward
        ##################################################

        with torch.no_grad():

            (

                _,

                _,

                _,

                mu_old,

                sigma_old,

                _,

                _

            ) = self.old_actor(

                graph_embedding,

                social_features

            )

        ##################################################
        # Old Policy Distribution
        ##################################################

        old_distribution = torch.distributions.Normal(

            mu_old,

            sigma_old

        )

        ##################################################
        # Sample K Counterfactual Actions
        ##################################################

        candidate_actions = []

        candidate_log_probs_old = []

        for _ in range(self.K):

            action = old_distribution.rsample()

            log_prob_old = old_distribution.log_prob(

                action

            ).sum(

                dim=1,

                keepdim=True

            )

            candidate_actions.append(

                action

            )

            candidate_log_probs_old.append(

                log_prob_old

            )

        return (

            candidate_actions,

            candidate_log_probs_old

        )

    ##################################################
    # Counterfactual PPO Update
    ##################################################

    def counterfactual_update(

            self,

            graph_embedding,

            social_features,

            candidate_actions,

            candidate_log_probs_old,

            advantages

    ):

        candidate_scores = []

        candidate_actor_states = []

        ##################################################
        # Evaluate Every Candidate
        ##################################################

        for j in range(self.K):

            ##################################################
            # Restore Current Actor
            ##################################################

            self.actor.load_state_dict(

                self.old_actor.state_dict()

            )

            ##################################################
            # New Policy Forward
            ##################################################

            (

                _,

                _,

                entropy,

                mu_new,

                sigma_new,

                _,

                _

            ) = self.actor(

                graph_embedding,

                social_features

            )

            new_distribution = torch.distributions.Normal(

                mu_new,

                sigma_new

            )

            log_prob_new = new_distribution.log_prob(

                candidate_actions[j]

            ).sum(

                dim=1,

                keepdim=True

            )

            ##################################################
            # PPO Ratio
            ##################################################

            ratio = torch.exp(

                log_prob_new -

                candidate_log_probs_old[j]

            )

            ##################################################
            # PPO Objective
            ##################################################

            surr1 = ratio * advantages

            surr2 = torch.clamp(

                ratio,

                1.0 - self.clip,

                1.0 + self.clip

            ) * advantages

            objective = torch.min(

                surr1,

                surr2

            )

            loss = -objective.mean()

            ##################################################
            # Update Actor Parameters
            ##################################################

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                self.actor.parameters(),

                0.5

            )

            self.optimizer.step()

            ##################################################
            # Recompute Exact PPO Objective
            ##################################################

            (

                _,

                _,

                _,

                mu_updated,

                sigma_updated,

                _,

                _

            ) = self.actor(

                graph_embedding,

                social_features

            )

            updated_distribution = torch.distributions.Normal(

                mu_updated,

                sigma_updated

            )

            updated_log_prob = updated_distribution.log_prob(

                candidate_actions[j]

            ).sum(

                dim=1,

                keepdim=True

            )

            updated_ratio = torch.exp(

                updated_log_prob -

                candidate_log_probs_old[j]

            )

            updated_score = torch.min(

                updated_ratio * advantages,

                torch.clamp(

                    updated_ratio,

                    1.0 - self.clip,

                    1.0 + self.clip

                ) * advantages

            )

            candidate_scores.append(

                updated_score.detach()

            )

            candidate_actor_states.append(

                copy.deepcopy(

                    self.actor.state_dict()

                )

            )

        ##################################################
        # Select Best Counterfactual
        ##################################################

        scores = torch.cat(

            candidate_scores,

            dim=0

        )

        best_index = torch.argmax(

            scores

        ).item()

        ##################################################
        # Keep Best Updated Policy
        ##################################################

        self.actor.load_state_dict(

            candidate_actor_states[best_index]

        )

        ##################################################
        # Synchronize Old Policy
        ##################################################

        self.old_actor.load_state_dict(

            self.actor.state_dict()

        )

        ##################################################
        # Return
        ##################################################

        return {

            "best_action": candidate_actions[best_index],

            "best_score": candidate_scores[best_index],

            "best_index": best_index

        }
