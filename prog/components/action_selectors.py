"""Implements action selection strategies."""

import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():
    """Action selection from a multinomial distribution.
    
    Attributes:
        args: experiment arguments.
        schedule: exploration schedule.
        epsilon: degree of exploration.
        test_greedy: whether to test with greedy policy or not.
    """

    def __init__(self, args):
        """Initializes the action selector."""
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start,
                                              args.epsilon_finish,
                                              args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """Selects the action.
        
        Args:
            agent_inputs: input observations for the agent.
            avail_actions: actions available for selection.
            t_env: timestep in environment.
            test_mode: whether testing or not.
        """
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():
    """Implements the epsilon greedy selection strategy.
    
    Attributes:
        args: experiment arguments.
        schedule: exploration schedule.
        epsilon: degree of exploration.
        test_greedy: whether to test with greedy policy or not.
    """

    def __init__(self, args):
        """Initializes the action selection object."""
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start,
                                              args.epsilon_finish,
                                              args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """Selects the action.
        
        Args:
            agent_inputs: input observations for the agent.
            avail_actions: actions available for selection.
            t_env: timestep in environment.
            test_mode: whether testing or not.
        """

        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            self.epsilon = 0.0

        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions +
        (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
