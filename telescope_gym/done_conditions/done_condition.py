"""
A terminal or truncated condition 

(Not making any distinction, except in constructing the gym env)
"""

from abc import ABC, abstractmethod


class DoneCondition(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def reset(self, initial_state):
        """
        Function to be called each time the environment is reset.

        :param initial_state: The initial State of the reset environment.
        """
        raise NotImplementedError

    @abstractmethod
    def condition_met(self, current_state):
        """
        Function to determine if a State is terminal/truncated. This will be called once per step, and must return either
        `True` or `False` if the current episode should be terminated at this state.

        :param current_state: The current State

        :return: Bool representing whether the current state is terminated/truncated
        """
        raise NotImplementedError