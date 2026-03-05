from abc import ABC, abstractmethod

class Reward(ABC):
    @abstractmethod
    def reset(self, initial_state):
        """
        Function to be called each time the environment is reset. This is meant to enable users to design stateful reward
        functions that maintain information about the game throughout an episode to determine a reward.

        :param initial_state: The initial state of the reset environment.
        """
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, current_state):
        """
        Function to compute the reward of the current timestep
.
        :param current_state: The current State
 
        :return: Current computed reward
        """
        raise NotImplementedError

    def get_final_reward(self, current_state):
        """
        Function to compute the reward the final step of an episode. This will be called only once, when
        it is determined that the current state is a terminal one. This may be useful for sparse reward signals that only
        produce a value at the final step of an environment. By default, the regular get_reward is used.

        :param current_state: The current State

        :return: A reward for the current state
        """
        return self.get_reward(current_state)