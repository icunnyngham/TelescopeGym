from abc import ABC, abstractmethod

class ObsBuilder(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_obs_space(self):
        """
        Function that returns the observation space type. It will be called during the initialization of the environment.

        :return: The type of the observation space
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, initial_state):
        """
        Function to be called each time the environment is reset. Note that this does not need to return anything,
        the environment will call `build_obs` automatically after reset, so the initial observation for a policy will be
        constructed in the same way as every other observation.

        :param initial_state: The initial game state of the reset environment.
        """
        raise NotImplementedError

    @abstractmethod
    def build_obs(self, current_state):
        """
        Function to build observations for a policy. This is where all observations will be constructed every step and
        every reset. 

        :param state: The current StateContainer  

        :return: An observation for the current State
        """
        raise NotImplementedError