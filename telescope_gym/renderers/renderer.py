from abc import ABC, abstractmethod

class Renderer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def start(self):
        """
        Function to initate the renderer if necessary (useful when running interactively)
        """
        pass

    @abstractmethod
    def reset(self, initial_state):
        """
        Function to be called each time the environment is reset. 

        :param initial_state: The initial game state of the reset environment.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, current_state):
        """
        Renders a frame for the current state

        :param state: The current StateContainer  

        :return: if render_mode is rgb_array, should return, otherwise, probably no return
        """
        raise NotImplementedError

    def close(self):
        pass