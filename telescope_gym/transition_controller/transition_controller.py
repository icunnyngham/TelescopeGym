from abc import ABC, abstractmethod

class TransitionController(ABC):
    """
    Class that controls how actions from the agent are applied to the state each step

    For instance, in training you may want to apply full corrections each step, but in demonstration use a gain or more complex loop

    Note: Applied after ActionParser and before ObsBuilder
    """
    def __init__(self):
        pass

    @abstractmethod
    def step(self, state, actions):
        """
        Takes StateContainer object and actions, and applies actions to modify state in-place
        """
        raise NotImplementedError