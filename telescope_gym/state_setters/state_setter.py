from abc import ABC, abstractmethod

class StateSetter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build_state(self):
        @abstractmethod
        def build_state(self):
            """
            Function that builds a new StateContainer object only on environment initialization

            :returns: A State object
            """
            raise NotImplementedError
        
        @abstractmethod
        def reset(self, state):
            """
            Function that resets a StateContainer object in-place
            """
            raise NotImplementedError