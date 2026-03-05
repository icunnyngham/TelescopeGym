from abc import ABC, abstractmethod

class StateContainer(ABC):
    """
    Class that contains the current environment state

    This sets out the bare minimum structure that all downstream functions expect

    For instance, a setup utilizing obs/actuator history would store the current state as well as the history to be used

    Notes: 
     - This should be set by StateSetter
     - The ObsBuidler may add auxilery information used downstream by other function (e.g. Rewards)
     - Expectation is that ObsBuilder.reset() is always called after StateSettter.reset()
    """
    def __init__(self):
        pass