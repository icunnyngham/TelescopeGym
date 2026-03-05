from abc import ABC, abstractmethod

class ActionParser(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_action_space(self):
        """
        Function that returns the action space type. It will be called during the initialization of the environment.
        
        :return: The type of the action space
        """
        raise NotImplementedError
    
    @abstractmethod
    def parse_actions(self, actions):
        """
        Function that parses actions and returns a format the telescope simulator understand

        e.g. for predicting sign and magnitude of actuators seperately, the input might be 
        actions[:n_actuators] = sign
        actions[n_actuators:] = magnitude
        and this would return a numpy array (1, n_actuator) = sign*magnitude

        :return: Numpy array in a telescope_sim compatible format
        """
        raise NotImplementedError