import numpy as np
from telescope_gym.done_conditions import DoneCondition

# Assumes ObsBuilder is computing strehl and storing in state

class StrehlDivergedCondition(DoneCondition):
    def __init__(self, div_strehl):
        self.div_strehl = div_strehl
    
    def reset(self, initial_state):
        pass

    def condition_met(self, current_state):
        return current_state.strehl <= self.div_strehl


class StrehlConvergedCondition(DoneCondition):
    def __init__(self, conv_strehl):
        self.conv_strehl = conv_strehl

    def reset(self, initial_state):
        pass

    def condition_met(self, current_state):
        return current_state.strehl >= self.conv_strehl