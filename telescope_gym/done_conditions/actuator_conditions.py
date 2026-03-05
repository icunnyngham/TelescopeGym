import numpy as np
from telescope_gym.done_conditions import DoneCondition

# Assume using ActuatorState which has .acts

class ActRMSDivergedCondition(DoneCondition):
    def __init__(self, div_rms):
        self.div_rms = div_rms
    
    def reset(self, initial_state):
        pass

    def condition_met(self, current_state):
        acts = current_state.acts
        rms = np.sqrt(np.mean(np.power(acts, 2)))
        return rms >= self.div_rms


class ActRMSConvergedCondition(DoneCondition):
    def __init__(self, close_rms):
        self.close_rms = close_rms

    def reset(self, initial_state):
        pass

    def condition_met(self, current_state):
        acts = current_state.acts
        rms = np.sqrt(np.mean(np.power(acts, 2)))
        return rms <= self.close_rms