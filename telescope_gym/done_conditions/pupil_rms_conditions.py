import numpy as np
from telescope_gym.done_conditions import DoneCondition

# Assumes ObsBuilder is computing pupil RMS and storing in State

class PupilRMSDivergedCondition(DoneCondition):
    def __init__(self, div_rms):
        self.div_rms = div_rms
    
    def reset(self, initial_state):
        pass

    def condition_met(self, current_state):
        return current_state.pupil_rms >= self.div_rms


class PupilRMSConvergedCondition(DoneCondition):
    def __init__(self, close_rms):
        self.close_rms = close_rms

    def reset(self, initial_state):
        pass

    def condition_met(self, current_state):
        return current_state.pupil_rms <= self.close_rms