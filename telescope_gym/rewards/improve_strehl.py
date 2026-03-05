import numpy as np
from telescope_gym.rewards import Reward

# Assumes ObsBuilder is measuring strehl and storing in state

class ImproveStrehl(Reward):
    def __init__(
        self, 
        weight=1
    ):
        self.weight = weight

    def reset(self, initial_state):
        self.prev_strehl = initial_state.strehl 
    
    def get_reward(self, current_state):
        cur_strehl = current_state.strehl 
        strehl_improvement = cur_strehl - self.prev_strehl
        self.prev_strehl = cur_strehl
        return float(self.weight * strehl_improvement)