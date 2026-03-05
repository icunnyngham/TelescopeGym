import numpy as np
from telescope_gym.rewards import Reward

# Assumes using ActuatorState which has .acts

class ImproveActsConverge(Reward):
    def __init__(
        self, 
        mae_weight=1, 
        converged_weight=1
    ):
        self.mae_weight = mae_weight
        self.converged_weight = converged_weight

    def _mae(self, acts):
        return np.mean(np.abs(acts))
    
    def reset(self, initial_state):
        self.prev_mae = self._mae(initial_state.acts)

    def get_reward(self, current_state):
        cur_mae = self._mae(current_state.acts)
        mae_improvement = self.prev_mae - cur_mae
        self.prev_mae = cur_mae
        return self.mae_weight * mae_improvement

    def get_final_reward(self, current_state):
        # Assumes terminated (not truncated) only happens on convergence
        return self.converged_weight + self.get_reward(current_state)
