import numpy as np
from telescope_gym.rewards import Reward

# Assumes ObsBuilder is computing pupil RMS and storing in State

class ImprovePupilRMS(Reward):
    def __init__(
        self, 
        weight=1
    ):
        self.weight = weight

    def reset(self, initial_state):
        self.prev_rms = initial_state.pupil_rms 
    
    def get_reward(self, current_state):
        cur_rms = current_state.pupil_rms 
        rms_improvement = self.prev_rms - cur_rms
        self.prev_rms = cur_rms
        return float(self.weight * rms_improvement)


class ImprovePupilRMSmodDistance(Reward):
    def __init__(
        self, 
        diff_weight=1,
        dist_weight=1,
    ):
        self.diff_weight = diff_weight
        self.dist_weight = dist_weight

    def reset(self, initial_state):
        self.prev_rms = initial_state.pupil_rms 
    
    def get_reward(self, current_state):
        cur_rms = current_state.pupil_rms 
        rms_improvement = self.prev_rms - cur_rms
        self.prev_rms = cur_rms

        log_diff_dist_rew = rms_improvement * np.clip(1-np.log(cur_rms), a_min=1, a_max=None)
        dist_rew = np.clip(1-cur_rms, a_min=0, a_max=None)
        return self.diff_weight * log_diff_dist_rew + self.dist_weight * dist_rew



class ImprovePupilRMSDistConverge(Reward):
    def __init__(
        self, 
        diff_weight=1,
        converged_weight=1,
        max_inv_dist = 100
    ):
        self.diff_weight = diff_weight
        self.converged_weight = converged_weight
        self.best_rms = 1
        self.max_inv_dist = max_inv_dist

    def reset(self, initial_state):
        self.prev_rms = initial_state.pupil_rms 
    
    def get_reward(self, current_state):
        cur_rms = current_state.pupil_rms 
        if cur_rms < self.best_rms:
            self.best_rms = cur_rms
        rms_improvement = self.prev_rms - cur_rms
        self.prev_rms = cur_rms

        log_diff_dist_rew = rms_improvement * np.clip(1-np.log(cur_rms), a_min=1, a_max=None)
        return self.diff_weight * log_diff_dist_rew

    def get_final_reward(self, current_state):
        # Assumes terminated (not truncated) only happens on convergence
        return self.converged_weight + self.get_reward(current_state)