import numpy as np

from telescope_gym.state_setters import StateSetter
from telescope_gym.state_containers import ActuatorState

class NormalActSetter(StateSetter):
    def __init__(
        self,
        act_shape,
        error_sigma = .1
    ):
        self.act_shape = act_shape
        self.error_sigma = error_sigma

        # If the environment provides itʻs own seeded random number generator
        # use that, otherwise, use unseeded numpy
        self.np_random = None

    def build_state(self):
        return ActuatorState(act_shape=self.act_shape)

    def reset(self, act_state):
        if self.np_random is not None:
            errs = self.np_random.normal(0, self.error_sigma, act_state.acts.shape)
        else:
            errs = np.random.normal(0, self.error_sigma, act_state.acts.shape)
            
        act_state.acts = errs