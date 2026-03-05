import numpy as np

from telescope_gym.state_setters import StateSetter
from telescope_gym.state_containers import ActuatorState

class LoadStateSetter(StateSetter):
    def __init__(
        self,
        act_shape,
        init_states_array   # Assumed to be of nparray shape (n_states,)+(act_shape)
    ):
        self.act_shape = act_shape
        self.init_states_array = init_states_array

        self.n_states = init_states_array.shape[0]

        # If the environment provides itʻs own seeded random number generator
        # use that, otherwise, use unseeded numpy
        self.np_random = None

    def build_state(self):
        return ActuatorState(act_shape=self.act_shape)

    def reset(self, act_state):
        rand = np.random if self.np_random is None else self.np_random
        sel_state = rand.integers(self.n_states)
        act_state.acts = self.init_states_array[sel_state]