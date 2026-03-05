from gymnasium import spaces
import numpy as np
from telescope_gym.action_parsers import ActionParser

class ActuatorActions(ActionParser):
    def __init__(self, n_act):
        self.n_act = n_act

    def get_action_space(self) -> spaces.Space:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_act,) )

    def parse_actions(self, actions):
        # actions = actions.reshape((-1, self.n_act))
        actions = actions.reshape((self.n_act,))

        # You could imagine needing to apply signs in FF, etc

        return actions
    

class ReshapeActuatorActions(ActionParser):
    def __init__(self, act_shape):
        self.act_shape = act_shape

    def get_action_space(self) -> spaces.Space:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(np.prod(self.act_shape),) )

    def parse_actions(self, actions):
        return actions.reshape(self.act_shape)

class MaskedActuatorActions(ActionParser):
    def __init__(self, act_shape, mask):
        self.act_shape = act_shape
        self.mask = mask

    def get_action_space(self) -> spaces.Space:
        masked_shape = np.zeros(self.act_shape)[self.mask].shape
        return spaces.Box(low=-np.inf, high=np.inf, shape=masked_shape )

    def parse_actions(self, actions):
        out_action = np.zeros(self.act_shape)
        out_action[self.mask] = actions
        return out_action