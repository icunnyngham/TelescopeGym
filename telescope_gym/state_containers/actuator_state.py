import numpy as np
from telescope_gym.state_containers import StateContainer

class ActuatorState(StateContainer):
    def __init__(self, act_shape):
        self.act_shape = act_shape
        self.acts = np.zeros(act_shape)

        # Expected to be set in ObsBuilder
        self.psf = None
        self.y_fit = None
        self.phase_screen = None
        self.pupil_rms = None
        self.strehl = None