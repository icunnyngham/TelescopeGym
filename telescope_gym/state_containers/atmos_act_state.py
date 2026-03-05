import numpy as np
from telescope_gym.state_containers import StateContainer

class AtmosActState(StateContainer):
    def __init__(self, act_shape):
        self.act_shape = act_shape
        self.acts = np.zeros(act_shape)

        # Atmos specific quanties
        self.atmos = None     # HCIPy Atmosphere Layer (generated in StateSetter)
        self.t_atmos = 0      # Time (seconds) of the current atmosphere
        self.t_atmos_step = 0 # Time (seconds) to evolve the atmosphere each step 

        # Photon flux (if using)
        self.int_phot_flux = None

        # Expected to be set in ObsBuilder
        self.psf = None
        self.y_fit = None
        self.phase_screen = None
        self.pupil_rms = None
        self.strehl = None