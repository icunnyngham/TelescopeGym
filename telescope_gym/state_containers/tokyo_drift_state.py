import numpy as np
from telescope_gym.state_containers import StateContainer

class TokyoDriftState(StateContainer):
    def __init__(self, act_shape):
        self.act_shape = act_shape
        self.acts = np.zeros(act_shape)

        # Atmos specific quanties 
        # (Leaving this in, though need to write a setter for TD+atmosphere)
        self.atmos = None     # HCIPy Atmosphere Layer (generated in StateSetter)
        self.t_atmos = 0      # Time (seconds) of the current atmosphere
        self.t_atmos_step = 0 # Time (seconds) to evolve the atmosphere each step 

        # Photon flux (if using)
        self.int_phot_flux = None

        # Expected to be set in StateSetter and ObsBuilder
        self.init_err = None
        self.crop_xy = None
        self.psf_hist = []
        self.act_hist = []
        self.y_fit = None
        self.phase_screen = None
        self.pupil_rms = None
        self.strehl = None