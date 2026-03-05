import numpy as np

from telescope_gym.state_setters import StateSetter
from telescope_gym.state_containers import TokyoDriftState

class TokyoDriftSetter(StateSetter):
    def __init__(
        self,
        act_shape,
        error_sigma = .1,
        n_frames = 2,
        sign_error_prob = .5,   # [0 to 1] Sign error probability for initial actuations (0 is none, .5 is even odds for error)
        gain_range = [.01, .99],
        int_phot_flux_range = None,
        random_crop_pix = None, 
        random_actuation_sigma = None
    ):
        self.act_shape = act_shape
        self.error_sigma = error_sigma
        self.n_frames = n_frames
        self.sign_error_prob = sign_error_prob
        self.gain_range = gain_range
        self.int_phot_flux_range = int_phot_flux_range
        self.random_crop_pix = random_crop_pix
        self.random_actuation_sigma = random_actuation_sigma

        # If the environment provides itʻs own seeded random number generator
        # use that, otherwise, use unseeded numpy
        self.np_random = None

    def build_state(self):
        return TokyoDriftState(act_shape=self.act_shape)

    def reset(self, act_state):
        rand = np.random if self.np_random is None else self.np_random

        act_state.psf_hist = []
        act_state.act_hist = []

        init_err = rand.normal(0, self.error_sigma, self.act_shape)
        act_state.init_err = np.copy(init_err)

        gain = rand.uniform(*self.gain_range)

        cur_err = np.copy(init_err)
        for _ in range(self.n_frames-1):
            sign_errors = rand.choice(
                [1.0, -1.0], 
                self.act_shape,
                p=[1-self.sign_error_prob, self.sign_error_prob]
            )
            messy_actuation = cur_err * gain * sign_errors
            if self.random_actuation_sigma is not None:
                messy_actuation += rand(0, self.random_actuation_sigma, self.act_shape)
            cur_err -= messy_actuation
            act_state.act_hist += [ np.copy(messy_actuation) ]

        # if self.random_crop_pix is not None:
        #     self.crop_xy = rand.randint(0, self.random_crop_pix+1, (2, ))

        if self.int_phot_flux_range is not None:
            int_phot_flux = np.random.uniform(*self.int_phot_flux_range)
            act_state.int_phot_flux = np.power(10, int_phot_flux)
        
        if self.random_crop_pix is not None:
            act_state.crop_xy = rand.integers(0, self.random_crop_pix+1, 2)
        