import numpy as np
from gymnasium import spaces
from telescope_gym.obs_builders import ObsBuilder

class SingleFrameObs(ObsBuilder):
    def __init__(
        self, 
        telescope_sampler,
        use_atmos=False,
        use_phot_flux=False,
        obs_dtype=np.float32
    ):
        self.telescope_sampler = telescope_sampler
        self.obs_dtype = obs_dtype

        # These require current_state to look like AtmosActState
        self.use_atmos = use_atmos
        self.use_phot_flux = use_phot_flux

        # Probably overkill, but generate test frame to get shape
        # (Maybe we need the reference PSF later?)
        x, _ = telescope_sampler.sample()
        
        # Default is channels last (need to modify TelescopeSim)
        # self.obs_shape = x.shape

        # torch = channels first >:(
        self.obs_shape = (1,)+x.shape[:-1]
        ####
        # self.obs_shape = (3,)+x.shape[:-1]

        # Get aperture mask for RMS calculation
        self.aper_mask = self.telescope_sampler.aper == 1

    def get_obs_space(self):
        return spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=self.obs_dtype)
    
    def reset(self, initial_state):
        pass

    def build_obs(self, current_state):

        # Evolve the atmosphere if set
        if self.use_atmos:
            current_state.t_atmos += current_state.t_atmos_step
            current_state.atmos.evolve_until(current_state.t_atmos)
            atmos = current_state.atmos
        else:
            atmos = None

        # Get the current int_phot_flux if set
        int_phot_flux = current_state.int_phot_flux if self.use_phot_flux else None

        # Generate the samples
        x, y_fit, strehls = self.telescope_sampler.sample(
            current_state.acts,
            atmos=atmos,
            int_phot_flux=int_phot_flux,
            meas_strehl=True
        ) 
        
        # Store results in the state if needed elsewhere
        current_state.psf = x[..., 0]
        current_state.y_fit = y_fit
        current_state.strehl = strehls[0]

        current_state.phase_screen = self.telescope_sampler.getPhaseScreen(atmos=atmos)

        pupil_rad = current_state.phase_screen[self.aper_mask]
        current_state.pupil_rms = float(np.sqrt(np.mean(np.power(pupil_rad - pupil_rad.mean(), 2.0))))
        
        # Return the current obs
        # (Ugly) make channel first 
        return current_state.psf[None, ...].astype(self.obs_dtype)

