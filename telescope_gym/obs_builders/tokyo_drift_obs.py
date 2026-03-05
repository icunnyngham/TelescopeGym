import numpy as np
from gymnasium import spaces
from telescope_gym.obs_builders import ObsBuilder

class TokyoDriftObs(ObsBuilder):
    def __init__(
        self, 
        telescope_sampler,
        n_frames=2,
        meas_strehl=True,
        random_crop_pix=None,
        obs_dtype=np.float32
    ):
        self.telescope_sampler = telescope_sampler
        self.n_frames = n_frames
        self.meas_strehl = meas_strehl
        self.random_crop_pix = random_crop_pix
        self.obs_dtype = obs_dtype

        # Probably overkill, but generate test frame to get shape
        # (Maybe we need the reference PSF later?)
        x, y = telescope_sampler.sample()
        
        # Default is channels last (need to modify TelescopeSim)
        # self.obs_shape = x.shape

        # torch = channels first >:(
        psf_shape = x.shape[:-1]
        if self.random_crop_pix is not None:
            psf_shape = (psf_shape[0]-self.random_crop_pix, psf_shape[1]-self.random_crop_pix)
        self.obs_psf_shape = (n_frames,)+psf_shape

        # actuation shape
        self.act_shape = y.shape

        # Get aperture mask for RMS calculation
        self.aper_mask = self.telescope_sampler.aper == 1

    def get_obs_space(self):
        return spaces.Dict({
            'psfs': spaces.Box(low=0, high=1, shape=self.obs_psf_shape, dtype=self.obs_dtype),
            'actuation': spaces.Box(low=-np.inf, high=np.inf, shape=((self.n_frames-1)*np.prod(self.act_shape) ,) )
        })
    
    def crop(self, psf, crop_xy):
        rcp = self.random_crop_pix
        c_x, c_y = crop_xy
        psf = psf[c_y:-(rcp-c_y)] if c_y < rcp else psf[c_y:]
        psf = psf[:, c_x:-(rcp-c_x)] if c_x < rcp else psf[:, c_x:]
        return psf
    
    def reset(self, initial_state):
        # atmos = None  # not implmented

        cur_act = np.copy(initial_state.init_err)
        for i in range(self.n_frames-1):
            x, _ = self.telescope_sampler.sample(
                cur_act,
                # atmos=atmos,
                int_phot_flux=initial_state.int_phot_flux
            ) 
            if self.random_crop_pix is not None:
                x = self.crop(x, initial_state.crop_xy)
            initial_state.psf_hist += [ x[..., 0] ]
            cur_act -= initial_state.act_hist[i]

        initial_state.acts = cur_act

    def build_obs(self, current_state):

        # Evolve the atmosphere if set
        # if self.use_atmos:
        #     current_state.t_atmos += current_state.t_atmos_step
        #     current_state.atmos.evolve_until(current_state.t_atmos)
        #     atmos = current_state.atmos
        # else:
        #     atmos = None

        # Get the current int_phot_flux if set
        int_phot_flux = current_state.int_phot_flux 

        # Generate the samples
        if self.meas_strehl:
            x, y_fit, strehls = self.telescope_sampler.sample(
                current_state.acts,
                # atmos=atmos,
                int_phot_flux=int_phot_flux,
                meas_strehl=True
            ) 
            current_state.strehl = strehls[0]
        else:
            x, y_fit = self.telescope_sampler.sample(
                current_state.acts,
                # atmos=atmos,
                int_phot_flux=int_phot_flux
            ) 
        
        if self.random_crop_pix is not None:
            x = self.crop(x, current_state.crop_xy)
        
        current_state.psf_hist += [ x[..., 0] ]
        # Trim psf_hist to only the number of frames that go into the model
        current_state.psf_hist = current_state.psf_hist[-self.n_frames:]

        # Store results in the state if needed elsewhere
        current_state.y_fit = y_fit

        # current_state.phase_screen = self.telescope_sampler.getPhaseScreen(atmos=atmos)
        current_state.phase_screen = self.telescope_sampler.getPhaseScreen()

        pupil_rad = current_state.phase_screen[self.aper_mask]
        current_state.pupil_rms = float(np.sqrt(np.mean(np.power(pupil_rad - pupil_rad.mean(), 2.0))))
        
        # Return the current obs
        # (Ugly) make channel first 
        # return current_state.psf[None, ...].astype(self.obs_dtype)
        return {
            'psfs': np.stack(current_state.psf_hist).astype(self.obs_dtype),
            'actuation': np.stack(current_state.act_hist[-(self.n_frames-1):]).flatten().astype(self.obs_dtype)
        }

