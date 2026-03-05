import numpy as np
from scipy.signal import fftconvolve
from telescope_gym.rewards import Reward

# Assumes ObsBuilder is computing pupil RMS and storing in State

class ImproveDarkHole(Reward):
    def __init__(
        self,
        ref_psf, 
        hole_radius_pix,
        max_loc_radius_pix=None,
        hole_location_pix=None, 
        fix_reset_location=False,
        strehl_power = 1.0
    ):
        self.ref_psf = ref_psf
        self.hole_radius_pix = hole_radius_pix
        self.max_loc_radius_pix = max_loc_radius_pix
        self.hole_location_pix = hole_location_pix
        self.fix_reset_location = fix_reset_location
        self.strehl_power = strehl_power

        self.di = di = int(np.ceil(2*hole_radius_pix + 1))
        ws = np.linspace(-di/2, di/2, di)
        xs, ys = np.meshgrid(ws, ws)
        rs = np.sqrt( xs**2 + ys**2 )

        circ_kernel = np.exp(-(rs/hole_radius_pix)**30)
        self.hole_kernel = circ_kernel.reshape((di, di))

        if hole_location_pix is None:
            self.conv_ref = fftconvolve(ref_psf, self.hole_kernel, mode='same')
        else:
            self._set_fixed_hole_location()

        res = ref_psf.shape[0]
        width=int(res/2)
        ws = np.linspace(-width, width, res)
        xs, ys = np.meshgrid(ws, ws)
        rs = np.sqrt( xs**2 + ys**2 )
        self.exclude_mask = 1e9*(rs<=hole_radius_pix).astype(np.float64)

        if max_loc_radius_pix is not None:

            self.exclude_mask[rs>max_loc_radius_pix] = 1e9

    def _set_fixed_hole_location(self):
        ym, xm = self.hole_location_pix
        dh = int(self.di/2)
        syl, syr = ym-dh, ym-dh+self.di
        sxl, sxr = xm-dh, xm-dh+self.di
        sub_psf = np.copy(self.ref_psf[syl:syr, sxl:sxr])
        sub_psf *= self.hole_kernel 
        self.ref_int_hole = np.sum(sub_psf)
        self.sub_psf_coords = (syl, syr, sxl, sxr)

    def reset(self, initial_state):
        if self.fix_reset_location:
            self.hole_location_pix = None

        self._fitness(initial_state)
        self.prev_fitness = float(initial_state.hole_fitness)

        if self.hole_location_pix is not None:
            initial_state.hole_location = self.hole_location_pix
        if self.fix_reset_location:
            self.hole_location_pix = initial_state.hole_location
            self._set_fixed_hole_location()
    
    def get_reward(self, current_state):
        self._fitness(current_state)
        cur_fitness = float(current_state.hole_fitness)
        fit_improvement = self.prev_fitness - cur_fitness
        self.prev_fitness = cur_fitness
        return fit_improvement

    def _fitness(self, state):
        if self.hole_location_pix is None:
            state.hole_fitness, state.hole_location = self._any_hole_psf_score(
                state.psf, state.strehl, min_loc=True
            )
        else:
            state.hole_fitness = self._specific_spot_score(state.psf, state.strehl)


    def _any_hole_psf_score(self, psf, strehl, min_loc=False):
        psf = fftconvolve(psf, self.hole_kernel, mode='same')
        strehl_div = np.power(strehl, self.strehl_power)
        fitness = psf/self.conv_ref/strehl_div
        if self.exclude_mask is not None:
            fitness += self.exclude_mask
        cur_min = fitness.min()
        if min_loc:
            return cur_min, np.unravel_index(np.argmin(fitness), fitness.shape)
        return cur_min

    def _specific_spot_score(self, psf, strehl):
        syl, syr, sxl, sxr = self.sub_psf_coords
        sub_psf = np.copy(psf[syl:syr, sxl:sxr])
        sub_psf *= self.hole_kernel 
        psf_int_hole = np.sum(sub_psf)
        strehl_div = np.power(strehl, self.strehl_power)
        return psf_int_hole/self.ref_int_hole/strehl_div
