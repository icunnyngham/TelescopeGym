import numpy as np
import hcipy

from telescope_gym.state_setters import StateSetter
from telescope_gym.state_containers import AtmosActState

# Expects to use AtmosActState

class AtmosActSetter(StateSetter):
    def __init__(
        self,
        act_shape,
        telescope_sampler,
        r0_range,              # [r0_low, r0_high] (meters) Array containing range of r0s to uniformly sample from
        t_atmos_step,          # Time in seconds to evolve the atmosphere each step
        velocity_range = [10, 10], # [vel_low, vel_high] (meters / second) Array layer velocity to uniformly sample from
        phot_flux_range = None,    # [phot_flux_low, phot_flux_high]  Array with log10 range of photon flux
        act_error_sigma = None,    # Sigma of additional random, nomrally distributed actuator errors
        outer_scale = 30,      # (meters) outer scale
        multi = False,         # (bool) Whether to use a single turbulence layer or HCIPys multi-layer
        scintillation = False, # (bool) Whether to simulate scintilation
    ):
        self.act_shape = act_shape
        self.telescope_sampler = telescope_sampler
        self.r0_range = r0_range
        self.t_atmos_step = t_atmos_step
        self.velocity_range = velocity_range
        self.phot_flux_range = phot_flux_range
        self.act_error_sigma = act_error_sigma
        self.outer_scale = outer_scale
        self.multi = multi
        self.scintillation = scintillation

        # If the environment provides itʻs own seeded random number generator
        # use that, otherwise, use unseeded numpy
        self.np_random = None

    def build_state(self):
        return AtmosActState(act_shape=self.act_shape)

    def reset(self, current_state):

        current_state.acts = np.zeros(self.act_shape)

        current_state.t_atmos_step = self.t_atmos_step

        if self.np_random is not None:
            np_random = self.np_random
        else:
            np_random = np.random

        # Select a random r0
        r0 = np_random.uniform(*self.r0_range)
        
        # Random wind velocity
        atmos_velocity = np_random.uniform(*self.velocity_range)

        # Integrated photon flux if set 
        if self.phot_flux_range is not None:
            phot_flux_exp = np_random.uniform(*self.phot_flux_range)
            current_state.int_phot_flux = np.power(10, phot_flux_exp)

        # Additional random actuation errors if set
        if self.act_error_sigma is not None:
            current_state.acts = np_random.normal(0, self.act_error_sigma, current_state.acts.shape)
        
        ### Generate atmosphere (adapted from TelescopeSim)
        # Calculate C_n^2 from given Fried param, r0 @ 550nm 
        cn2 = hcipy.Cn_squared_from_fried_parameter(r0, 550e-9)
            
        if self.multi:
            # Multi-layer atmosphere
            layers = hcipy.make_standard_atmospheric_layers(
                self.telescope_sampler.pupil_grid, 
                self.outer_scale
            )
            for i_l in range(len(layers)):
                # Set velocity of each layer to vector of specified magnitude with random direction
                layers[i_l].velocity = self.telescope_sampler._from_mag_gen_rand_vec(self.atmos_velocity)
                
            atmos = hcipy.MultiLayerAtmosphere(
                layers, 
                scintillation=self.scintillation
            )
            atmos.Cn_squared = cn2
            
            atmos.reset()
        else:
            # Single layer atmosphere
            atmos = hcipy.InfiniteAtmosphericLayer(
                self.telescope_sampler.pupil_grid, 
                cn2, 
                self.outer_scale, 
                atmos_velocity, 
                100  # Height of single layer in meters, but may not be important for now
            )
        
        current_state.atmos = atmos
        current_state.t_atmos = 0