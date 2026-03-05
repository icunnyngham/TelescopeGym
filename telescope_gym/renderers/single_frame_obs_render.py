import numpy as np
import hcipy
import matplotlib.pyplot as plt
try:
    from IPython.display import display, clear_output
except ImportError:
    display = clear_output = None
from matplotlib.colors import LogNorm

from telescope_gym.renderers import Renderer

class SingleFrameObsRender(Renderer):

    def __init__(self, 
        telescope_sampler, 
        plot_stat="pupil_rms",
        stat_plot_len=100
    ):
        self.telescope_sampler = telescope_sampler
        self.stat_plot_len = stat_plot_len

        self.render_modes = ["notebook", "rgb_array"]

        self.available_stats = ["pupil_rms", "strehl"]
        assert plot_stat in self.available_stats
        self.plot_stat = plot_stat

        self.fig_size = [10, 8]
        self.dpi = 180

    def start(self):
        """Useful if you need to reinitialize a session on the client end"""

        if self.render_mode in self.render_modes:
            self.steps = 0
            self.stat_hist = []
            self.plot_steps = []
            self.reset_steps = []

            self.aper = self.telescope_sampler.aper

            self.f_ext = f_ext = self.telescope_sampler.filter_configs[0]['focal_extent']
            self.ang_extent = [-.5*f_ext, .5*f_ext, -.5*f_ext, .5*f_ext]
            self.im_plot_norm = LogNorm(vmax=1, vmin=1e-4)

        if self.render_mode == "notebook":
            self.fig = plt.figure(figsize=[10, 8])

    def reset(self, initial_state):
        if self.render_mode in self.render_modes:
            self.reset_steps = [self.steps]
            self.stat_hist = []
            self.plot_steps = []
            
            
    def render(self, current_state):
        if self.render_mode in self.render_modes:
            if self.plot_stat == "pupil_rms":
                self.stat_hist += [current_state.pupil_rms]
            elif self.plot_stat == "strehl":
                self.stat_hist += [current_state.strehl]
            self.plot_steps += [self.steps]

            if len(self.stat_hist) > self.stat_plot_len:
                self.stat_hist = self.stat_hist[1:]
                self.plot_steps = self.plot_steps[1:]
            if self.reset_steps[0] < self.steps - self.stat_plot_len:
                self.reset_steps = self.reset_steps[1:]

            if self.render_mode == "notebook":
                self.fig.clf()
            if self.render_mode == "rgb_array":
                self.fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)

            ax = self.fig.add_subplot(2, 2, 1)
            pupil_im = hcipy.imshow_field(current_state.phase_screen, mask=self.aper, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, ax=ax)
            plt.colorbar(pupil_im, ax=ax)
            ax.set_xlabel('pupil plane (m)')
            
            ax = self.fig.add_subplot(2, 2, 2)
            focal_im = ax.imshow(current_state.psf, cmap='inferno', extent=self.ang_extent, norm=self.im_plot_norm) # vmin=log_min, 
            # plt.title(f'{photon_flux} photons/m^2/s')
            plt.colorbar(focal_im, ax=ax)
            ax.set_xlabel('focal plane (arcsec)')
            
            ax = self.fig.add_subplot(2, 1, 2)
            ax.plot(self.plot_steps, self.stat_hist, '.')
            for rs in self.reset_steps:
                ax.axvline(rs, linestyle=':')
            ax.set_xlabel('step')
            if self.plot_stat == "pupil_rms":
                ax.set_ylabel('pupil phase RMS')
            elif self.plot_stat == "strehl":
                ax.set_ylabel('strehl')

            self.steps += 1

            if self.render_mode == "notebook":
                display(self.fig)

                clear_output(wait = True)

            if self.render_mode == "rgb_array":
                self.fig.canvas.draw()
                renderer = self.fig.canvas.get_renderer()
                w, h = int(renderer.width), int(renderer.height)
                buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
                img_arr = buf.reshape((h, w, 4))
                plt.close(self.fig)
                return img_arr[..., :3]


