import numpy as np
from scipy.interpolate import RegularGridInterpolator
import hcipy
import matplotlib.pyplot as plt
try:
    from IPython.display import display, clear_output
except ImportError:
    display = clear_output = None
from matplotlib.colors import LogNorm

from telescope_gym.renderers import Renderer

class DarkHoleRender(Renderer):

    def __init__(self, 
        telescope_sampler, 
        airy_pix,
        hole_airy_div,
        samp_airy_lim,
        stat_plot_len=100,
    ):
        self.telescope_sampler = telescope_sampler
        self.airy_pix = airy_pix
        self.hole_airy_div = hole_airy_div
        self.samp_airy_lim = samp_airy_lim
        self.stat_plot_len = stat_plot_len

        self.render_modes = ["notebook", "rgb_array"]

        self.fig_size = [13, 10]
        self.dpi = 180

        self.samp_r_spread = .4
        self.samp_n = 5

    def start(self):
        """Useful if you need to reinitialize a session on the client end"""

        if self.render_mode in self.render_modes:
            self.steps = 0
            self.stat_hist = []
            self.plot_steps = []
            self.reset_steps = []

            self.aper = self.telescope_sampler.aper

            x, _ = self.telescope_sampler.sample()
            self.ref_psf = x[..., 0]

            self.res = self.ref_psf.shape[0]
            self.f_ext = f_ext = self.telescope_sampler.filter_configs[0]['focal_extent']
            self.plate_scale = f_ext/self.res
            self.airy_arcsec = self.airy_pix * self.plate_scale
            self.ang_extent = [-.5*f_ext, .5*f_ext, -.5*f_ext, .5*f_ext]
            self.im_plot_norm = LogNorm(vmax=1, vmin=1e-5)


            self.samp_pix_lim = int(self.samp_airy_lim * self.airy_pix)
            self.x_s = np.linspace(0, self.samp_pix_lim, self.samp_pix_lim+1)
            self.y_s = np.zeros(self.samp_pix_lim+1)
            self.x_airy = self.x_s/self.airy_pix

            self.pix_x = np.linspace(-self.res/2, self.res/2, self.res)
            self.pix_ys, self.pix_xs = np.meshgrid(self.pix_x, self.pix_x)
            self.pix_rs = np.sqrt(self.pix_ys**2 + self.pix_xs**2)

            self.ref_interp = RegularGridInterpolator((self.pix_x, self.pix_x), self.ref_psf)

        if self.render_mode == "notebook":
            self.fig = plt.figure(figsize=self.fig_size)

    def reset(self, initial_state):
        if self.render_mode in self.render_modes:
            self.reset_steps = [self.steps]
            self.stat_hist = [initial_state.hole_fitness]
            self.plot_steps = [self.steps]
            self.steps += 1
            
            
    def render(self, current_state):
        if self.render_mode in self.render_modes:

            psf = current_state.psf
            strehl = current_state.strehl
            fit = current_state.hole_fitness
            min_loc_pix = current_state.hole_location

            self.stat_hist += [fit]
            self.plot_steps += [self.steps]

            if len(self.stat_hist) > self.stat_plot_len:
                self.stat_hist = self.stat_hist[1:]
                self.plot_steps = self.plot_steps[1:]
            if len(self.reset_steps) > 0 and self.reset_steps[0] < self.steps - self.stat_plot_len:
                self.reset_steps = self.reset_steps[1:]

            if self.render_mode == "notebook":
                self.fig.clf()
            if self.render_mode == "rgb_array":
                self.fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)

            min_cen_pix = np.array(min_loc_pix)-int(self.res/2)
            min_loc_arcsec = self.plate_scale * min_cen_pix
            min_loc_arcsec = min_loc_arcsec[::-1]
            min_loc_arcsec[1] *= -1

            psf_interp = RegularGridInterpolator((self.pix_x, self.pix_x), psf)

            th = np.arctan2(*min_cen_pix)
            r = np.sqrt(np.sum(np.power(min_loc_arcsec, 2))) / self.airy_arcsec

            del_th = np.arctan(1 / self.hole_airy_div / r)
            del_th *= self.samp_r_spread

            ref_ints = []
            psf_ints = []
            line_plot_coords = []
            del_ths = np.linspace(0, del_th, self.samp_n)
            del_ths = list(del_ths) + list(-del_ths[:0:-1])
            for d_th in del_ths:
                cur_th = th+d_th
                xs = self.x_s*np.cos(cur_th) - self.y_s*np.sin(cur_th)
                ys = self.x_s*np.sin(cur_th) + self.y_s*np.cos(cur_th)
                interp_coords = np.stack((ys, xs)).T

                ref_ints += [ self.ref_interp(interp_coords) ]
                psf_ints += [ psf_interp(interp_coords) ]

                sp_xs, sp_ys = self.plate_scale*xs, self.plate_scale*ys
                sp_ys *= -1
                line_plot_coords += [ (sp_xs, sp_ys)]
    
            ref_int_mins, ref_int_maxes = np.min(ref_ints, axis=0), np.max(ref_ints, axis=0)
            psf_int_mins, psf_int_maxes = np.min(psf_ints, axis=0), np.max(psf_ints, axis=0)
            rel_int_mins, rel_int_maxes = np.min(np.array(psf_ints)/np.array(ref_ints), axis=0), np.max(np.array(psf_ints)/np.array(ref_ints), axis=0)

            ax = self.fig.add_subplot(2, 2, 1)
            focal_im = ax.imshow(psf, cmap='inferno', extent=self.ang_extent, norm=self.im_plot_norm) 
            circle = plt.Circle((0,0), self.airy_arcsec, color='white', fill=False, alpha=.2)
            ax.add_patch(circle)
            circle = plt.Circle((0,0), 2*self.airy_arcsec, color='white', fill=False, alpha=.2)
            ax.add_patch(circle)
            circle = plt.Circle(min_loc_arcsec, self.airy_arcsec/self.hole_airy_div, color='white', fill=False, alpha=.4)
            ax.add_patch(circle)

            for i, (d_th, (p_xs, p_ys)) in enumerate(zip(del_ths, line_plot_coords)):
                if np.abs(d_th) == np.max(del_ths) or d_th == 0:
                    ax.plot(p_xs, p_ys, color='cyan', alpha=.2)  # color='lime',

            plt.colorbar(focal_im, ax=ax)
            ax.set_ylabel('focal plane (arcsec)')
            ax.set_title(f'strehl {strehl:.03f}')

            gs = self.fig.add_gridspec(nrows=4, ncols=2, hspace=0.00)
            # ax = self.fig.add_subplot(4, 2, 2)
            ax = self.fig.add_subplot(gs[0, 1])
            ax.axvline(r, alpha=.3, color='green')
            ax.axvline(r+1/self.hole_airy_div, alpha=.3, color='green')
            ax.axvline(r-1/self.hole_airy_div, alpha=.3, color='green')

            ax.fill_between(self.x_airy, ref_int_mins, ref_int_maxes, color='red', alpha=.1)
            ax.fill_between(self.x_airy, psf_int_mins, psf_int_maxes, color='black', alpha=.1)
            ax.plot(self.x_airy, ref_ints[0], color='red', label='ref')
            ax.plot(self.x_airy, psf_ints[0], color='black', label='psf')

            ax.set_yscale('log')
            ax.grid(which='both', axis='y', alpha=.3)
            ax.legend(loc='upper right')
            ax.set_xticklabels([])


            ax = self.fig.add_subplot(4, 2, 4, sharex=ax)

            hole_rs = np.sqrt((self.pix_ys-min_cen_pix[1])**2 + (self.pix_xs-min_cen_pix[0])**2)
            hole_mask = hole_rs < self.airy_pix/self.hole_airy_div

            hole_cen_r_pix = hole_rs[hole_mask]
            hole_r_pix = self.pix_rs[hole_mask]
            hole_r_airy = hole_r_pix/self.airy_pix
            hole_ref = self.ref_psf[hole_mask]
            hole_psf = psf[hole_mask]
            ax.scatter(hole_r_airy, hole_psf/hole_ref, c=1-np.power(hole_cen_r_pix, .5), cmap='Grays', marker='.', s=128, alpha=.5)

            ax.axvline(r, alpha=.3, color='green')
            ax.axvline(r+1/self.hole_airy_div, alpha=.3, color='green')
            ax.axvline(r-1/self.hole_airy_div, alpha=.3, color='green')
            ax.axhline(1, color='red')

            ax.plot(self.x_airy, psf_ints[0]/ref_ints[0], color='black', label='psf / ref')
            ax.fill_between(self.x_airy, rel_int_mins, rel_int_maxes, color='black', alpha=.1)

            ax.set_yscale('log')
            ax.grid(which='both', axis='y', alpha=.3)
            ax.legend(loc='upper right')
            ax.set_xlabel(r'$\lambda$ / D')

            ax = self.fig.add_subplot(2, 2, 3)
            pupil_im = hcipy.imshow_field(current_state.phase_screen, mask=self.aper, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, ax=ax)
            plt.colorbar(pupil_im, ax=ax)
            ax.set_ylabel('pupil plane (m)')
            

            ax = self.fig.add_subplot(2, 2, 4)
            ax.plot(self.plot_steps, self.stat_hist, '.')
            for rs in self.reset_steps:
                ax.axvline(rs, linestyle=':')
            ax.set_xlabel('step')
            ax.set_ylabel('fitness')

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


