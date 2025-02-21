import gymnasium as gym
from hcipy import *
from matplotlib import pyplot as plt
import numpy as np


class AOEnvArtiom(gym.Env):
    metadata = {"render_modes": ["actuators", "aperture", "atmosphere", "PSF", "WFS"], "render_fps": 4}

    def __init__(self, render_mode="aperture"):

        super(AOEnvArtiom, self).__init__()

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # telescope parameters definition
        self.telescope_diameter = 8. # meter
        self.central_obscuration = 1.2 # meter
        self.central_obscuration_ratio = self.central_obscuration / self.telescope_diameter
        self.spider_width = 0.05 # meter
        self.oversizing_factor = 16 / 15

        # pupil grid definition
        self.num_pupil_pixels = 240 * self.oversizing_factor
        self.pupil_grid_diameter = self.telescope_diameter * self.oversizing_factor
        self.pupil_grid = make_pupil_grid(self.num_pupil_pixels, self.pupil_grid_diameter)
        
        # aperture definition

        self.VLT_aperture_generator = make_obstructed_circular_aperture(self.telescope_diameter,
            self.central_obscuration_ratio, num_spiders=4, spider_width=self.spider_width)

        self.VLT_aperture = evaluate_supersampled(self.VLT_aperture_generator, self.pupil_grid, 4)

        # incoming wavefront 
        self.wavelength_wfs = 0.7e-6
        self.wavelength_sci = 2.2e-6
        wf = Wavefront(self.VLT_aperture, self.wavelength_sci)
        wf.total_power = 1
        
        # focal grid definition and propagator
        spatial_resolution = self.wavelength_sci / self.telescope_diameter
        self.focal_grid = make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)

        self.propagator = FraunhoferPropagator(self.pupil_grid, self.focal_grid)

        self.unaberrated_PSF = self.propagator.forward(wf).power

        ########## SHWFS setup
        f_number = 50
        num_lenslets = 40 # 40 lenslets along one diameter
        sh_diameter = 5e-3

        magnification = sh_diameter / self.telescope_diameter
        # we add magnification to the shwfs because its spatial diameter is smaller than the telescopes and we need them to correspond in order to view the aberrations at the right place on the pupil
        self.magnifier = Magnifier(magnification=magnification) 

        self.shwfs = SquareShackHartmannWavefrontSensorOptics(
            input_grid=self.pupil_grid.scaled(magnification),
            f_number=f_number,
            num_lenslets=num_lenslets,
            pupil_diameter=sh_diameter,
        )
        self.shwfse = ShackHartmannWavefrontSensorEstimator(mla_grid=self.shwfs.mla_grid, mla_index=self.shwfs.micro_lens_array.mla_index)

        self.camera = NoiselessDetector(detector_grid=self.focal_grid)

        # defining the DM controls
        self.num_modes = 500

        dm_modes = make_disk_harmonic_basis(self.pupil_grid, num_modes=self.num_modes, D=self.telescope_diameter, bc='neumann') 
        dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], self.pupil_grid)
        self.deformable_mirror = DeformableMirror(dm_modes)

        # atmosphere parameters definition
        self.seeing = 0.6 # arcsec@500nm (convention)
        self.outer_scale = 40 # meter
        self.tau0 = 0.005  # seconds
        self.delta_t = 1e-3

        self.fried_parameter = seeing_to_fried_parameter(self.seeing)
        self.Cn_squared = Cn_squared_from_fried_parameter(self.fried_parameter, 500e-9)
        self.velocity = 0.314 * self.fried_parameter / self.tau0

        self.layer = InfiniteAtmosphericLayer(
            self.pupil_grid, self.Cn_squared, self.outer_scale, self.velocity
        )

        self.action_space = gym.spaces.Discrete(1) # TODO: this should be eventually fixed to be the actual actions provided by the DM
        self.observation_space = gym.spaces.Discrete(1) # TODO: this eventually be fixed to be the WFS observations
        # gym.spaces.Box(low=#wfs measurement
        #              high=#wfs measurement
        #                  )

    def reset(self):
        # probably all this will end up in the __init__ method
        # 1. create the pupil grid
        # 1.1create the aperture (field)
        # 2. create the focal grid
        # 3. create the propagator
        # 4. create the DM
        self.layer.reset()
        self.deformable_mirror.flatten()
        pass

    def step(self, action):
        actuator_nb = np.random.randint(0, self.num_modes)
        self.deformable_mirror.actuators[actuator_nb] = np.random.random()
        self.layer.t += self.delta_t


        return 1, 1, False, False, {}

    def reward(self):
        pass

    def render(self):
        mode = self.render_mode
        plt.ion()
        if mode == 'aperture':
            imshow_field(self.VLT_aperture, cmap='gray')
            plt.xlabel('x position(m)')
            plt.ylabel('y position(m)')
            title = "Aperture plot"
        elif mode == 'atmosphere':
            phase_screen_phase = self.layer.phase_for(self.wavelength_wfs)
            imshow_field(phase_screen_phase, cmap='RdBu')
            # plt.colorbar()
            title = ('Atmospheric Phase Screen')
        elif mode == 'WFS':
            wf = Wavefront(self.VLT_aperture, wavelength=self.wavelength_wfs)

            self.camera.integrate(self.shwfs(self.magnifier(wf)), 1)

            image_ref = self.camera.read_out()

            title = "WFS plot"
            imshow_field(image_ref, cmap='inferno')
        elif mode == "PSF":
            imshow_field(np.log10(self.unaberrated_PSF / self.unaberrated_PSF.max()), cmap='inferno', vmin=-6)
            title = ('Point Spread Function')
        elif mode == 'actuators':
            controls = np.array(self.deformable_mirror.actuators)
            plt.plot(controls, label='controls')
            title = "actuators plot"
        else:
            raise Exception("Unknown render mode")
        plt.title(f"{title} at {self.layer.t} s")
        plt.draw()
        plt.pause(0.001)
