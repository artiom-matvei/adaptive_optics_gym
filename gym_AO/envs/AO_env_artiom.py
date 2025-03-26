from typing import Union
import gymnasium as gym
from hcipy import (
    Cn_squared_from_fried_parameter,
    DeformableMirror,
    FraunhoferPropagator,
    InfiniteAtmosphericLayer,
    ModeBasis,
    Wavefront,
    imshow_field,
    make_disk_harmonic_basis,
    make_focal_grid,
    make_pupil_grid,
    make_obstructed_circular_aperture,
    evaluate_supersampled,
    Magnifier,
    SquareShackHartmannWavefrontSensorOptics,
    ShackHartmannWavefrontSensorEstimator,
    NoiselessDetector,
    PyramidWavefrontSensorOptics,
    get_strehl_from_focal,
    seeing_to_fried_parameter,
)
from matplotlib import pyplot as plt
import numpy as np

PYRAMID_WFS = "pwfs"
SH_WFS = "shwfs"


class Telescope:
    def __init__(
        self,
        telescope_diameter=8.0,
        central_obscuration=1.2,
        spider_width=0.05,
        oversizing_factor=16 / 15,
    ):
        # telescope parameters definition
        self.telescope_diameter = telescope_diameter  # meter

        self.central_obscuration = central_obscuration  # meter
        self.central_obscuration_ratio = (
            self.central_obscuration / self.telescope_diameter
        )
        self.spider_width = spider_width  # meter
        self.oversizing_factor = oversizing_factor

        # pupil grid definition
        self.num_pupil_pixels = 240 * self.oversizing_factor
        self.pupil_grid_diameter = self.telescope_diameter * self.oversizing_factor
        self.pupil_grid = make_pupil_grid(
            self.num_pupil_pixels, self.pupil_grid_diameter
        )

        # aperture definition

        self.VLT_aperture_generator = make_obstructed_circular_aperture(
            self.telescope_diameter,
            self.central_obscuration_ratio,
            num_spiders=4,
            spider_width=self.spider_width,
        )

        self.VLT_aperture = evaluate_supersampled(
            self.VLT_aperture_generator, self.pupil_grid, 4
        )

        # incoming wavefront
        self.wavelength_wfs = 0.7e-6
        self.wavelength_sci = 2.2e-6

        self.wf_wfs = Wavefront(self.VLT_aperture, self.wavelength_wfs)
        self.wf_sci = Wavefront(self.VLT_aperture, self.wavelength_sci)
        self.wf_sci.total_power = 1

        # focal grid definition and propagator
        spatial_resolution = self.wavelength_sci / self.telescope_diameter
        self.focal_grid = make_focal_grid(
            q=4, num_airy=30, spatial_resolution=spatial_resolution
        )

        self.propagator = FraunhoferPropagator(self.pupil_grid, self.focal_grid)

        self.unaberrated_PSF = self.propagator.forward(self.wf_sci).power


class AOEnvArtiom(gym.Env):
    metadata = {
        "wfs_modes": [PYRAMID_WFS, SH_WFS],
        "atmospheric_turbulence": [True, False],
        "render_modes": ["actuators", "aperture", "atmosphere", "PSF", "WFS"],
        "render_fps": 4,
    }

    def __init__(self, wfs_mode=SH_WFS, atmospheric_turbulence=True):
        super(AOEnvArtiom, self).__init__()

        assert wfs_mode in self.metadata["wfs_modes"]
        self.wfs_mode = wfs_mode

        assert atmospheric_turbulence in self.metadata["atmospheric_turbulence"]
        self.atmospheric_turbulence = atmospheric_turbulence
        self.telescope = Telescope()

        ########## This part can be probably abstracted away
        ########## SHWFS setup

        self.camera = self.get_wfs_camera()

        # defining the DM controls
        self.num_modes = 500

        dm_modes = make_disk_harmonic_basis(
            self.telescope.pupil_grid,
            num_modes=self.num_modes,
            D=self.telescope.telescope_diameter,
            bc="neumann",
        )
        dm_modes = ModeBasis(
            [mode / np.ptp(mode) for mode in dm_modes], self.telescope.pupil_grid
        )
        self.deformable_mirror = DeformableMirror(dm_modes)
        
        # reconstruction matrix
        # TODO: this should be computed from the DM modes

        # atmosphere parameters definition
        self.seeing = 0.6  # arcsec@500nm (convention)
        self.outer_scale = 40  # meter
        self.tau0 = 0.005  # seconds
        self.delta_t = 1e-3

        self.fried_parameter = seeing_to_fried_parameter(self.seeing)
        self.Cn_squared = Cn_squared_from_fried_parameter(self.fried_parameter, 500e-9)
        self.velocity = 0.314 * self.fried_parameter / self.tau0

        self.layer = InfiniteAtmosphericLayer(
            self.telescope.pupil_grid, self.Cn_squared, self.outer_scale, self.velocity
        )

        self.action_space = gym.spaces.Discrete(
            1
        )  # TODO: this should be eventually fixed to be the actual actions provided by the DM
        self.observation_space = gym.spaces.Discrete(
            1
        )  # TODO: this eventually be fixed to be the WFS observations
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
        wf_wfs_after_atmos = (
            self.layer(self.telescope.wf_wfs)
            if self.atmospheric_turbulence
            else self.telescope.wf_wfs
        )

        wf_wfs_after_dm = self.deformable_mirror(wf_wfs_after_atmos)

        self.wf_wfs_on_wfs = self.get_wf_on_wfs(wf_wfs_after_dm)
        pass

    def step(self, action: Union[float, int]):
        # next time step of actuator state
        self.deformable_mirror.actuators[0] += 0.0000001 * action

        # next time step of atmosphere state
        self.layer.t += self.delta_t

        # propagate through atmosphere and deformable mirror
        # a. for wfs wf
        wf_wfs_after_atmos = (
            self.layer(self.telescope.wf_wfs)
            if self.atmospheric_turbulence
            else self.telescope.wf_wfs
        )

        wf_wfs_after_dm = self.deformable_mirror(wf_wfs_after_atmos)
        self.wf_wfs_on_wfs = self.get_wf_on_wfs(wf_wfs_after_dm)

        # b. for sci wf

        return 1, self.reward(), False, False, {}

    def render(self):
        plt.ion()

        # if mode == 'aperture':
        plt.subplot(2, 3, 1)
        imshow_field(self.telescope.VLT_aperture, cmap="gray")
        plt.xlabel("x position(m)")
        plt.ylabel("y position(m)")
        plt.title("Aperture plot")

        # elif mode == 'atmosphere':
        plt.subplot(2, 3, 2)
        phase_screen_phase = self.layer.phase_for(self.telescope.wavelength_wfs)
        imshow_field(phase_screen_phase, cmap="RdBu")
        # plt.colorbar()
        plt.title("Atmospheric Phase Screen")

        # elif mode == 'WFS':
        plt.subplot(2, 3, 3)
        # propagate through atmosphere and deformable mirror

        self.camera.integrate(self.wf_wfs_on_wfs, 1)
        image_ref = self.camera.read_out()
        # it look slike the PWFS is normalized with like follows
        image_ref /= image_ref.sum() if self.wfs_mode == PYRAMID_WFS else 1
        plt.title("WFS plot")
        imshow_field(image_ref, cmap="inferno")

        # elif mode == "PSF":
        plt.subplot(2, 3, 4)
        plt.title("Point Spread Function")
        wf_sci_focal_plane = self.telescope.propagator(
            self.deformable_mirror(
                # self.layer(
                self.telescope.wf_sci
            )
            # )
        )
        imshow_field(
            np.log10(wf_sci_focal_plane.power / wf_sci_focal_plane.power.max()),
            cmap="inferno",
            vmin=-6,
        )

        # elif mode == 'actuators':
        plt.subplot(2, 3, 5)
        controls = np.array(self.deformable_mirror.actuators)
        plt.plot(controls, label="controls")
        title = "actuators plot"
        plt.title(f"{title} at {self.layer.t:.3f} s")

        # mode == 'DM surface'
        plt.subplot(2, 3, 6)
        plt.title("DM surface [$\\mu$m]")
        imshow_field(
            self.deformable_mirror.surface * 1e6,
            cmap="RdBu",
            vmin=-2,
            vmax=2,
            mask=self.telescope.VLT_aperture,
        )
        # plt.colorbar()

        plt.draw()
        plt.pause(0.00001)

    def reward(self):
        # we need to look at the wf_sci after the atmosphere and after the DM
        wf_sci_focal_plane = self.telescope.propagator(
            self.deformable_mirror(self.layer(self.telescope.wf_sci))
        )

        strehl_ratio = (
            get_strehl_from_focal(
                wf_sci_focal_plane.power,
                self.telescope.unaberrated_PSF * self.telescope.wf_wfs.total_power,
            )
            * 100
        )
        return strehl_ratio

    def get_wfs_camera(self):
        if self.wfs_mode == SH_WFS:
            f_number = 50

            num_lenslets = 40  # 40 lenslets along one diameter
            sh_diameter = 5e-3

            magnification = sh_diameter / self.telescope.telescope_diameter
            # we add magnification to the shwfs because its spatial diameter is smaller than the telescopes and we need them to correspond in order to view the aberrations at the right place on the pupil
            self.magnifier = Magnifier(magnification=magnification)

            self.wfs = SquareShackHartmannWavefrontSensorOptics(
                input_grid=self.telescope.pupil_grid.scaled(magnification),
                f_number=f_number,
                num_lenslets=num_lenslets,
                pupil_diameter=sh_diameter,
            )

            # it is currently not used, see tuto #3 for how and when to use it
            self.wfse = ShackHartmannWavefrontSensorEstimator(
                mla_grid=self.wfs.mla_grid,
                mla_index=self.wfs.micro_lens_array.mla_index,
            )
            camera = NoiselessDetector(detector_grid=self.telescope.focal_grid)
            return camera
        elif self.wfs_mode == PYRAMID_WFS:
            pwfs_grid = make_pupil_grid(120, 2 * self.telescope.pupil_grid_diameter)
            self.wfs = PyramidWavefrontSensorOptics(
                self.telescope.pupil_grid,
                pwfs_grid,
                separation=self.telescope.pupil_grid_diameter,
                pupil_diameter=self.telescope.telescope_diameter,
                wavelength_0=self.telescope.wavelength_wfs,
                q=3,
            )

            camera = NoiselessDetector(pwfs_grid)
            return camera
        else:
            raise ValueError(f"Incorrect wfs_mode parameter: {self.wfs_mode}")

    def get_wf_on_wfs(self, wf_wfs_after_dm):
        """
        this is supposed to be after atmosphere and deformable mirror, and right before projecting onto the wfs
        """
        if self.wfs_mode == SH_WFS:
            wf_wfs_on_wfs = self.wfs(self.magnifier(wf_wfs_after_dm))
        elif self.wfs_mode == PYRAMID_WFS:
            wf_wfs_on_wfs = self.wfs(wf_wfs_after_dm)
        else:
            raise ValueError(f"Incorrect self.wfs_mode argument: {self.wfs_mode}")
        return wf_wfs_on_wfs
