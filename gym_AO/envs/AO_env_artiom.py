import hashlib
import json
import os
import pickle
from typing import Tuple, Union
import gymnasium as gym
from hcipy import (
    Cn_squared_from_fried_parameter,
    DeformableMirror,
    FraunhoferPropagator,
    InfiniteAtmosphericLayer,
    ModeBasis,
    Wavefront,
    imshow_field,
    inverse_tikhonov,
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
        self.num_pupil_pixels = 240 # * self.oversizing_factor
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


class State:
    def __init__(self, deformable_mirror, wf_wfs_on_wfs):
        self.deformable_mirror = deformable_mirror
        self.wf_wfs_on_wfs = wf_wfs_on_wfs

    def update_state(self, deformable_mirror, wf_wfs_on_wfs):
        self.deformable_mirror = deformable_mirror
        self.wf_wfs_on_wfs = wf_wfs_on_wfs


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
        self.num_modes = 100

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
        # this can be achieved by poking each mode and measuring its influence on the WFS
        # there is a tutorial that help in achieving this, see RLAO - overleaf

        ########## 
        self.reconstruction_matrix, self.image_ref = self._compute_reconstruction_matrix()

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

        self.observation_space, self.action_space = self._define_spaces()

    def _define_spaces(self):
        """Returns tuple of (action_space, observation_space)"""
        action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_modes,),
            dtype=np.float64
        ) 

        # Calculate observation space size based on WFS mode
        if self.wfs_mode == SH_WFS:
            # For SH WFS, the observation is the slopes from the lenslet array
            # BUG: probably wrong, we don't ned SH_WFS for now
            obs_size = self.wfs.mla_grid.size
        else:  # PYRAMID_WFS
            # For Pyramid WFS, the observation is the camera image
            obs_size = self.camera.detector_grid.dims

        observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=obs_size,
            dtype=np.float64
        )

        return action_space, observation_space

    def reset(self, deformable_mirror_flat=True, seed=0):
        # probably all this will end up in the __init__ method
        # 1. create the pupil grid
        # 1.1create the aperture (field)
        # 2. create the focal grid
        # 3. create the propagator
        # 4. create the DM
        self.layer.reset()
        self.deformable_mirror.flatten() if deformable_mirror_flat is True else self.deformable_mirror.random(
            0.2 * self.telescope.wavelength_wfs
        )
        wf_wfs_after_atmos = (
            self.layer(self.telescope.wf_wfs)
            if self.atmospheric_turbulence
            else self.telescope.wf_wfs
        )

        wf_wfs_after_dm = self.deformable_mirror(wf_wfs_after_atmos)

        self.wf_wfs_on_wfs = self.get_wf_on_wfs(wf_wfs_after_dm)

        # PLB: concatenate as a vector, 
        self.camera.integrate(self.wf_wfs_on_wfs, 1)
        obs = self.camera.read_out()

        return obs, {}

    def step(self, action: Union[float, int, np.ndarray]):
        # next time step of actuator state
        assert (
            action.shape == self.deformable_mirror.actuators.shape
            if isinstance(action, np.ndarray)
            else True
        )
        # if isinstance(action, (int, float)):
        #     self.deformable_mirror.actuators[0] += 0.0000001 * action
        # else:
        self.deformable_mirror.actuators = action

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
        # BUG: we might need to use fxn .forward() from get_wf_on_wfs
        self.wf_wfs_on_wfs = self.get_wf_on_wfs(wf_wfs_after_dm)

        self.camera.integrate(self.wf_wfs_on_wfs, 1)
        obs = self.camera.read_out()

        return obs, self.reward(), False, {}

    def render(self):
        # plt.ion()
        strehl_ratio = self.reward()

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
        plt.title(f"WFS plot\nStrehl Ratio: {strehl_ratio:.2f}")
        imshow_field(image_ref, cmap="inferno")

        # elif mode == "PSF":
        plt.subplot(2, 3, 4)
        plt.title("Point Spread Function")
        wf_sci_focal_plane = self.telescope.propagator(
            self.deformable_mirror(
                self.layer(self.telescope.wf_sci)
                if self.atmospheric_turbulence
                else self.telescope.wf_sci
            )
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

        # plt.draw()
        # plt.pause(0.00001)
        fig = plt.gcf()
        return fig

    def reward(self):
        # we need to look at the wf_sci after the atmosphere and after the DM
        wf_sci_focal_plane = self.telescope.propagator(
            self.deformable_mirror(
                self.layer(self.telescope.wf_sci) 
                if self.atmospheric_turbulence 
                else self.telescope.wf_sci
            )
        )

        strehl_ratio = (
            get_strehl_from_focal(
                wf_sci_focal_plane.power,
                self.telescope.unaberrated_PSF * self.telescope.wf_sci.total_power,
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
            # dims needs to be twise the num_pupil_pixels which is currently ~240
            # let's change from 120 to 240 and see what happes
            # the resolution becomes better but the relative size is the same
            pwfs_grid = make_pupil_grid(240, 2 * self.telescope.pupil_grid_diameter)
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

    def get_reconstruction_matrix(self):
        return self.reconstruction_matrix


    def _compute_reconstruction_matrix(self):
        """Compute the reconstruction matrix for the wavefront sensor.
        Checks for a cached version first. If not found, computes and caches it.

        Returns tuple: (reconstruction_matrix, image_ref)
        """
        # Define cache directory and parameters for cache key
        cache_dir = ".cache/reconstruction_matrices"
        os.makedirs(cache_dir, exist_ok=True)

        cache_params = {
            'wfs_mode': self.wfs_mode,
            'num_modes': self.num_modes,
            'wavelength_wfs': self.telescope.wavelength_wfs,
            'telescope_diameter': self.telescope.telescope_diameter,
            'central_obscuration': self.telescope.central_obscuration,
            'spider_width': self.telescope.spider_width,
            'oversizing_factor': self.telescope.oversizing_factor,
            # Add other relevant parameters if needed
        }
        # Sort params for consistent hashing
        params_string = json.dumps(cache_params, sort_keys=True)
        cache_key = hashlib.md5(params_string.encode('utf-8')).hexdigest()
        cache_filename = os.path.join(cache_dir, f"reconstruction_{cache_key}.pkl")

        if os.path.exists(cache_filename):
            print(f"Loading reconstruction matrix from cache: {cache_filename}")
            try:
                with open(cache_filename, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Ensure loaded data is float32
                    image_ref = cached_data['image_ref'].astype(np.float32)
                    reconstruction_matrix = cached_data['reconstruction_matrix'] # Assuming this should remain float64 for math? Check usage.
                # Verify shape and type after loading
                if not isinstance(image_ref, np.ndarray) or image_ref.dtype != np.float32:
                     raise TypeError("Cached image_ref has incorrect type or structure.")
                # Add check for reconstruction_matrix if needed
                print("Successfully loaded reconstruction matrix from cache.")
                return reconstruction_matrix, image_ref
            except Exception as e:
                print(f"Warning: Failed to load or validate cache file {cache_filename}. Recomputing. Error: {e}")
                # Clear potentially corrupted loaded state
                image_ref = None
                reconstruction_matrix = None

        print("Computing reconstruction matrix...")
        try:
            wf = Wavefront(self.telescope.VLT_aperture, self.telescope.wavelength_wfs)
            wf.total_power = 1

            self.camera.integrate(self.wfs.forward(wf), 1)
            # -- Robustness Change for Reference Image --
            raw_image_ref = self.camera.read_out()
             # Clip raw values before casting to prevent potential overflow in reference
            max_float32 = np.finfo(np.float32).max
            clipped_image_ref = np.clip(raw_image_ref, -max_float32, max_float32)
            image_ref = clipped_image_ref.astype(np.float32) # Ensure float32
            image_ref_sum = np.sum(image_ref)
            if image_ref_sum > 1e-9: # Add epsilon check
                image_ref /= (image_ref_sum + 1e-9) # Add epsilon
            else:
                print("Warning: Zero sum encountered in reference WFS image during calibration.")
            # -- End Robustness Change --

            probe_amp = 0.1 * self.telescope.wavelength_wfs
            slopes = []

            wf = Wavefront(self.telescope.VLT_aperture, self.telescope.wavelength_wfs)
            wf.total_power = 1

            for ind in range(self.deformable_mirror.num_actuators):
                if ind % 10 == 0:
                    print(f"Measure response to mode {ind + 1} / {self.deformable_mirror.num_actuators}")
                slope = 0

                # Probe the phase response
                for s in [1, -1]:
                    amp = np.zeros((self.deformable_mirror.num_actuators,))
                    amp[ind] = s * probe_amp
                    self.deformable_mirror.actuators = amp.astype(np.float64) # Ensure DM gets float64 if needed

                    dm_wf = self.deformable_mirror.forward(wf)
                    wfs_wf = self.wfs.forward(dm_wf)

                    self.camera.integrate(wfs_wf, 1)
                    # -- Robustness change for probe images --
                    raw_image = self.camera.read_out()
                    clipped_image = np.clip(raw_image, -max_float32, max_float32)
                    image = clipped_image.astype(np.float32) # Get float32 image
                    image_sum = np.sum(image)
                    if image_sum > 1e-9: # Add epsilon check
                         image /= (image_sum + 1e-9) # Add epsilon
                    # -- End Robustness Change --

                    slope += s * (image - image_ref) # image and image_ref are float32

                slopes.append(slope.ravel() / (2 * probe_amp)) # slope is float32

            self.deformable_mirror.flatten()

            slopes_matrix = np.array(slopes).T # Shape (pixels, actuators)
            # Reconstruction matrix computation might require float64 for stability
            reconstruction_matrix = inverse_tikhonov(slopes_matrix.astype(np.float64), 1e-4) # Cast back for inversion?

            # Cache the results (ensure image_ref is saved as float32)
            cache_data = {
                'image_ref': image_ref, # Already float32
                'reconstruction_matrix': reconstruction_matrix # Keep as computed (likely float64)
            }
            try:
                with open(cache_filename, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"Reconstruction matrix computed and saved to cache: {cache_filename}")
            except Exception as e:
                print(f"Warning: Failed to save reconstruction matrix to cache {cache_filename}. Error: {e}")
            
            return reconstruction_matrix, image_ref

        except Exception as e:
            # Clean up potentially partial results
            image_ref = None
            reconstruction_matrix = None
            raise RuntimeError(f"Failed to compute reconstruction matrix: {str(e)}")