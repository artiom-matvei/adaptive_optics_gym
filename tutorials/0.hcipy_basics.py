
#%%
from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
# %%
# Grid: it is like an array, an observation space
grid = make_uniform_grid([64,64], [2,1])

plt.plot(grid.x, grid.y, '+')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
# Field: actual observation on the grid
values2 = np.exp(1j * grid.x * 20) * np.sqrt(np.abs(np.sinc(5*grid.y)))
field2 = Field(values2, grid)

imshow_field(field2)
plt.show()

# %%
# make_*_aperture, circular_apertue: field generators, yield functions that can be evaluated on a grid
pupil_grid = make_pupil_grid(128)
aperture = make_magellan_aperture(True)

imshow_field(aperture(pupil_grid), cmap='gray')
plt.show()
# %%
# evaluate_supersampled,: evaluates the field generator on a grid using specified supersampling
pupil = evaluate_supersampled(aperture, pupil_grid, 8)
imshow_field(pupil, cmap='gray')
plt.show()