import numpy as np
import functools as ft
import os

clear = lambda: os.system('clear')

nx, ny = 32, 32
iter = 3000

mesh = np.ones((ny, nx))
mesh[1:ny-1, 1:nx-1] = 0
mesh[0,:]=2

rho_0=1
u_0=0.1

Re=1000
viscosity=(ny-1)*u_0/Re
tau=(6*viscosity+1)/2

C, E, S, W, N, NE, SE, SW, NW = range(9)
#  % directions are indiced as follows:
#  % 8 4 5
#  % 3 0 1
#  % 7 2 6

f = np.zeros((ny*nx, 9))        # distribution function values for each cell
feq = np.zeros((ny*nx, 9))      # equilibrium distribution function value
rho = np.zeros((ny*nx))      # macroscopic density
ux = np.zeros((ny*nx))       # macroscopic velocity in direction x
uy = np.zeros((ny*nx))       # macroscopic velocity in direction y
usqr = np.zeros((ny*nx))     # helper variable

# begin initial values
f[:, C] = rho_0*4/9.
f[:, [E, S, W, N]] = rho_0/9
f[:,[NE, SE, SW, NW]] = rho_0/36
# end initial values

FL = [d[0][0] * ny + d[0][1] for d in filter(lambda d: d[1]==0, np.ndenumerate(mesh))]    # Fluid cells
WALL = [d[0][0] * ny + d[0][1] for d in filter(lambda d: d[1]==1, np.ndenumerate(mesh))]    # Wall cells
DR = [d[0][0] * ny + d[0][1] for d in filter(lambda d: d[1]==2, np.ndenumerate(mesh))]    # Driving cells

print("Starting compute loop... ")
for i in range(iter):
	# print(f"iter {i}: ", end="")
	# Begin collision step =====================================================

	assert not np.isnan(rho).any(), "NaN detected in rho!"
	assert not np.isnan(ux).any(), "NaN detected in ux!"
	assert not np.isnan(uy).any(), "NaN detected in uy!"
	assert not np.isnan(f).any(), "NaN detected in f!"
	assert np.all(usqr < 1e3), f"usqr too large: max(usqr) = {np.max(usqr)}"

	# begin distribution function value transformation to macroscopic values
	rho[:] = [sum(f_i) for f_i in f]                                                # macroscopic density
	ux[:] = (f[:, E] - f[:, W] + f[:, NE] + f[:, SE] - f[:, SW] - f[:, NW]) / rho   # x velocity
	uy[:] = (f[:, N] - f[:, S] + f[:, NE] + f[:, NW] - f[:, SE] - f[:, SW]) / rho   # y velocity
	# end distribution function value transformation to macroscopic values   

	ux[DR] = u_0    # set x velocity for driving cells
	uy[DR] = 0		# set x velocity for driving cells
	usqr[:] = ux * ux + uy * uy	# calculate helper variable value

	# begin equilibrium distribution function value calculation
	feq[:, C] = (4/9) * rho * (1 - 1.5 * usqr)
	feq[:, E] = (1/9) * rho * (1 + 3 * ux + 4.5 * (ux * ux) - 1.5 * usqr)
	feq[:, S] = (1/9) * rho * (1 - 3 * uy + 4.5 * (uy * uy) - 1.5 * usqr)
	feq[:, W] = (1/9) * rho * (1 - 3 * ux + 4.5 * (ux * ux) - 1.5 * usqr)
	feq[:, N] = (1/9) * rho * (1 + 3 * uy + 4.5 * (uy * uy) - 1.5 * usqr)
	feq[:, NE] = (1/36) * rho * (1 + 3 * (ux + uy) + 4.5 * (ux + uy) ** 2 - 1.5 * usqr)
	feq[:, SE] = (1/36) * rho * (1 + 3 * (ux - uy) + 4.5 * (ux - uy) ** 2 - 1.5 * usqr)
	feq[:, SW] = (1/36) * rho * (1 + 3 * (-ux - uy) + 4.5 * (-ux - uy) ** 2 - 1.5 * usqr)
	feq[:, NW] = (1/36) * rho * (1 + 3 * (-ux + uy) + 4.5 * (-ux + uy) ** 2 - 1.5 * usqr)
	# end equilibrium distribution function value calculation


	# begin wall cell f calculation (bounce back)
	for d, s in zip([C, E, S, W, N, NE, SE, SW, NW], [C, W, N, E, S, SW, NW, NE, SE]):
		f[WALL, d] = f[WALL, s]
	# end wall cell f calculation (bounce back)

	# begin driving cell f calculation
	f[DR, :] = feq[DR, :]	# distribution function value = equilibrium value
	# end driving cell f calculation

	# begin fluid cell f calculation
	f[FL, :] = f[FL, :] * (1 - 1 / tau) + feq[FL, :] / tau
	# end fluid cell f calculation

	# End collision step =====================================================

	# Begin propagation step =====================================================

	f = np.reshape(f, (ny, nx, 9))

	# begin particle propagation
	f[:, 1:, E] = f[:, :-1, E]
	f[1:, :, S] = f[:-1, :, S]
	f[:, :-1, W] = f[:, 1:, W]
	f[:-1, :, N] = f[1:, :, N]
	f[:-1, 1:, NE] = f[1:, :-1, NE]
	f[1:, 1:, SE] = f[:-1, :-1, SE]
	f[1:, :-1, SW] = f[:-1, 1:, SW]
	f[:-1, :-1, NW] = f[1:, 1:, NW]
	# end particle propagation

	f = np.reshape(f, (ny * nx, 9))
	# u = np.sqrt(ux * ux + uy * uy)/u_0
	# u = np.reshape(u, (ny, nx))
	# clear()
	# print(f"Iteration {i+1}/{iter}")
	# print(u)


	# print("Done.")

import matplotlib as mpl
from matplotlib import pyplot

u = np.sqrt(ux * ux + uy * uy)/u_0
u = np.reshape(u, (ny, nx))

# make values from -5 to 5, for this example

# make a color map of fixed colors
cmap = mpl.colors.LinearSegmentedColormap.from_list('color_map', ['blue','yellow','red'], 256)
# bounds=[0, 0.33,0.66,1]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = pyplot.imshow(u, interpolation='nearest',
                    cmap = cmap) #,norm=norm)

# make a color bar
pyplot.colorbar(img,cmap=cmap) #,
                # norm=norm,boundaries=bounds,ticks=[i/10 for i in range(11)])

pyplot.show()
print("Done.")