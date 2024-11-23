import numpy as np
import functools as ft

nx, ny = 8, 8
iter = 2000

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
rho = np.zeros((ny*nx, 1))      # macroscopic density
ux = np.zeros((ny*nx, 1))       # macroscopic velocity in direction x
uy = np.zeros((ny*nx, 1))       # macroscopic velocity in direction y
usqr = np.zeros((ny*nx, 1))     # helper variable

# begin initial values
f[:, C] = rho_0*4/9.
f[:, [E, S, W, N]] = rho_0/9
f[:,[NE, SE, SW, NW]] = rho_0/36
# end initial values

FL = [d[0] for d in filter(lambda d: d[1]==0, np.ndenumerate(mesh))]    # Fluid cells
WALL = [d[0] for d in filter(lambda d: d[1]==1, np.ndenumerate(mesh))]    # Wall cells
DR = [d[0] for d in filter(lambda d: d[1]==2, np.ndenumerate(mesh))]    # Driving cells

for i in range(iter):
    pass