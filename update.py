# module containing integration sheme:
"""
all in units of g, mol, fs, ns, K
"""

import settings
import force
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def VelocityVerlet(x, y, z, vx, vy, vz, fx, fy, fz, L, epsfluid, sigmafluid, cutofffluid,
                    epswall, sigmawall, cutoffwall, dt, mass, N, wE):

    """!!! Is now adapted in for length of Box in z-direction being Lz=2L !!!"""

    fx0 = np.zeros(N)
    fy0 = np.zeros(N)
    
    #update the position at t+dt
    for i in prange(N):
        x[i] = (x[i] + vx[i] * dt + fx[i] * dt * dt * 0.5) % L
        y[i] = (y[i] + vy[i] * dt + fy[i] * dt * dt * 0.5) % L
        z[i] = (z[i] + vz[i] * dt + fz[i] * dt * dt * 0.5)     # changed this here, 
                                                               # removed pbc, wallForce will avoid z<0 or z>2L !
        
    # save the force at t
    fx0 = fx
    fy0 = fy
    fz0 = fz
    # update acceleration at t+dt
    fx, fy, fz, epot = force.forceLJ_and_walls(x, y, z, N, epsfluid, mass, sigmafluid,
                                               cutofffluid, L, epswall, sigmawall, cutoffwall, wE)

    # update the velocity
    for i in prange(N):        
        vx[i] += 0.5 * dt * (fx[i] + fx0[i])
        vy[i] += 0.5 * dt * (fy[i] + fy0[i])
        vz[i] += 0.5 * dt * (fz[i] + fz0[i])
    
    return x, y, z, vx, vy, vz, fx, fy, fz, epot

 
@njit(parallel=True)
def KineticEnergy(vx, vy, vz, mass):

# calcualte the kinetic energy in joule
    ekin = 0
    N = len(vx)
    
    for i in prange(N):
        ekin += 0.5 * mass * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
    return ekin/N
