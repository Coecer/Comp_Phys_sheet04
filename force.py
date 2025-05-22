import settings
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def forceLJ(x, y, z, N, epsfluid, mass, sigfluid, epswall, sigwall, cutofffluid, cutoffwall, L, wE):
    
    fx = np.zeros(N)
    fy = np.zeros(N)
    fz = np.zeros(N)

    i = 0

    prefac = 4* epsfluid/mass
    sigfluid12 = sigfluid**12
    sigfluid6 = sigfluid**6
    epot = 0


    for i in prange(N-1):
        j = i + 1
        for j in range(i+1, N):
            rijx = pbc(x[i], x[j], L) # calculate pbc distance
            rijy = pbc(y[i], y[j], L)
            rijz = pbc(z[i], z[j], L)
            
            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            # calculate fx, fy, fz
            if r2 < cutofffluid * cutofffluid:
                rabs14 = r2**7 
                rabs8 = r2**4
                ff = prefac * (sigfluid12/rabs14 - sigfluid6/rabs8)
                fx[i] -= ff*rijx
                fy[i] -= ff*rijy
                fz[i] -= ff*rijz

                fx[j] += ff * rijx
                fy[j] += ff * rijy
                fz[j] += ff * rijz
            if wE:
                epot += 4* epsfluid *(sigfluid12/r2**6-sigfluid6/r2**3) # langfristig muss das nochmal gefixt werden idk

            
    # if wE:
    #     for i in prange(N):
    #         for j in range(N):
    #             rijx = pbc(x[i], x[j], L) # calculate pbc distance
    #             rijy = pbc(y[i], y[j], L)
    #             rijz = pbc(z[i], z[j], L)
                
    #             r2 = rijx**2 + rijy**2 + rijz**2
    #             epot += 4* epsfluid *(sigfluid12/r2**12-sigfluid6/r2**6)


            
    return fx, fy, fz, 2*epot/N # is already divided by mass --> acceleration not force ## 2* because pairs and per particle
  
@njit  
def pbc(xi, xj, L):
    
    xi = xi
    xj = xj
    
    rij = xj - xi  
    if abs(rij) > 0.5*L:
        rij = rij - np.sign(rij) * L
        
    return rij

############### LJ force and wall force ###############
def forceLJ_and_walls(x, y, z, N, epsfluid, mass, sigfluid, cutofffluid, L, epsWall, sigWall, wE):

    fx = np.zeros(N)
    fy = np.zeros(N)
    fz = np.zeros(N)
    # i = 0  not needed imo
    prefac = 4* epsfluid/mass
    sigfluid12 = sigfluid**12
    sigfluid6 = sigfluid**6
    epot = 0

    cutofffluidWall = 2.5 * sigWall  # same as cutofffluid actually
    sigfluidW3 = 3*sigWall**3
    sigfluidW9 = 9*sigWall**9

    for i in prange(N-1):
        for j in range(i+1, N):
            rijx = pbc(x[i], x[j], L) # calculate pbc distance
            rijy = pbc(y[i], y[j], L)
            rijz = z[i] - z[j]  #  will be squared so no need for abs()
            
            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            # calculate fx, fy, fz
            if r2 < cutofffluid * cutofffluid:
                rabs14 = r2**7 
                rabs8 = r2**4
                ff = prefac * (sigfluid12/rabs14 - sigfluid6/rabs8)
                fx[i] -= ff*rijx
                fy[i] -= ff*rijy
                fz[i] -= ff*rijz

                fx[j] += ff * rijx
                fy[j] += ff * rijy
                fz[j] += ff * rijz
            if wE:
                epot += 4* epsfluid *(sigfluid12/r2**6-sigfluid6/r2**3)

    epot = 2*epot/N # is already divided by mass --> acceleration not force ## 2* because pairs and per particle

    prefacW = 3*np.sqrt(3) *0.5 * epsWall
    ######### Add fz force of wall now as well ############
    for i in prange(N):
        d_left  = z[i]        # z distance to left wall at z=0
        d_right = 2*L - z[i]  # z distance to right wall at z=2L

        if d_left < cutofffluidWall and d_left < 0:  # ensure particle isnt behind wall
            fz[i] += prefacW * (sigfluidW3/d_left**4 - sigfluidW9/d_left**10)
            # no pairwise contribution for wall and particels, only part contribute
            epot += prefacW * (sigfluidW9/d_left**9 - sigfluidW3/d_left**3) 
        # use elif to skip if already left distance was smaller since
        # cutofffluidWall should always prevent that 
        elif d_right < cutofffluidWall and d_right < 0: 
            fz[i] += prefacW * (sigfluidW3/d_right**4 - sigfluidW9/d_right**10)
            epot += prefacW * (sigfluidW9/d_left**9 - sigfluidW3/d_left**3)

    
    return fx, fy, fz, epot # doubled epot contribution bc of pairs is implemented  



if __name__ == '__main__':
    settings.init()
    x = np.array([settings.sigfluid/3, settings.L-settings.sigfluid/3,])
    y = np.array([0,0])
    z = np.array([0,0])
    
    fx, fy, fz, epot = forceLJ_and_walls(x, y, z, 2, settings.epsfluid, settings.mass, settings.sigfluid, settings.cutofffluid, settings.L, settings.epsWall, settings.sigWall, False)
    print(fx)
    print(fy)
    #


    ### glaube sieht gut aus alles :)
