import settings
import random
import math
import numpy as np
 
 
def InitializeAtoms():
    
    nx = 0
    ny = 0
    nz = 0
    n = 0
    x = np.zeros(shape=(settings.N))
    y = np.zeros(shape=(settings.N))
    z = np.zeros(shape=(settings.N))
    vx = np.zeros(shape=(settings.N))
    vy = np.zeros(shape=(settings.N))
    vz = np.zeros(shape=(settings.N))

    while nx < settings.n12:
        ny=0
        while ny < settings.n12:
            nz=0
            while nz < settings.n3:
                x0 = (nx+1/2) * settings.deltaxy
                y0 = (ny+1/2) * settings.deltaxy
                z0 = (nz+1/2) * settings.deltaz
                
                vx0 = 0.5 - random.randint(0, 1)
                vy0 = 0.5 - random.randint(0, 1)
                vz0 = 0.5 - random.randint(0, 1)
                    
                    
                x[n] = x0
                y[n] = y0
                z[n] = z0
                
                vx[n] = vx0
                vy[n] = vy0
                vz[n] = vz0
                n += 1
                    
                nz += 1
            ny +=1
        nx += 1
        
    # cancel the linear momentum
    svx = np.sum(vx)
    svy = np.sum(vy)
    svz = np.sum(vz)
    
    vx -= svx / settings.N 
    vy -= svy / settings.N 
    vz -= svz / settings.N 
    # svx = np.sum(vx)
    
    # rescale the velocity to the desired temperature
    Trandom = temperature(vx, vy, vz)
    vx, vy, vz = rescalevelocity(vx, vy, vz, settings.Tdesired, Trandom)
    
    # cancel the linear momentum
    svx = np.sum(vx)
    svy = np.sum(vy)
    svz = np.sum(vz)
    
    vx -= svx / settings.N 
    vy -= svy / settings.N 
    vz -= svz / settings.N 
    
    return x, y, z, vx, vy, vz

def temperature(vx, vy, vz):
    vsq = np.sum(np.multiply(vx, vx) + np.multiply(vy, vy) + np.multiply(vz, vz))
    return settings.mass * vsq / 3 /settings.kb / settings.N
    
    
def rescalevelocity(vx, vy, vz, T1, T2):
    fac = math.sqrt(T1 / T2)
    vx = vx * fac
    vy = vy * fac
    vz = vz * fac
    return vx, vy, vz      


if __name__ == '__main__':
    settings.init()
    #print(InitializeAtoms()[0])
    print(settings.deltaxy,settings.deltaz)
    import matplotlib.pyplot as plt
    x, y, z, _, _, _ = InitializeAtoms()
    # plt.figure(figsize=[10,20])
    plt.scatter(x,z)
    plt.xlim([0, settings.L])
    plt.ylim([0, settings.Lz])
    plt.show()
    
    
    
