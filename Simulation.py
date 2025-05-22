import numpy as np
import settings
import initialize
import update
import force
import misc
from tqdm import tqdm
from numba import njit, prange



# def SimulationThermostat(wT,Tname, wE, Ename, wG, Gname, everyN): 
#     # initialize shit
#     settings.init()
#     x, y, z, vx, vy, vz = initialize.InitializeAtoms()

#     # fx, fy, fz, epot = force.forceLJ(x, y, z, settings.N, settings.eps, settings.mass, settings.sig, settings.cutoff, settings.L, wE)
#     fx, fy, fz, epot = force.forceLJ_and_walls(x, y, z, settings.N, settings.eps, 
#                                                settings.mass, settings.sig, settings.cutoff,
#                                                settings.L, settings.epsWall, settings.sigWall, wE)


#     # open documents for eq run
#     if wT:
#         fileoutputeq = open(Tname+ str(everyN)+ 'eq', "w")
#         misc.WriteTrajectory3d(fileoutputeq, 0,x,y,z)

#     if wE:
#         fileenergyeq = open(Ename + str(everyN) + 'eq', 'w')
#         misc.WriteEnergy(fileenergyeq, 0, epot, update.KineticEnergy(vx, vy, vz, settings.mass), np.mean(vx**2), np.mean(vy**2), np.mean(vz**2))

    
#     # do eq run
#     for i in tqdm(range(settings.eqsteps)):
#         x, y, z, vx, vy, vz, fx, fy, fz, epot = update.VelocityVerlet(x, y, z, vx, vy, vz, fx, fy, fz, settings.L,
#                                                                        settings.eps, settings.sig, settings.cutoff,
#                                                                          settings.dt, settings.mass, settings.N, wE)
#         # save shit every n
#         if i % everyN == 0:
#             if wT:
#                 misc.WriteTrajectory3d(fileoutputeq, i,x,y,z)
#             if wE:
#                 misc.WriteEnergy(fileenergyeq, i, epot, update.KineticEnergy(vx, vy, vz, settings.mass), np.mean(vx**2), np.mean(vy**2), np.mean(vz**2))

#         # rescale
#         if i % 10 == 0:
#             vx, vy, vz = initialize.rescalevelocity(vx, vy, vz,settings.Tdesired, initialize.temperature(vx, vy, vz))

#     #open shit for prod
#     if wT:
#         fileoutputprod = open(Tname+ str(everyN)+ 'prod', "w")
#         misc.WriteTrajectory3d(fileoutputprod, 0,x,y,z)

#     if wE:
#         fileenergyprod = open(Ename + str(everyN) + 'prod', 'w')
#         misc.WriteEnergy(fileenergyprod, 0, epot, update.KineticEnergy(vx, vy, vz, settings.mass), np.mean(vx**2), np.mean(vy**2), np.mean(vz**2))


#     # do prod
#     Ngr = 0   # will be used for normalizing g_of_r
#     nbins = int(settings.L/2 / settings.dr)

#     hist = np.zeros(nbins) # type: ignore
#     for i in tqdm(range(1,settings.nsteps)):
#         x, y, z, vx, vy, vz, fx, fy, fz, epot = update.VelocityVerlet(x, y, z, vx, vy, vz, fx, fy, fz, settings.L,
#                                                                        settings.eps, settings.sig, settings.cutoff,
#                                                                          settings.dt, settings.mass, settings.N, wE)
#         # save shit
#         if i % everyN == 0:
#             if wT:
#                 misc.WriteTrajectory3d(fileoutputprod, i,x,y,z)
#             if wE:
#                 misc.WriteEnergy(fileenergyprod, i, epot, update.KineticEnergy(vx, vy, vz, settings.mass), np.mean(vx**2), np.mean(vy**2), np.mean(vz**2))
#         if i % settings.Nanalyze == 0:

#             hist = update_hist(hist, x, y, z , settings.dr, settings.N, settings.L)

#             Ngr += 1 # another position
#     g = calcg(Ngr,hist, settings.dr)
#     return g


def calcg(Ngr, hist, dr):
    r= np.linspace(0,settings.L, len(hist))
    nid = 4*np.pi *settings.rho /3 * ((r+ dr)**3-r**3) # type: ignore
    n = hist/ settings.N /Ngr   
    return n/nid



@njit
def update_hist(hist, x,y,z, dr, N, L):
    for i in prange(N-1):
        # j = i + 1
        for j in range(i+1,N):
            rijx = force.pbc(x[i], x[j], L) # calculate pbc distance
            rijy = force.pbc(y[i], y[j], L)
            rijz = force.pbc(z[i], z[j], L)
            r = np.sqrt(rijx**2 + rijy**2 + rijz**2)
            if r != 0:
                bin = int(r / dr) # find the bin
                hist[bin] += 1 # we are counting pairs
    return hist


'''Ich würde nochmal ne neue klare Simlationsfunktion schreiben weil die andere überladen ist mit funktionen die wir icht brauchen'''

def SimluationSheet4(wT,Tname, everyN):
    # initialize shit
    settings.init()
    x, y, z, vx, vy, vz = initialize.InitializeAtoms()
    fx, fy, fz, epot = force.forceLJ_and_walls(x, y, z, settings.N, settings.epsfluid, settings.mass, settings.sigfluid,
                                               settings.cutofffluid, settings.L, settings.epsWall, settings.sigWall, settings.cutoffWall, False)


    # open documents for eq run
    if wT:
        fileoutputeq = open(Tname+ str(everyN)+ 'eq', "w")
        misc.WriteTrajectory3d(fileoutputeq, 0,x,y,z) # noch anpassen an L, Lz
    
    # do eq run
    for i in tqdm(range(settings.eqsteps)):
        x, y, z, vx, vy, vz, fx, fy, fz, _ = update.VelocityVerlet(x, y, z, vx, vy, vz, fx, fy, fz, settings.L, settings.epsfluid, settings.sigfluid,
                                                                       settings.cutofffluid, settings.epsWall, settings.sigWall,
                                                                       settings.cutoffWall, settings.dt, settings.mass, settings.N, False) # we dont case about the energies for now
        # save shit every n
        if i % everyN == 0:
            if wT:
                misc.WriteTrajectory3d(fileoutputeq, i,x,y,z)

        # if i % 50:
        #     print(f'x, y, z, vx, vy, vz, fx, fy, fz = {x[0], y[0], z[0], vx[0], vy[0], vz[0], fx[0], fy[0], fz[0]}')

        # rescale
        # if i % 10 == 0:
        #     vx, vy, vz = initialize.rescalevelocity(vx, vy, vz,settings.Tdesired, initialize.temperature(vx, vy, vz))

    #open shit for prod
    if wT:
        fileoutputprod = open(Tname+ str(everyN)+ 'prod', "w")
        misc.WriteTrajectory3d(fileoutputprod, 0,x,y,z)

    for i in tqdm(range(1,settings.nsteps)):
        x, y, z, vx, vy, vz, fx, fy, fz, _ = update.VelocityVerlet(x, y, z, vx, vy, vz, fx, fy, fz, settings.L, settings.epsfluid, settings.sigfluid,
                                                                       settings.cutofffluid, settings.epsWall, settings.sigWall,
                                                                       settings.cutoffWall, settings.dt, settings.mass, settings.N, False) # we dont case about the energies for now
        # save shit
        if i % everyN == 0:
            if wT:
                misc.WriteTrajectory3d(fileoutputprod, i,x,y,z)

    


