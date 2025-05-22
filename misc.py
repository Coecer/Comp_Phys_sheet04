import settings
import numpy as np
import math
import sys
import time

def WriteEnergy(fileenergy, itime, epot, ekin, vx2, vy2, vz2):
    
    fileenergy.write("%i %e %e %e %e %e\n" % (itime, epot, ekin, vx2, vy2, vz2))

def WriteTrajectory3d(fileoutput, itime, x, y, z):

    fileoutput.write("ITEM: TIMESTEP \n")
    fileoutput.write("%i \n" % itime)
    fileoutput.write("ITEM: NUMBER OF ATOMS \n")
    fileoutput.write("%i \n" % (settings.N))
    fileoutput.write("ITEM: BOX BOUNDS \n")
    fileoutput.write("%e %e xlo xhi \n" % (0, settings.L))
    fileoutput.write("%e %e xlo xhi \n" % (0, settings.L))
    fileoutput.write("%e %e xlo xhi \n" % (0, settings.Lz))
    fileoutput.write("ITEM: ATOMS id type x y z \n")
    
    for i in range(0, settings.N):
        fileoutput.write("%i %i %e %e %e \n" % (i, 0, x[i], y[i], z[i]))
        

# def inputset():
#     return settings.xlo, settings.xhi, settings.ylo, settings.yhi, settings.eps, settings.r0, settings.cutoff, settings.deltat, settings.mass
    
def squarevelocity(vx, vy, mass):
    vx2 = 0
    vy2 = 0
    i = 0
    for i in range(0, len(vx)):
        vx2 += vx[i]**2
        vy2 += vy[i]**2
    return 0.5*mass*vx2, 0.5*mass*vy2
    
def timeit(func, *args, **kwargs): # kind regards to chatgpt for this function
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"{func.__name__} took {duration:.6f} seconds")
    return result, duration  

# def writeovito3d(x, y, z, name,L, mod, dt):
#     with open(name, "w") as file:
#         k=0
#         for i in range(traj.shape[0]):
#             if i % mod == 0:
#                 # wir beginne einen neuene  Zeitschritt
#                 file.write('ITEM: TIMESTEP'+ '\n')
#                 # file.write(str(k)+ '\n')
#                 # k += 1
#                 file.write(str(i * dt)+ '\n')
#                 file.write('ITEM: NUMBER OF ATOMS'+ '\n')
#                 file.write(str(traj.shape[1])+ '\n')
#                 file.write('ITEM: BOX BOUNDS'+ '\n')
#                 file.write('0 '+ str(L)+ '\n')
#                 file.write('0 '+ str(L)+ '\n')
#                 file.write('0 '+ str(L)+ '\n')
#                 file.write('ITEM: ATOMS id type x y z'+ '\n')
#                 for d in range(traj.shape[1]):
#                     file.write(str(d) + ' ' + str(0)+ ' '  +str(traj[i,d,0])+ ' '  + str(traj[i,d,1])+ ' '+ str(0)+'\n')
