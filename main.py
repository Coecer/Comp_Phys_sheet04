# MD code in microcanonical ensemble
# force field = LJ with eps = alpha * kb * T and r0 = 15 angstrom
# the treatment of the discontinuity at the cutoff is not implemented, i.e. the potential does not go smoothly to 0 at r = cutoff

import settings
import initialize
import force
import update
import sys
import time
import misc

start = time.time()


fileoutput = open("output_equilibration.txt", "w")
fileenergy = open("energy_equilibration.txt", "w")
fileenergy.write("#step  PE  KE  vx2 vy2\n")
# initialization of global variable
settings.init()

# create atomic locations and velocities + cancel linear momentum + rescale velocity to desired temperature
x, y, vx, vy = initialize.InitializeAtoms()

# save configuration to visualize
misc.WriteTrajectory(fileoutput, 0, x, y)

#initialize the forces
xlo, xhi, ylo, yhi, eps, sigma, cutoff, deltat, mass = misc.inputset()
fx, fy, epot = force.forceLJ(x, y, xlo, xhi, ylo, yhi, eps, sigma, cutoff)

#-------------- EQUILIBRATION ---------------#
for step in range(0, 1*settings.nsteps): #equilibration
    
    x, y, vx, vy, fx, fy, epot = update.VelocityVerlet(x, y, vx, vy, fx, fy, xlo, xhi, ylo, yhi, eps, sigma, cutoff, deltat, mass)
    
    if settings.Trescale  == 1 and step % 20 == 0: # rescaling of the temperature # the following lines should be defined as a routine in misc
        Trandom = initialize.temperature(vx, vy)
        vx, vy = initialize.rescalevelocity(vx, vy, settings.Tdesired, Trandom)
        Trandom1 = initialize.temperature(vx, vy)
        
    if step % 100 == 0:  #save the trajectory
        ekin = update.KineticEnergy(vx, vy, mass) # calculate the kinetic energy
        vx2, vy2 = misc.squarevelocity(vx, vy, mass) # calculate v_x^2 to compare with 0.5Nk_BT
        misc.WriteEnergy(fileenergy, step, epot, ekin, vx2, vy2)
        misc.WriteTrajectory(fileoutput, step, x, y)
        
fileoutput.close()
fileenergy.close()

#-------------- PRODUCTION ---------------#
fileoutput = open("output_prod.txt", "w")
fileenergy = open("energy_prod.txt", "w")   
fileenergy.write("#step  PE  KE  vx2 vy2\n")   
settings.Trescale = 0
for step in range(0, 2*settings.nsteps): #production
    
    x, y, vx, vy, fx, fy, epot = update.VelocityVerlet(x, y, vx, vy, fx, fy, xlo, xhi, ylo, yhi, eps, sigma, cutoff, deltat, mass)
    
    if step % 100 == 0:  #save the trajectory
        misc.WriteTrajectory(fileoutput, step, x, y)
        ekin = update.KineticEnergy(vx, vy, mass) # calculate the kinetic energy
        vx2, vy2 = misc.squarevelocity(vx, vy, mass)# calculate v_x^2 to compare with 0.5Nk_BT
        misc.WriteEnergy(fileenergy, step, epot, ekin, vx2, vy2)    
        
fileoutput.close()
fileenergy.close()
        
print("total time = ",time.time()-start)

