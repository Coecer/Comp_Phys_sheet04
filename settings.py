#settings


import numpy as np

def init():
    
    global nsteps            # number of time step to analyze
    nsteps = 1000         # t_prod/dt = 0.2ns / 2fs = 100 000
    global eqsteps          # eqsteps
    eqsteps = 500         # t_eq/dt = 0.1ns / 2fs = 50 000
    global mass              # mass of the LJ particles (gram/mole)
    mass = 39.95
    global kb                # boltzmann's constant (g*nm^2/fs^2/mole/K) 
    kb =0.0019858775 * 4.184e-6
    global Tdesired          # temperature of the experiment in K
    Tdesired = 300.
    global epsfluid               # eps in LJ (g*nm^2/fs^2/mole)
    epsfluid = 0.29788162 * 4.184e-6         # value from bottom table
    # epsfluid = 0.1488 * 4.184e-6               # value from text
    global sigfluid                # r0 in LJ (nm)
    sigfluid = 0.188

    # Wall values here
    global epsWall
    epsWall = 1.4887 * 4.184e-6     # CAN BE CHANGED LATER
    global sigWall
    sigWall = 0.0376        # CAN BE CHANGED LATER
    global cutoffWall
    cutoffWall = 2.5*sigWall


    global cutofffluid            # cutoff arbitrary at 2.5 r0
    cutofffluid = 2.5*sigfluid
    global dt          # time step (fs)
    dt = 0.1              # changed from 0.1 --> 2fs change back again for debuging
    global dr 
    dr = sigfluid/30
    global Nanalyze
    Nanalyze = 10

    
    # number of particle = n1*n2 distributed on s square lattice
    # global n
    global n12 #adding those as globals becaus we want to use them in initialize
    n12 = 2
    global n3
    n3 = 3
    global N #particle number
    N = n12 * n12 * n3
    global rho
    rho = 0.25*sigfluid**(-3) 
    global L # box length
    L = (N/(2*rho))**(1/3) # adaptation because z side is 2L long (so l is calculated in another manner)
    global Lz # new variable Lz
    Lz = 2*L

    
    global deltaxy # lattice parameter to setup the initial configuration on a lattice
    deltaxy = L/n12
    global deltaz
    deltaz = Lz/n3
    
    #rescaling of temperature
    global Trescale
    Trescale = 1 #1 = rescale temperature; 0 = no rescaling
    
    

    
