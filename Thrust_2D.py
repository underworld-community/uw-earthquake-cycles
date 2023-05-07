#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import warnings
warnings.filterwarnings("ignore")
import underworld as uw
import math
from underworld import function as fn
import numpy as np
import os,csv
import mpi4py 
# comm = mpi4py.MPI.COMM_WORLD
import random

from scipy.interpolate import interp1d

from underworld.scaling import units as u
from underworld.scaling import non_dimensionalise as nd

from scipy.interpolate import interp1d
import underworld as uw
from mpi4py import MPI as _MPI

comm = _MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# In[ ]:

inputPath = os.path.join(os.path.abspath("."),"LMS72096_dep33_2D/")
outputPath = os.path.join(os.path.abspath("."),"LMS72096_dep33_2D/")

if uw.mpi.rank==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
if uw.mpi.rank==0:
    if not os.path.exists(inputPath):
        os.makedirs(inputPath)
uw.mpi.barrier
# continue running from specific time step
LoadFromFile= False
# partial melting funciton in mantle
MeltFunction= False
# set upper limit of the topography
MaskTopo = False
# refine mesh in specific area
meshdeform = False
# topo_effect
Topography = False
# useFreeSurface = True


# Define scale criteria
tempMin = 273.*u.degK 
tempMax = (1400.+ 273.)*u.degK
bodyforce = 3300 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2
velocity = 1e8*u.centimeter/u.year

KL = 100e3*u.meter
Kt = KL/velocity
KT = tempMax 
KM = bodyforce * KL**2 * Kt**2
K  = 1.*u.mole
lengthScale = 100e3

scaling_coefficients = uw.scaling.get_coefficients()

scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM
scaling_coefficients["[temperature]"] = KT
scaling_coefficients["[substance]"] = K

gravity =  nd(9.81 * u.meter / u.second**2)
R = nd(8.3144621 * u.joule / u.mole / u.degK)

# use low resolution if running in serial

xRes = 720
zRes = 96
dim  = 2

minX = nd(   -200.* u.kilometer)
maxX = nd( 200. * u.kilometer)
minZ = nd(   -40. * u.kilometer)
maxZ = nd( 0. * u.kilometer)

stickyAirthick = nd(0. * u.kilometer)
stressNormalFn = nd(25e6*u.pascal)
meshV = (maxX-minX)*nd(1.*u.centimeter/u.year)/nd(400.*u.kilometer)# nd(1.*u.centimeter/u.year)

    
elementType = "Q1/dQ0"
resolution = (xRes,zRes)


H = nd(18*u.kilometer)
V0 = nd(4e-9*u.meter/u.second) # nd(4e-9*u.meter/u.second) # 

a_field = 0.003 #0.011 # 

b0 = 0.009# 0.017
b_max = 0.001
    
miu0 = 0.3
# b = 0.015 #0.009 #
L = nd(0.01*u.meter)#nd(0.01*u.meter) 

theta_rock = nd(1.9e16*u.year) # nd(102000.*u.year) #

pre_f  = 1.
BC_f = 2.
f_vep = 0.15

stressNormalFn = nd(30e6*u.pascal)

#index materials

UC = 1
LC = 2
fault = 3
faultLC = 4

# for mpi run
def global_max(localmax):
    return comm.allreduce(localmax, op=mpi4py.MPI.MAX) 

def global_min(localmin):
    return comm.allreduce(localmax, op=mpi4py.MPI.MIN) 

if(LoadFromFile == True):
    step = 6600
    step_out = 100
    maxSteps = 20000
    timestep = float(np.load(outputPath+"time"+str(step).zfill(4)+".npy"))
    dt_e =  fn.misc.constant(float(np.load(outputPath+"dt"+str(step).zfill(4)+".npy")))
    Eqk = True
else:
    step = 0
    step_out = 100
    maxSteps = 20000
    timestep = 0. 
    dt_e    = fn.misc.constant(nd(1e3*u.year)) 
    Eqk = True

    # %%
    
dx = (maxX-minX)/xRes
dy = (maxZ-minZ)/zRes

dt_min = nd(1e-2*u.second)
dt_max = nd(100000.*u.year)
dx_min = 3.*1.*dx #nd(3.*u.kilometer)    

LC_vis = nd(1e21 * u.pascal * u.second)
fault_lowDip = -nd(33.*u.kilometer)
x_shift = nd(0.*u.kilometer)
# In[ ]:

mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"),
                                 elementRes  = (xRes, zRes), 
                                 minCoord    = (minX, minZ), 
                                 maxCoord    = (maxX, maxZ),
                                 periodic    = [False, False]) 
velocityField   = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=dim )
pressureField   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureFieldCopy    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )

previousVmMesh = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=mesh.dim)
velAMesh  =  uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=mesh.dim)
vel_effMesh  =  uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=mesh.dim)

velocityField.data[:] = 0.
pressureField.data[:] = 0.



if(LoadFromFile == True):
    # Setup mesh and temperature field for 64*64 data file.
    # read in saved steady state temperature field data

#     temperatureField.load(inputPath+"temperatureField"+str(step).zfill(4)) 
    velocityField.load(inputPath+"velocityField"+str(step).zfill(4))
    previousVmMesh.load(inputPath+"previousVmMesh"+str(step).zfill(4))
#     pressureField.load(inputPath+"pressureField"+str(step).zfill(4))    
    #velocityField.data[:] = [0.,0.]

# In[ ]:

# set initial conditions (and boundary values)


    
# send boundary condition information to underworld
iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]

top    = mesh.specialSets["MaxJ_VertexSet"]
base   = mesh.specialSets["MinJ_VertexSet"]
# left   = mesh.specialSets["MinI_VertexSet"]


swarm = uw.swarm.Swarm( mesh=mesh,particleEscape=True )
faultSwarm = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
surfaceSwarm1 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
surfaceSwarm2 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
surfaceV1 = surfaceSwarm1.add_variable( dataType="double", count=mesh.dim )
surfaceV2 = surfaceSwarm2.add_variable( dataType="double", count=mesh.dim )
pop_control = uw.swarm.PopulationControl(swarm,aggressive=True,particlesPerCell= 30)

materialVariable   = swarm.add_variable( dataType="int", count=1 )
plasticStrain  = swarm.add_variable( dataType="double",  count=1 )
frictionInf  = swarm.add_variable( dataType="double",  count=1 )
cohesion  = swarm.add_variable( dataType="double",  count=1 )


previousVm =  swarm.add_variable( dataType="double", count=mesh.dim  )
previousVm2 =  swarm.add_variable( dataType="double", count=mesh.dim  )
velA  =  swarm.add_variable( dataType="double", count=mesh.dim  )
vel_eff  =  swarm.add_variable( dataType="double", count=mesh.dim  )

# a_field  = swarm.add_variable( dataType="double",  count=1 )
b  = swarm.add_variable( dataType="double",  count=1 )

thetaField = swarm.add_variable( dataType="double",  count=1 )
swarmYield = swarm.add_variable( dataType="double",  count=1 )
previousStress         = swarm.add_variable( dataType="double", count=3 )

markSwarm1 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
markSwarm2 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
markSwarm3 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
markSwarm4 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )


markSwarm1.add_particles_with_coordinates(np.array([[-nd(1.5*u.kilometer)+x_shift+0.5*dx_min,-nd(3*u.kilometer)]]))
markSwarm2.add_particles_with_coordinates(np.array([[-nd(14.5*u.kilometer)+x_shift+0.5*dx_min,-nd(10*u.kilometer)]]))
markSwarm3.add_particles_with_coordinates(np.array([[-nd(14.5*u.kilometer)+x_shift+0.5*dx_min,-nd(10*u.kilometer)]]))
# markSwarm3.add_particles_with_coordinates(np.array([[-nd(30*u.kilometer),-nd(20.*u.kilometer)]]))
markSwarm4.add_particles_with_coordinates(np.array([[-nd(63*u.kilometer)+x_shift+0.5*dx_min,-nd(28*u.kilometer)]]))



if(LoadFromFile == False): 
    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell= 30 )
    swarm.populate_using_layout( layout=swarmLayout )
    
    previousVm.data[:] = 0.
    previousVmMesh.data[:] = 0.

    
if(LoadFromFile == True):    
    swarm.load(inputPath+"swarm"+str(step).zfill(4))
    materialVariable.load(inputPath+"materialVariable"+str(step).zfill(4))   
#     plasticStrain.load(inputPath+"plasticStrain"+str(step).zfill(4))
    previousStress.load(inputPath+"previousStress"+str(step).zfill(4))
    b.load(inputPath+"a_field"+str(step).zfill(4))
    thetaField.load(inputPath+"thetaField"+str(step).zfill(4))
#     surfaceSwarm.load(inputPath+"surfaceSwarm"+str(step).zfill(4))
    previousVm.load(inputPath+"previousVm"+str(step).zfill(4))
#     previousVm2.load(inputPath+"previousVm2"+str(step).zfill(4))   
    
p01 = fn.misc.constant([-nd(120.*u.kilometer),nd(0.0*u.kilometer),nd(0.0*u.kilometer)])
p02 = fn.misc.constant([nd(0.*u.kilometer),nd(0.0*u.kilometer),nd(0.0*u.kilometer)])


coord = fn.input()
x=coord[0]
y=coord[1]


x_ref = nd(-120.*u.kilometer)
#     x_ref = nd(-50.*u.kilometer)
x1 = x_ref+x_shift#-Tibet_ELC_width #nd(-210.*u.kilometer)
top_coord = nd(-2.*u.kilometer)
base_coord = fault_lowDip #nd(-15.*u.kilometer)
diff_coord = top_coord-base_coord
y1 = diff_coord/((x_shift-x1)**2)*(x-x1)**2+base_coord
x1_r = fn.math.abs(fn.math.sqrt((y-base_coord)*(x1-x_shift)**2/diff_coord))+x1


material_map = [(((y>y1-1*dx_min) & (x<x_shift) & (x>x1) & (y<y1) & (y<top_coord) & (y>nd(-30.*u.kilometer))),fault),
(((x>x1_r) & (x<x1_r+dx_min) & (y<y1) & (y<top_coord) & (y>nd(-30.*u.kilometer)) ),fault),               
(((y>y1-1*dx_min) & (x<x_shift) & (x>x1) & (y<y1) & (y<top_coord) & (y<=nd(-30.*u.kilometer))),faultLC),
(((x>x1_r) & (x<x1_r+dx_min) & (y<y1) & (y<top_coord) & (y<=nd(-30.*u.kilometer))  ),faultLC),             
                (y>nd(-30*u.kilometer),UC),                    
                (True,LC)]


materialVariable.data[:] = fn.branching.conditional(material_map).evaluate(swarm)


plasticStrain.data[:] = 0.

countz = 100
surfacePoints = np.zeros((countz,2))
surfacePoints[:,0] = np.linspace(x1,x_shift,countz)
surfacePoints[:,1] = diff_coord/((x_shift-x1)**2)*(surfacePoints[:,0]-x1)**2+base_coord
faultSwarm.add_particles_with_coordinates( surfacePoints )

countz = 100
surfacePoints1 = np.zeros((countz,2))

surfacePoints1[:,0] = np.linspace(minX,maxX,countz)
surfacePoints1[:,1] = 0.

surfacePoints2 = np.zeros((countz,2))

surfacePoints2[:,0] = np.linspace(minX,maxX,countz)
surfacePoints2[:,1] = 0.

surfaceSwarm1.add_particles_with_coordinates( surfacePoints1 )
surfaceSwarm2.add_particles_with_coordinates( surfacePoints2 )

surfaceSwarm1.save(outputPath+"surfaceSwarm1"+str(0).zfill(4))
surfaceSwarm2.save(outputPath+"surfaceSwarm2"+str(0).zfill(4))

if LoadFromFile == False:    
    coordz =  fn.input()[1] 

    condition_b =      [(coordz>p01.value[1],b_max),
                        (((coordz>-nd(27.*u.kilometer)) & (coordz<= p01.value[1])),b0),
                        (coordz>-nd(30.*u.kilometer),b0-(b_max-b0)*(coordz+nd(27.*u.kilometer))/nd(3.*u.kilometer)),
                        (True, b_max)]
    
    b.data[:] = fn.branching.conditional(condition_b).evaluate(swarm)
    
    previousStress.data[:] = 0. #0.1*stressNormalFn#
    
    condition_theta = {         
                                   UC  : theta_rock,
                                   LC  : theta_rock,
                                fault  : nd(0.029*u.year),
                              faultLC  : nd(0.029*u.year),
                   }

    thetaField.data[:] = fn.branching.map( fn_key = materialVariable, 
                                               mapping = condition_theta ).evaluate(swarm)  
    
    
strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateFn)+nd(1e-18/u.second)


VpFn = 2.*strainRate_2ndInvariantFn*dx_min 
thetaFieldFn  =  L/VpFn+(thetaField-L/VpFn)*fn.math.exp(-VpFn/L*dt_e)


kernalX = VpFn/(2.*V0)*fn.math.exp((miu0 + b*fn.math.log(V0*thetaField/L))/a_field)
frictionFn = a_field*fn.math.log(kernalX+fn.math.sqrt(kernalX*kernalX+1.))



yieldStressFn0  = frictionFn*stressNormalFn #pressureField nd(1e6*u.pascal)+

yieldMax = nd(1e20*u.pascal)

yieldMap =  {                
                               UC  : yieldMax,
                               LC  : yieldMax,
                            fault  : yieldStressFn0,
                          faultLC  : yieldStressFn0,
            }

yieldStressFn  = fn.branching.map( fn_key = materialVariable, 
                                           mapping = yieldMap )


viscosityMap = {            
                               UC  : nd(1e27 * u.pascal * u.second),
                               LC  : LC_vis,
                          fault    : nd(1e27 * u.pascal * u.second),
                         faultLC   : LC_vis,
               }

viscosityMapFn1  = fn.branching.map( fn_key = materialVariable, 
                                           mapping = viscosityMap )
# viscosityMapFn = fn.exception.SafeMaths( fn.misc.min(yieldingViscosityFn ,backgroundViscosityFn))


mu0      = nd(3e10*u.pascal) # elastic modulus

muMap =        {           
                               UC  : mu0,
                               LC  : mu0, 
                            fault  : mu0,
                          faultLC  : mu0,
               }

mu  = fn.branching.map( fn_key = materialVariable, 
                       mapping = muMap )


alpha   = viscosityMapFn1 / mu                         # viscoelastic relaxation time

viscoelasticViscosity  = ( viscosityMapFn1 * dt_e ) / (alpha + dt_e)  # effective viscosity

visElsMap = { 
                                UC  : viscoelasticViscosity,
                                LC  : viscoelasticViscosity,
                            fault   : viscoelasticViscosity,
                            faultLC   : viscoelasticViscosity,
               }

viscosityMapFn  = fn.branching.map( fn_key = materialVariable, 
                                           mapping = visElsMap )

strainRate_effective = strainRateFn + 0.5*previousStress/(mu*dt_e)
strainRate_effective_2ndInvariant = fn.tensor.second_invariant(strainRate_effective)+nd(1e-18/u.second)
yieldingViscosityFn =  0.5 * yieldStressFn / strainRate_effective_2ndInvariant

viscosityFn0 = ( fn.misc.min(yieldingViscosityFn,viscosityMapFn))


viscosityFnMp = {           
                               UC  : viscosityFn0,
                               LC  : viscosityFn0,
                           fault   : viscosityFn0,
                         faultLC   : viscosityFn0,
               }

viscosityFn  = fn.branching.map( fn_key = materialVariable, 
                                           mapping = viscosityFnMp )

# contribution from elastic rheology
tauHistoryFn    = viscosityFn / ( mu * dt_e ) * previousStress 
# stress from all contributions, including elastic,viscous,plastic (if yielded)
allStressFn     = 2. * viscosityFn * strainRate_effective#
allStressFn_2nd = fn.tensor.second_invariant(allStressFn)

visStrainRateFn = allStressFn/(2.*viscosityMapFn1)

elaStrainRateFn = (allStressFn-previousStress)/dt_e/(2.*mu)

plaStrainRateFn = strainRateFn - visStrainRateFn - elaStrainRateFn

plaStrainRateFn_2nd = fn.tensor.second_invariant(plaStrainRateFn)  

vis_vp = viscosityMapFn1*allStressFn_2nd/(2.*viscosityMapFn1*plaStrainRateFn_2nd+allStressFn_2nd)

swarmYield_Cond = [(viscosityMapFn>viscosityFn,1.),
                   (True,0.)]
swarmYieldFn = fn.branching.conditional(swarmYield_Cond)

plaIncrement = plaStrainRateFn_2nd*swarmYieldFn

densityMap0  = {           
                               UC  : nd(   2700. * u.kilogram / u.metre**3),
                               LC  : nd(   2950. * u.kilogram / u.metre**3),
                         fault     : nd(   2700. * u.kilogram / u.metre**3),
                       faultLC     : nd(   2950. * u.kilogram / u.metre**3)
               }

densityFn = fn.branching.map( fn_key = materialVariable, mapping = densityMap0 )

# Define our vertical unit vector using a python tuple
z_hat = ( 0.0, -1.0 )

# now create a buoyancy force vector
buoyancyFn = densityFn * z_hat * gravity



velocityField.data[:] = 0.
for index in mesh.specialSets["MinI_VertexSet"]:
    velocityField.data[index,0] = meshV # 0. + (mesh.data[index][1]-minY)*meshV/(maxY-minY)
for index in mesh.specialSets["MaxI_VertexSet"]:
    velocityField.data[index,0] = 0.

        
freeslipBC = uw.conditions.DirichletCondition( variable        = velocityField, 
                                               indexSetsPerDof = (iWalls,base) )



LHS_fn = densityFn/dt_e 
RHS_fn = densityFn*previousVm/dt_e

stokes = uw.systems.Stokes(    velocityField = velocityField, 
                           pressureField = pressureField,
                           voronoi_swarm = swarm, 
                           conditions    = freeslipBC,
                           fn_viscosity  = viscosityFn, 
                           #fn_bodyforce  = buoyancyFn,
                           fn_bodyforce  = buoyancyFn+RHS_fn,
                           fn_stresshistory = tauHistoryFn)

massMatrixTerm = uw.systems.sle.MatrixAssemblyTerm_NA__NB__Fn(
                    assembledObject  = stokes._kmatrix,
                    integrationSwarm = stokes._constitMatTerm._integrationSwarm,
                    fn   = LHS_fn,
                    mesh = mesh)
    
# Create solver & solve
solver = uw.systems.Solver(stokes)


# In[ ]:

# use "lu" direct solve if running in serial
if(uw.mpi.size==1):
    solver.set_inner_method("lu")
else:
    solver.set_inner_method("mumps")    
solver.set_penalty(1.0e-3)


# In[7]:


advector1  = uw.systems.SwarmAdvector( swarm=swarm,     velocityField=velocityField, order=2 )

surfaceArea = uw.utils.Integral(fn=1.0,mesh=mesh, integrationType='surface', surfaceIndexSet=top)
surfacePressureIntegral = uw.utils.Integral(fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=top)

def pressure_calibrate():
    
    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate()
    offset = p0/area
    #print "Zeroing pressure using mean upper surface pressure {}".format( offset )
    pressureField.data[:] -= offset



#The root mean square Velocity
velSquared = uw.utils.Integral( fn.math.dot(velocityField,velocityField), mesh )
area = uw.utils.Integral( 1., mesh )
Vrms = math.sqrt( velSquared.evaluate()[0]/area.evaluate()[0] )

G_star = mu/(1.-0.5)
k_stiff = (2./3.1415926)*G_star/dx_min


pusei = 0.25*fn.math.pow((k_stiff*L/(a_field*stressNormalFn)-(b-a_field)/a_field),2.)-k_stiff*L/(a_field*stressNormalFn)

pusei_Cond = [(pusei>0,a_field*stressNormalFn/(k_stiff*L-(b-a_field)*stressNormalFn)),
              (True,1.-(b-a_field)*stressNormalFn/(k_stiff*L))]
puseiFn = fn.branching.conditional(pusei_Cond)
   
dt_theta = fn.misc.min(fn.misc.constant(0.2),puseiFn)


dt_wFn0 = dt_theta *L/VpFn #fn.branching.conditional(condw)
dt_wFn = fn.view.min_max(dt_wFn0)

dt_vepFn = fn.view.min_max(vis_vp/mu)

dt_hFn = fn.view.min_max(thetaField*0.2)

time_factor = nd(1.*u.year)

Km_fn = fn.view.min_max((1.*dx)**2*densityFn/viscosityFn)



stressSample = np.zeros([4,1])
thetaSample = np.zeros([4,1])
fricSample = np.zeros([4,1])
velSample = np.zeros([4,1])

if step == 0:
    title = ['step','time','F1','F2','F3','F4','dt_e','V1','V2','V3','V4']
    with open(outputPath+'Sample.csv', 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(title)

    title = ['step','time','Theta1','Theta2','Theta3','Theta4','fric1','fric2','fric3','fric4']
    with open(outputPath+'Sample2.csv', 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(title)
    
    title = ['step','time','dt_vep','dt_km','dt_default']
    with open(outputPath+'Dt.csv', 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(title)
        
v_abs = fn.math.sqrt(fn.math.dot(velocityField,velocityField))       

while step <= maxSteps:
    
    # Solve non linear Stokes system
    solver.solve( nonLinearIterate=True,  nonLinearTolerance=1e-3, nonLinearMaxIterations=15,callback_post_solve = pressure_calibrate)


        
    stress2Data = fn.tensor.second_invariant(allStressFn)
    meshStress = uw.mesh.MeshVariable( mesh, 1 )
    projectorStress = uw.utils.MeshVariable_Projection( meshStress, stress2Data, type=0 )
    projectorStress.solve()
    
    meshFriction = uw.mesh.MeshVariable( mesh, 1 )
    projectorFriction = uw.utils.MeshVariable_Projection( meshFriction, frictionFn, type=0 )
    projectorFriction.solve()
    
    surfaceV1.data[:] = velocityField.evaluate(surfaceSwarm1)
    surfaceV2.data[:] = velocityField.evaluate(surfaceSwarm2)
    
    # output figure to file at intervals = steps_output    
    if step % step_out == 0 or step == maxSteps-1:

        meshViscosity = uw.mesh.MeshVariable( mesh, 1 )
        meshViscosity2 = uw.mesh.MeshVariable( mesh, 1 )
        
        
        meshMeltF = uw.mesh.MeshVariable( mesh, 1 )
#        
        
        projectorViscosity = uw.utils.MeshVariable_Projection( meshViscosity,viscosityFn, type=0 )
        projectorViscosity.solve()

        projectorViscosity2 = uw.utils.MeshVariable_Projection( meshViscosity2,viscosityMapFn1, type=0 )
        projectorViscosity2.solve()
        
#         projectorStress = uw.utils.MeshVariable_Projection( meshDevStress, stress2Data, type=0 )
#         projectorStress.solve()
        
        meshViscosity2.save(outputPath+"meshViscosity2"+str(step).zfill(4))        
        meshViscosity.save(outputPath+"meshViscosity"+str(step).zfill(4)) 
        

#         meshStress = uw.mesh.MeshVariable( mesh, 3 )
#         meshStress.data[:] = allStressFn.evaluate(mesh)
        


        swarm.save(outputPath+"swarm"+str(step).zfill(4))
        mesh.save(outputPath+"mesh"+str(step).zfill(4))

        temperatureField.save(outputPath+"temperatureField"+str(step).zfill(4)) 
        previousStress.save(outputPath+"previousStress"+str(step).zfill(4))
        materialVariable.save(outputPath+"materialVariable"+str(step).zfill(4))

#         temperatureField.save(outputPath+"temperatureField"+str(step).zfill(4))
#         pressureField.save(outputPath+"pressureField"+str(step).zfill(4))       
#         plasticStrain.save(outputPath+"plasticStrain"+str(step).zfill(4))
        previousVm.save(outputPath+"previousVm"+str(step).zfill(4))
        b.save(outputPath+"a_field"+str(step).zfill(4))
        thetaField.save(outputPath+"thetaField"+str(step).zfill(4))
        meshFriction.save(outputPath+"meshFriction"+str(step).zfill(4))
        previousVmMesh.save(outputPath+"previousVmMesh"+str(step).zfill(4))    

    velocityField.save(outputPath+"velocityField"+str(step).zfill(4))
    meshStress.save(outputPath+"meshStress"+str(step).zfill(4)) 
    surfaceV1.save(outputPath+"surfaceV1"+str(step).zfill(4))
    surfaceV2.save(outputPath+"surfaceV2"+str(step).zfill(4))
#     surfaceSwarm.save(outputPath+"surfaceSwarm"+str(step).zfill(4)) 
    if uw.mpi.rank==0:

        np.save(outputPath+"time"+str(step).zfill(4),timestep)
        np.save(outputPath+"dt"+str(step).zfill(4),dt_e.value)

    uw.mpi.barrier

    dicMesh = { 'elements' : mesh.elementRes, 
                'minCoord' : mesh.minCoord,
                'maxCoord' : mesh.maxCoord}

    fo = open(outputPath+"dicMesh"+str(step).zfill(4),'w')
    fo.write(str(dicMesh))
    fo.close()
        
    stressMM = fn.view.min_max(fn.math.abs(meshStress))
    stressMM.evaluate(markSwarm1)
    stressSample[0] = stressMM.max_global()

    stressMM = fn.view.min_max(fn.math.abs(meshStress))
    stressMM.evaluate(markSwarm2)
    stressSample[1] = stressMM.max_global()

    stressMM = fn.view.min_max(fn.math.abs(meshStress))
    stressMM.evaluate(markSwarm3)
    stressSample[2] = stressMM.max_global()

    stressMM = fn.view.min_max(fn.math.abs(meshStress))
    stressMM.evaluate(markSwarm4)
    stressSample[3] = stressMM.max_global()

    velMM = fn.view.min_max(fn.math.abs(VpFn ))
    velMM.evaluate(markSwarm1)
    velSample[0] = velMM.max_global()

    velMM = fn.view.min_max(fn.math.abs(VpFn ))
    velMM.evaluate(markSwarm2)
    velSample[1] = velMM.max_global()

    velMM = fn.view.min_max(fn.math.abs(VpFn ))
    velMM.evaluate(markSwarm3)
    velSample[2] = velMM.max_global()

    velMM = fn.view.min_max(fn.math.abs(VpFn ))
    velMM.evaluate(markSwarm4)
    velSample[3] = velMM.max_global()

    
    if uw.mpi.rank==0:
        SP_output = [step,timestep,stressSample[0,0],stressSample[1,0],stressSample[2,0],stressSample[3,0],dt_e.value,velSample[0,0],velSample[1,0],velSample[2,0],velSample[3,0]]
        with open(outputPath+'Sample.csv', 'a') as f:
            csv_write = csv.writer(f)  
#             csv_write.writerow(SP_output)
            csv_write.writerow(['{:.18e}'.format(x) for x in SP_output])
    uw.mpi.barrier()
    

    
    Vrms = math.sqrt( velSquared.evaluate()[0]/area.evaluate()[0] )
    
    if uw.mpi.rank==0:
        print('step = {0:6d}; time = {1:.3e} yr; Vrms = {2:.3e}'.format(step,timestep/nd(1.*u.year),Vrms))  
    uw.mpi.barrier()
    
    dt = dt_e.value
    
    velA.data[:] = velocityField.evaluate(swarm)  
#     vel_eff.data[:] = 1./4.*(velA.data[:]+2.*previousVm.data[:]+previousVm2.data[:])   
    vel_eff.data[:] = 1./2.*(velA.data[:]+1.*previousVm.data[:])  
#     previousVm2.data[:] = np.copy(previousVm.data[:])
    previousVm.data[:] = np.copy(velA.data[:])      

    velAMesh.data[:] = velocityField.evaluate(mesh)  
    vel_effMesh.data[:] = 1./2.*(velAMesh.data[:]+1.*previousVmMesh.data[:])  

        
    previousStress.data[:] = allStressFn.evaluate(swarm)
    
 
    condition_theta = {      
                               UC  : theta_rock,
                               LC  : theta_rock,

                            fault  : thetaFieldFn,
                           faultLC  : thetaFieldFn,
               }

    thetaField.data[:] = fn.branching.map( fn_key = materialVariable, 
                                           mapping = condition_theta ).evaluate(swarm) 

     
    if step>15:
        dt_vepFn.reset()
        dt_vepFn.evaluate(swarm)
        dt_vep0 = dt_vepFn.min_global()

    

    
        dt_vep = f_vep*dt_vep0
        dt0 = np.min([dt_max,dt_vep])
        dt_e.value =  np.max([dt0,dt_min]) 
        

            
    
    timestep = timestep+dt
    step = step+1