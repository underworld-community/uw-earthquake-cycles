# %%
# coding: utf-8
import underworld as uw
from underworld import function as fn

import numpy as np
import math
import os,csv
import mpi4py
import random
comm = mpi4py.MPI.COMM_WORLD

from underworld.scaling import units as u
from underworld.scaling import non_dimensionalise as nd



inputPath = os.path.join(os.path.abspath("."),"BENCHMARK_JGR_3D_128_64_64_dtvep2_Pre1BC1_FixedVStressD3_D1ms_Max50yr_0T1s_shearSameHalf_NoAdv/")
outputPath = inputPath
if uw.mpi.rank==0:
    print (uw.__version__)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.mpi.barrier

if uw.mpi.rank==0:
    if not os.path.exists(inputPath):
        os.makedirs(inputPath)
uw.mpi.barrier
# continue running from specific time step
LoadFromFile = False
Elasticity = True
meshdeform= False

scaling_coefficients = uw.scaling.get_coefficients()

# Define scale criteria
tempMin = 273.*u.degK 
tempMax = (500.+ 273.)*u.degK
bodyforce = 3300 * u.kilogram / u.metre**3 * 9.8 * u.meter / u.second**2
velocity = 4e10*u.centimeter/u.year

KL = 100e3*u.meter
Kt = KL/velocity
KT = tempMax 
KM = bodyforce * KL**2 * Kt**2
K  = 1.*u.mole
lengthScale = 100e3

scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM
scaling_coefficients["[temperature]"] = KT
scaling_coefficients["[substance]"] = K

gravity = nd(9.81 * u.meter / u.second**2)
R = nd(8.3144621 * u.joule / u.mole / u.degK)

# use low resolution if running in serial
xRes = 128
yRes = 64
zRes = 64
dim  = 3

minX = nd(   -48.* u.kilometer)
maxX = nd( 48. * u.kilometer)
minY = nd(  -50. * u.kilometer)
maxY = nd( 50. * u.kilometer)
minZ = nd(   -40. * u.kilometer)
maxZ = nd( 0. * u.kilometer)
stickyAirthick = nd(0. * u.kilometer)
stressNormalFn = nd(25e6*u.pascal)

H = nd(18*u.kilometer)
V0 = nd(1e-6*u.meter/u.second) # nd(4e-9*u.meter/u.second) # 

# Rate-and-state properties
miu0 = 0.6

L = nd(0.14*u.meter) 
# L = nd(0.01*u.meter) 
mu = nd(3.2e10*u.pascal) # elastic modulus
cs = nd(3464*u.meter/u.second)   #shear wave velocity
a_max = 0.04 #0.015 # 
a0 = 0.004 #0.003 #
b = 0.03 #0.009 #  

theta_rock = nd(1e16*u.year) # nd(102000.*u.year) #

pre_f  = 1.
BC_f = 1.

V_plate = nd(1e-9*u.meter/u.second)
shearVelocity = 0.5*V_plate #/np.pi*np.arctan(maxX/H)   #2*nd(6.3*u.centimeter/u.year)

thickUpCrust = nd(15. * u.kilometer)
BDLayer = nd(0. * u.kilometer)

stickyAirIndex        = 0
crustSouthIndex       = 1
crustNorthUpIndex     = 2
crustNorthLowIndex    = 3
crustValleyUpIndex    = 4
crustValleyLowIndex   = 5
mantleIndex           = 6
mantleWeekIndex       = 7
crustWeekIndex        = 8
fault                 = 9

if(LoadFromFile == True):
    step = 3000
    step_out = 100
    nsteps = 10000
    timestep = float(np.load(inputPath+"time"+str(step).zfill(4)+".npy"))
    dt_e = fn.misc.constant(float(np.load(outputPath+"dt"+str(step).zfill(4)+".npy"))) #fn.misc.constant(nd(0.02*u.second)) #
    Eqk = True
else:
    step = 0
    step_out = 100
    nsteps = 10000
    timestep = 0. 
    dt_e    = fn.misc.constant(nd(1.*u.second)) #fn.misc.constant(nd(0.02*u.year)) 
    Eqk = True
    # %%

dt_min= nd(1e-5*u.second)
dt_max = nd(50.*u.year)


mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"),
                                 elementRes  = (xRes, yRes, zRes), 
                                 minCoord    = (minX, minY, minZ), 
                                 maxCoord    = (maxX, maxY, maxZ),
                                 periodic    = [False, True, False]) 

# function to define refined mesh in the fault zone if needed
# not used in the example shown in this ms, but can be implemented easily

def mesh_Uni(section,minX,maxX,res,x1,x2,factor,mi):
    # section: mesh to be refined and return
    # res: segments numbers, same as resX(Y/Z)
    # x1: startpoint of the area to be refined
    # x2: endpoint of the area to be refine (x2>x1)
    # factor: the ratio of the finest area over the average of the section (maxX-minX)/resX
    # mi: power of two ending segments; mi>1 is required 
    section_cp = np.copy(section)
    Uni_all = (maxX-minX)/res
    spacing_Refine = Uni_all*factor 
    N_refine = ((x2-x1)/spacing_Refine)
    
    midPoint = (x1+x2)/2
    ratio = 0.5#(x1-minX)/(maxX-x2+x1-minX)
    startPoint1 =  midPoint-Uni_all*N_refine*ratio
    startPoint2 =  midPoint+Uni_all*N_refine*(1-ratio)
    
    
#     print (spacing_Refine,N_refine,startPoint1, startPoint2)
    
#     startPoint2-startPoint1)

    for index in range(len(section)):
        
        
        if section_cp[index]<=startPoint2 and section_cp[index]>=startPoint1:
            section[index] = x1 + (section_cp[index]-startPoint1)/Uni_all*spacing_Refine
            if section[index]>x2:
                section[index] = x2
                
        if  section_cp[index]<startPoint1:
            section[index] = x1 + (minX-x1)*((section_cp[index]-startPoint1)/(minX-startPoint1))**mi

        if section_cp[index]>startPoint2:        
            section[index] = x2 + (maxX-x2)*((section_cp[index]-startPoint2)/(maxX-startPoint2))**mi
            
    return section


dx = (maxX-minX)/xRes
dy = (maxY-minY)/yRes
dz = (maxZ-minZ)/zRes

interface_z = 0.
mesh.reset()
if (meshdeform == True):
    with mesh.deform_mesh():
         mesh_Uni(mesh.data[:,0],minX,maxX,xRes,-nd(1000*u.meter),nd(1000*u.meter),0.2,1.2);

    dx_min = (maxX-minX)/xRes*0.2
    
else:
    dx_min = dx

velocityField   = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=dim )
pressureField   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
pressureField0   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
stressField    = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=3 )

maskMesh    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureFieldCopy    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )

stressField.data[:] = [0.,0.,0.]
velocityField.data[:] = [0.,0.,0.]
pressureField.data[:] = 0.
pressureField0.data[:] = 0.


    
# send boundary condition information to underworld
iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
kWalls = mesh.specialSets["MinK_VertexSet"] + mesh.specialSets["MaxK_VertexSet"]
base    = mesh.specialSets["MinK_VertexSet"]
back    = mesh.specialSets["MaxJ_VertexSet"]
top    = mesh.specialSets["MaxK_VertexSet"]
left   = mesh.specialSets["MinI_VertexSet"]
right   = mesh.specialSets["MaxI_VertexSet"]
baseFix = mesh.specialSets['Empty']
leftFix = mesh.specialSets['Empty']
rightFix = mesh.specialSets['Empty']

velocityField.data[:] = 0.

# set "easier" intial velocity for the solver to solve

x = fn.input()[0]
conditionVMesh = [(True,-shearVelocity+(x-minX)*2.*shearVelocity/(maxX-minX))]
velocityField.data[:,1] = fn.branching.conditional(conditionVMesh).evaluate(mesh)[:,0]


    
for index in mesh.specialSets["MaxI_VertexSet"]:
    velocityField.data[index] = [0.,shearVelocity,0.] 
    
for index in mesh.specialSets["MinI_VertexSet"]:
    velocityField.data[index] = [0.,-shearVelocity,0.]


half_width = pre_f*dx_min
BC_half_width = BC_f*dx_min

for index in mesh.specialSets["MinK_VertexSet"]:
    #if mesh.data[index][0]<-BC_half_width:
    if mesh.data[index][0]<0.:
        velocityField.data[index][1] = -shearVelocity #+ (mesh.data[index][0]-minX)*2.*shearVelocity/(maxX-minX)
    #elif mesh.data[index][0]>=-BC_half_width and mesh.data[index][0]<=BC_half_width:
    elif mesh.data[index][0]>=0. and mesh.data[index][0]<=BC_half_width:
        # velocityField.data[index][1] = -shearVelocity + (mesh.data[index][0]+BC_half_width)*shearVelocity/BC_half_width
        velocityField.data[index][1] = -shearVelocity + (mesh.data[index][0])*shearVelocity/BC_half_width
    else:
        velocityField.data[index][1] = shearVelocity
        
freeslipBC = uw.conditions.DirichletCondition( variable        = velocityField,
                                               indexSetsPerDof = (iWalls,iWalls+base,kWalls) )    
    
    
swarm = uw.swarm.Swarm( mesh=mesh,particleEscape=True )
pop_control = uw.swarm.PopulationControl(swarm,aggressive=True,particlesPerCell=125)
surfaceSwarm = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
surfaceSwarm2 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
faultSwarm = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
previousVm =  swarm.add_variable( dataType="double", count=3 )
previousVm2 =  swarm.add_variable( dataType="double", count=3 )

velA  =  swarm.add_variable( dataType="double", count=3 )
vel_eff  =  swarm.add_variable( dataType="double", count=3 )

materialVariable   = swarm.add_variable( dataType="int", count=1 )

markSwarm1 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
markSwarm2 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
markSwarm3 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )
markSwarm4 = uw.swarm.Swarm( mesh=mesh,particleEscape=True  )

plasticStrain  = swarm.add_variable( dataType="double",  count=1 )
plasticStrain0 = swarm.add_variable( dataType="double",  count=1 )



cohesionStrength  = swarm.add_variable( dataType="double",  count=1 )
cohesionStrength_slip  = swarm.add_variable( dataType="double",  count=1 )
a_field  = swarm.add_variable( dataType="double",  count=1 )
L_field  = swarm.add_variable( dataType="double",  count=1 )
thetaField = swarm.add_variable( dataType="double",  count=1 )
swarmYield = swarm.add_variable( dataType="double",  count=1 )

#frictionInf  = swarm.add_variable( dataType="double",  count=1 )
#cohesion  = swarm.add_variable( dataType="double",  count=1 )
previousStress         = swarm.add_variable( dataType="double", count=6 )
faultVariable = faultSwarm.add_variable( dataType="double", count=1)

if(LoadFromFile == False): 
    #swarmLayout = uw.swarm.layouts.PerCellGaussLayout( swarm=swarm, gaussPointCount=5 )
    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout(swarm=swarm,particlesPerCell= 125)
    swarm.populate_using_layout( layout=swarmLayout )

if(LoadFromFile == True):    
    #surfaceSwarm.load(inputPath+"surfaceSwarm"+str(step).zfill(4))
    swarm.load(inputPath+"swarm"+str(step).zfill(4))
    materialVariable.load(inputPath+"materialVariable"+str(step).zfill(4))  
#     materialIndex.load(inputPath+"materialIndex"+str(step).zfill(4))
#     materialIndex1.load(inputPath+"materialIndex1"+str(step).zfill(4))
#     materialIndex2.load(inputPath+"materialIndex2"+str(step).zfill(4))
#     materialIndex3.load(inputPath+"materialIndex3"+str(step).zfill(4))
#     materialIndex4.load(inputPath+"materialIndex4"+str(step).zfill(4))
    velocityField.load(inputPath+"velocityField"+str(step).zfill(4)) 
    
#     plasticStrain.load(inputPath+"plasticStrain"+str(step).zfill(4))
    previousStress.load(inputPath+"previousStress"+str(step).zfill(4))
    a_field.load(inputPath+"a_field"+str(step).zfill(4))
    L_field.load(inputPath+"L_field"+str(step).zfill(4))
    thetaField.load(inputPath+"thetaField"+str(step).zfill(4))
    surfaceSwarm.load(inputPath+"surfaceSwarm"+str(step).zfill(4))
    surfaceSwarm2.load(inputPath+"surfaceSwarm2"+str(step).zfill(4))
    previousVm.load(inputPath+"previousVm"+str(step).zfill(4))
    previousVm2.load(inputPath+"previousVm2"+str(step).zfill(4))    

# set observation points 

y_ob = (maxY+minY)/2.

markSwarm1.add_particles_with_coordinates(np.array([[half_width,y_ob,-nd(1.*u.kilometer)]]))
markSwarm2.add_particles_with_coordinates(np.array([[half_width,y_ob,-nd(10.*u.kilometer)]]))
markSwarm3.add_particles_with_coordinates(np.array([[half_width,y_ob,-nd(20.*u.kilometer)]]))
markSwarm4.add_particles_with_coordinates(np.array([[half_width,y_ob,-nd(28.*u.kilometer)]]))


if(LoadFromFile == False):

    
    xcd3 = 0.
    starty = -nd(50.*u.kilometer)
    endy   = nd(50.*u.kilometer)
    
    faultShape3  = np.array([ (xcd3,starty), (xcd3+half_width,starty), (xcd3+half_width,endy),(xcd3,endy)])
    fault3= fn.shape.Polygon( faultShape3 )
    
    starty1 = -nd(30.*u.kilometer)
    endy1   = -nd(18.*u.kilometer)
    faultShape4  = np.array([ (xcd3,starty1), (xcd3+half_width,starty1), (xcd3+half_width,endy1),(xcd3,endy1)])
    fault4= fn.shape.Polygon( faultShape4 )
    
#fn.misc.max(-coordz*nd(28e6*u.pascal/u.kilometer)-nd(100e6*u.pascal), -coordz*nd(10e6*u.pascal/u.kilometer))

if LoadFromFile == False:    
    coordz =  fn.input()[2] 
    coordy =  fn.input()[1] 
    condMat = [(fault3,fault),
               (coordz<nd(-40.*u.kilometer), crustNorthLowIndex),
               (True,crustNorthUpIndex)]
    materialVariable.data[:] = fn.branching.conditional(condMat).evaluate(swarm) 

    rangey0 = (coordy>nd(-30.*u.kilometer))  & (coordy<nd(30.*u.kilometer))
    range_L = (coordy<=nd(-30.*u.kilometer)) & (coordy>=nd(-32.*u.kilometer))
    range_R = (coordy>=nd(30.*u.kilometer))  & (coordy<=nd(32.*u.kilometer))
    rangez0  = (coordz>nd(-16.*u.kilometer)) & (coordz<nd(-4.*u.kilometer))
    rangez1  = coordz>nd(-40.*u.kilometer)
    
    Vi = nd(0.03*u.meter/u.second)
    theta0 = L/V_plate
    theta1 = L/Vi
    ####>>>>>>>>>>>Earthquake    
#     condition_theta = [ #(coordz<nd(-18.*u.kilometer), nd(1.9e16*u.year)),
#                         (fault3,nd(0.029*u.year)),
#                         (True, nd(1.9e16*u.year))]
#     range_lf = coordy<=nd(50.*u.kilometer) and coordy>=nd(-50.*u.kilometer)
    condition_theta = [ ((fault4 & rangez0),theta1 ),
                        ((fault3 & rangez1),theta0),
                        (True, nd(1.9e20*u.year))]
        
    thetaField.data[:] = fn.branching.conditional(condition_theta).evaluate(swarm)
    
    
    #thetaField.save(inputPath+"thetaField0"+str(step).zfill(4)) 
    
    

    
    condition_a = [     ((range_L & (coordz<-nd(2.*u.kilometer)) & (coordz>=-nd(18.*u.kilometer))),a0-(a_max-a0)*(coordy+nd(30.*u.kilometer))/nd(2.*u.kilometer)),
                        ((range_R & (coordz<-nd(2.*u.kilometer)) & (coordz>=-nd(18.*u.kilometer))),a0+(a_max-a0)*(coordy-nd(30.*u.kilometer))/nd(2.*u.kilometer)), 
                        ((rangey0 & (coordz<-nd(2.*u.kilometer)) & (coordz>=-nd(4.*u.kilometer))),a0-(-a_max+a0)*(coordz+nd(4.*u.kilometer))/nd(2.*u.kilometer)),
                        ((rangey0 & (coordz<-nd(4.*u.kilometer)) & (coordz>=-nd(16.*u.kilometer))),a0),
                        ((rangey0 & (coordz<-nd(16.*u.kilometer)) & (coordz>=-nd(18.*u.kilometer))),a0-(a_max-a0)*(coordz+nd(16.*u.kilometer))/nd(2.*u.kilometer)),
                        (True, a_max)]
    
#     condition_a = [ (True, a_max)]
    
#     condition_a =      [(coordz>-nd(15.*u.kilometer),a0),
#                        (True, a_max)]
    a_field.data[:] = fn.branching.conditional(condition_a).evaluate(swarm)
    
    condition_Vi = [    ((fault4 & rangez0),Vi),
                        (True, V_plate)]
        
    ViFn = fn.branching.conditional(condition_Vi)

    condition_L = [    ((fault4 & rangez0),nd(0.13*u.meter)),
                       (True, nd(0.14*u.meter))]
        
    L_field.data[:] = fn.branching.conditional(condition_L).evaluate(swarm)
    
    kernalX = ViFn/(2.*V0)*fn.math.exp((miu0 + b*fn.math.log(V0/ViFn))/a_field)
    frictionFn0 = a_field*fn.math.log(kernalX+fn.math.sqrt(kernalX*kernalX+1.))


    #eta_factor = 2.*mu/cs
    stressShearFn = frictionFn0*stressNormalFn
    previousStress.data[:] = 1e-20 #0.1*stressNormalFn#
    previousStress.data[:,3] = stressShearFn.evaluate(swarm)[:,0] # + eta_factor*V_plate ( + for static only) #nd(0.7e7*u.pascal) # 

    ####>>>>>>>>>>>Earthquake
    
    countz=zRes*2
    
    zcoord = np.linspace(minZ,maxZ, countz)
    surfacePoints = np.zeros((countz,3))
    
    for k in range(countz):       

        surfacePoints[k,0] = dx_min#xcoord[j] 
        surfacePoints[k,1] = y_ob
        surfacePoints[k,2] = zcoord[k]  
    surfaceSwarm.add_particles_with_coordinates( surfacePoints )
    

    county=yRes*2
    
    ycoord = np.linspace(minY+nd(10*u.kilometer), maxY-nd(10*u.kilometer), county)   
    surfacePoints2 = np.zeros((county,3))
    for k in range(county):       

        surfacePoints2[k,0] = dx_min#xcoord[j] 
        surfacePoints2[k,1] = ycoord[k]
        surfacePoints2[k,2] = nd(-10*u.kilometer)  
        
    surfaceSwarm2.add_particles_with_coordinates( surfacePoints2 )
    
    
# maskCoreFn1 = fn.branching.conditional([(materialIndex1>1e8, 1.),
#                                             (True, 0.)])
# maskCoreFn2 = fn.branching.conditional([(materialIndex2>1e8, 1.),
#                                             (True, 0.)])
# maskCoreFn3 = fn.branching.conditional([(materialIndex3>1e8, 1.),
#                                             (True, 0.)])
# maskCoreFn4 = fn.branching.conditional([(materialIndex4>1e8, 1.),
#                                             (True, 0.)])

# maskCoreFn = fn.branching.conditional([(materialIndex>1e8, 1.),
#                                             (True, 0.)])




strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateFn)+nd(1e-18/u.second)

plasticStrain0.data[:] = 0.

VpFn = 2.*strainRate_2ndInvariantFn*half_width
thetaFieldFn  =  L_field/VpFn+(thetaField-L_field/VpFn)*fn.math.exp(-VpFn/L_field*dt_e)

if Eqk == True:

    #kernalX = VpFn/(2.*V0)*fn.math.exp((miu0 + b*fn.math.log(V0*thetaField/L))/a_field)
    #frictionFn = a_field*fn.math.log(kernalX+fn.math.sqrt(kernalX*kernalX+1.))
    
    frictionFn = miu0 + a_field*fn.math.log(VpFn/V0) +b*fn.math.log(V0*thetaField/L_field)
    yieldStressFn = frictionFn*stressNormalFn#pressureField nd(1e6*u.pascal)+


    densityMap0 = { fault           : nd(   2670. * u.kilogram / u.metre**3),
                 crustNorthUpIndex  : nd(   2670. * u.kilogram / u.metre**3),
                 crustNorthLowIndex : nd(   2670. * u.kilogram / u.metre**3),
                 }
densityFn = fn.branching.map( fn_key = materialVariable, mapping = densityMap0 )
z_hat=( 0.0, 0.0, -1.0 )
buoyancyFn = densityFn * z_hat *gravity




if (Elasticity == True):
    mappingDictViscosity = {  fault             : nd(5e29 * u.pascal * u.second),
                             crustNorthUpIndex  : nd(5e29 * u.pascal * u.second),
                             crustNorthLowIndex : nd(5e29 * u.pascal * u.second)}
    viscosityMapFn1 = fn.branching.map( fn_key = materialVariable, 
                                           mapping = mappingDictViscosity )

    alpha   = viscosityMapFn1 / mu                         # viscoelastic relaxation time

    viscoelasticViscosity  = ( viscosityMapFn1 * dt_e ) / (alpha + dt_e)  # effective viscosity

    visElsMap = {                       fault  : viscoelasticViscosity,
                            crustNorthUpIndex  : viscoelasticViscosity,
                            crustNorthLowIndex : viscoelasticViscosity}

    viscosityMapFn  = fn.branching.map( fn_key = materialVariable, 
                                               mapping = visElsMap )
    
    strainRate_effective = strainRateFn + 0.5*previousStress/(mu*dt_e)
    strainRate_effective_2ndInvariant = fn.tensor.second_invariant(strainRate_effective)+nd(1e-18/u.second)
    yieldingViscosityFn =  0.5 * yieldStressFn / strainRate_effective_2ndInvariant

    #viscosityFn = fn.exception.SafeMaths( fn.misc.max(fn.misc.min(yieldingViscosityFn, 
    #                                                             viscosityMapFn), min_viscosity))    
    #viscosityFn = fn.exception.SafeMaths( fn.misc.min(yieldingViscosityFn,viscosityMapFn))
    
    
    yieldFnMap = {                    fault     : yieldingViscosityFn,
                             crustNorthUpIndex  : nd(1e20*u.pascal),
                             crustNorthLowIndex : nd(1e20*u.pascal)}

    yieldFn  = fn.branching.map( fn_key =  materialVariable, 
                                    mapping = yieldFnMap )
    
    viscosityFn = ( fn.misc.min(yieldFn,viscosityMapFn))
    
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
    
    
    #plaRate2nd = strainRate_2ndInvariant*swarmYieldFn
    #plaStrainRateFn_2nd = fn.tensor.second_invariant(plaStrainRateFn)
    #elaStrainRateFn_2nd = fn.tensor.second_invariant(elaStrainRateFn)
    #visStrainRateFn_2nd = fn.tensor.second_invariant(visStrainRateFn)
    
    plaIncrement = plaStrainRateFn_2nd*swarmYieldFn
    #elaIncrement = elaStrainRateFn_2nd*swarmYieldFn
    #visIncrement = visStrainRateFn_2nd*swarmYieldFn
    
    stressMapFn = allStressFn
    
    LHS_fn = densityFn/dt_e 
    RHS_fn = densityFn*previousVm/dt_e
    
    stokes = uw.systems.Stokes(velocityField = velocityField, 
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


# %%

# use "lu" direct solve if running in serial
if(uw.mpi.size==1):
    solver.set_inner_method("lu")
else:
    solver.set_inner_method('mg')    
solver.set_penalty(1e-3) 
#solver.options.ksp_rtol=1e-8

inner_rtol = 1e-5
solver.set_inner_rtol(inner_rtol)
solver.set_outer_rtol(10*inner_rtol)

# solver.options.ksp_rtol=1e-5
# solver.options.scr.ksp_rtol = 1.0e-5




surfaceArea = uw.utils.Integral(fn=1.0,mesh=mesh, integrationType='surface', surfaceIndexSet=top)
surfacePressureIntegral = uw.utils.Integral(fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=top)


def pressure_calibrate():
    
    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate()
    offset = p0/area
    #print "Zeroing pressure using mean upper surface pressure {}".format( offset )
    pressureField.data[:] -= offset
    #plasticStrain0.data[:] = plaIncrement.evaluate(swarm) # plaRate2nd.evaluate(swarm)
    

def nonLinearSolver(step, nl_tol=1e-2, nl_maxIts=10):
    # a hand written non linear loop for stokes, with pressure correction

    er = 1.0
    its = 0                      # iteration count
    v_old = velocityField.copy() # old velocityField 

    while er > nl_tol and its < nl_maxIts:

        v_old.data[:] = velocityField.data[:]
        solver.solve(nonLinearIterate=False)

        # pressure correction for bob (the feed back pressure)
        
        (area,) = surfaceArea.evaluate()
        (p0,) = surfacePressureIntegral.evaluate()
        offset = p0/area
        #print "Zeroing pressure using mean upper surface pressure {}".format( offset )
        pressureField.data[:] -= offset
        #plasticStrain0.data[:] = plaIncrement.evaluate(swarm)


        # calculate relative error
        absErr = uw.utils._nps_2norm(velocityField.data-v_old.data)
        magT   = uw.utils._nps_2norm(v_old.data)
        er = absErr/magT
        if uw.mpi.rank==0.:
            print ("tolerance=", er,"iteration times=",its)
        uw.mpi.barrier
        
        its += 1
        
G_star = mu/(1.-0.5)
k_stiff = (2./3.1415926)*G_star/dx_min


# see Herrendorfer et al., 2018 for the defination of de_thea, dt_w, dt_vep

pusei = 0.25*fn.math.pow((k_stiff*L_field/(a_field*stressNormalFn)-(b-a_field)/a_field),2.)-k_stiff*L/(a_field*stressNormalFn)

pusei_Cond = [(pusei>0,a_field*stressNormalFn/(k_stiff*L_field-(b-a_field)*stressNormalFn)),
              (True,1.-(b-a_field)*stressNormalFn/(k_stiff*L_field))]
puseiFn = fn.branching.conditional(pusei_Cond)

   
dt_theta = fn.misc.min(fn.misc.constant(0.2),puseiFn)

# condw = [(plasticStrain0>0,dt_theta *L_field/VpFn),
#               (True,dt_theta *L_field/VpFn)]



# dt_wFn0 = fn.branching.conditional(condw)
# dt_wFn = fn.view.min_max(dt_wFn0)

dt_wFn = fn.view.min_max(0.3*L_field/VpFn)

#dt_wFn.evaluate(swarm)
#dt_w = dt_wFn.min_global()

dt_hFn = fn.view.min_max(thetaField*0.2)
#dt_vepFn = fn.view.min_max(0.2*vis_vp/mu)
dt_vepFn = fn.view.min_max(0.2*vis_vp/mu)

#delta_Fn = (yieldingViscosityFn-allStressFn_2nd)**2./pressureField**2.
#surfaceArea = uw.utils.Integral(fn=1.0,mesh=mesh)
#surface_dF_Integral = uw.utils.Integral(fn=delta_Fn, mesh=mesh)
        
               
        
advMat  = uw.systems.SwarmAdvector( swarm=swarm,     velocityField=velocityField, order=2 )
advSurf  = uw.systems.SwarmAdvector( swarm=surfaceSwarm, velocityField=velocityField, order=2 )
advSurf2  = uw.systems.SwarmAdvector( swarm=surfaceSwarm2, velocityField=velocityField, order=2 )
#The root mean square Velocity
velSquared = uw.utils.Integral( fn.math.dot(velocityField,velocityField), mesh )
area = uw.utils.Integral( 1., mesh )
Vrms = math.sqrt( velSquared.evaluate()[0]/area.evaluate()[0] )


time_factor = nd(1*u.year)

def update():

    dt = dt_e.value
    velA.data[:] = velocityField.evaluate(swarm)  
    vel_eff.data[:] = 1./4.*(velA.data[:]+2.*previousVm.data[:]+previousVm2.data[:])   
    # vel_eff.data[:] = 1./2.*(velA.data[:]+1.*previousVm.data[:])  
    previousVm2.data[:] = np.copy(previousVm.data[:])
    previousVm.data[:] = np.copy(velA.data[:])      
    # swarm advection can be ignored in earthquake cycle simulations due to small displacement with respect to grid size
    # with swarm.deform_swarm():
    #     swarm.data[:] += vel_eff.data[:]*dt 

    #update theta value in the RS frictional relationship
    condition_theta =  {               fault  : thetaFieldFn,
                           crustNorthUpIndex  : theta_rock,
                           crustNorthLowIndex : theta_rock
                       }
        
    thetaField.data[:] = fn.branching.map( fn_key = materialVariable, 
                                           mapping = condition_theta ).evaluate(swarm) 
    
    
    stressMapFn_data = stressMapFn.evaluate(swarm)
 

    previousStress.data[:] = stressMapFn_data[:]

     
#     advMat.integrate(dt)
    advSurf.integrate(dt)
    advSurf2.integrate(dt)
    #advMark.integrate(dt)
    pop_control.repopulate()     

    dt_wFn.reset()       
    dt_wFn.evaluate(swarm)    
    dt_w = dt_wFn.min_global()    

    dt_hFn.reset()
    dt_hFn.evaluate(swarm)
    dt_h = dt_hFn.min_global()

    dt_vepFn.reset()
    dt_vepFn.evaluate(swarm)
    dt_vep = dt_vepFn.min_global()
    

    V_fault =  fn.view.min_max(VpFn)
    V_fault.evaluate(mesh.subMesh)
    Vp_max = V_fault.max_global()

#     dt0 = np.min([dt_max,dt_vep,dt_w])
#   dt0 = np.min([dt_max,dt_vep,dt_w,2*dt_e.value])
    dt0 = np.min([dt_max,dt_vep])
#     dt_e.value =  np.max([dt0,dt_min,0.5*dt_e.value]) 
    dt_e.value =  np.max([dt0,dt_min]) 

    return timestep+dt, step+1


stressSample = np.zeros([4,1])
thetaSample = np.zeros([4,1])
fricSample = np.zeros([4,1])
velSample = np.zeros([4,1])

plsticIncrement = np.zeros([5,1])
Dissipation = np.zeros([9,1])

yieldHis = np.zeros([4,1])

if step == 0:
    title = ['step','time','F1','F2','F3','F4','dt_e','V1','V2','V3','V4']
    with open(outputPath+'Sample.csv', 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(title)

    
vdotv = uw.utils.Integral(fn.math.dot(velocityField,velocityField),mesh=mesh)
meshInt = uw.utils.Integral(fn=1.0,mesh=mesh)
Vrms = shearVelocity
V_rate_old = 0.
jump_step = 1.
while step<nsteps:
    
        
    solver.solve( nonLinearIterate=True,  nonLinearTolerance=1e-3, nonLinearMaxIterations=15,callback_post_solve = pressure_calibrate)

    picard_h = solver.get_nonLinearStats() 
    
    if picard_h.picard_residual>0.005:
        if uw.mpi.rank==0:
            print('more itr is called')
        uw.mpi.barrier()
#         dt_e.value = 0.9*dt_e.value
        solver.solve( nonLinearIterate=True,  nonLinearTolerance=1e-3, nonLinearMaxIterations=5,callback_post_solve = pressure_calibrate)        
#         picard_h = solver.get_nonLinearStats()
   
    
    dt_inner = advMat.get_max_dt() 
    while dt_inner < dt_e.value :
        
        dt_e.value = 0.5*dt_e.value
        #nonLinearSolver(step, nl_tol=1e-3, nl_maxIts=30)
        solver.solve(nonLinearIterate=True,  nonLinearTolerance=1e-3, nonLinearMaxIterations=30,callback_post_solve = pressure_calibrate)       
        dt_inner = advMat.get_max_dt()  
    
    visMin = 0.
    
    VelMM = fn.view.min_max(fn.math.abs(velocityField[1])*maskMesh)
    VelMM.evaluate(mesh)
    Vrms_new = VelMM.max_global()
    #Vrms = Vrms_new

  
    meshStress = uw.mesh.MeshVariable( mesh, 1 )
    projectorStress = uw.utils.MeshVariable_Projection( meshStress, allStressFn[3], type=0 )
    projectorStress.solve() 
    # output figure to file at intervals = steps_output
    if step %step_out == 0 or step == nsteps-1:
        #Important to set the timestep for the store object here or will overwrite previous step

        if (Elasticity == True):
            previousStress.save(outputPath+"previousStress"+str(step).zfill(4))

        

        '''
        meshFriction = uw.mesh.MeshVariable( mesh, 1 )
        projectorStress = uw.utils.MeshVariable_Projection( meshFriction, frictionFn, type=0 )
        projectorStress.solve() 
        frictionInf.data[:] = frictionFn.evaluate(swarm)
        '''

        mesh.save(outputPath+"mesh"+str(step).zfill(4))
        swarm.save(outputPath+"swarm"+str(step).zfill(4))
        materialVariable.save(outputPath+"materialVariable"+str(step).zfill(4))

        a_field.save(outputPath+"a_field"+str(step).zfill(4))
        L_field.save(outputPath+"L_field"+str(step).zfill(4))
        thetaField.save(outputPath+"thetaField"+str(step).zfill(4)) 
        previousVm.save(outputPath+"previousVm"+str(step).zfill(4))
        previousVm2.save(outputPath+"previousVm2"+str(step).zfill(4))
        
        #swarmYield.save(inputPath+"yieldSwarm"+str(step).zfill(4))
    if step % 10 == 0 :
        meshStress.save(outputPath+"meshStress"+str(step).zfill(4))     
        velocityField.save(outputPath+"velocityField"+str(step).zfill(4)) 
           
    if uw.mpi.rank==0:
        np.save(outputPath+"time"+str(step).zfill(4),timestep)
        np.save(outputPath+"dt"+str(step).zfill(4),dt_e.value)
#         np.save(outputPath+"plstRateAll0"+str(step).zfill(4),plstRateAll0)
    uw.mpi.barrier()

    dicMesh = { 'elements' : mesh.elementRes, 
                'minCoord' : mesh.minCoord,
                'maxCoord' : mesh.maxCoord}

    fo = open(outputPath+"dicMesh"+str(step).zfill(4),'w')
    fo.write(str(dicMesh))
    fo.close()


    stressMM = fn.view.min_max(fn.math.abs(meshStress))
    stressMM.evaluate(markSwarm2)
    stressSample[1] = stressMM.max_global()

    stressMM = fn.view.min_max(fn.math.abs(meshStress))
    stressMM.evaluate(markSwarm3)
    stressSample[2] = stressMM.max_global()


    velMM = fn.view.min_max(fn.math.abs(velocityField[1]))
    velMM.evaluate(markSwarm2)
    velSample[3] = velMM.max_global()

    velMM = fn.view.min_max(fn.math.abs(velocityField[1]))
    velMM.evaluate(markSwarm3)
    velSample[2] = velMM.max_global()

    V_fault =  fn.view.min_max(VpFn)
    V_fault.evaluate(mesh.subMesh)
    velSample[1] = V_fault.max_global()


    
    
    if uw.mpi.rank==0:
        SP_output = [step,timestep,stressSample[0,0],stressSample[1,0],stressSample[2,0],stressSample[3,0],dt_e.value,velSample[0,0],velSample[1,0],velSample[2,0],velSample[3,0]]
        with open(outputPath+'Sample.csv', 'a') as f:
            csv_write = csv.writer(f)  
            csv_write.writerow(SP_output)
    uw.mpi.barrier()
    

    surfaceSwarm.save(outputPath+"surfaceSwarm"+str(step).zfill(4)) 
    surfaceSwarm2.save(outputPath+"surfaceSwarm2"+str(step).zfill(4)) 
    
    if uw.mpi.rank==0:
        print('step = {0:6d}; time = {1:.3e};'.format(step,timestep/nd(1.*u.year)))
    uw.mpi.barrier()
    
    
    timestep, step = update()




