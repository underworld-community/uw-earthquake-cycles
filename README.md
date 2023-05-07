<table>
 <tr>
  <td><img src='./Re40.gif'> Reynolds Number=40</td>
  <td><img src='./Re400.gif'> Reynolds Number=400</td>
 </tr>
 <tr>
  <td colspan="2"><img src='./Wave_Propagation.gif'> Viscoelastic Wave Propagation</td>
 </tr>
</table>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworld-community/template-project/master)


Abstract
-----
Visco-elastic-plastic modelling approaches for long-term tectonic deformation assume that co-seismic fault displacement can be integrated over 1,000s-10,000s years (tens of earthquake cycles) with the appropriate failure law, and that short-timescale fluctuations in the stress field due to individual earthquakes have no effect on long-term behavior. Models of the earthquake rupture process generally assume that the tectonic (long-range) stress field or kinematic boundary conditions are steady over the course of multiple earthquake cycles. This study is aimed to fill the gap between long-term and short-term deformations by modeling earthquake cycles with the rate-and-state frictional relationship in Navier-Stokes equations. We reproduce benchmarks at the earthquake timescale to demonstrate the effectiveness of our approach. We then discuss how these high-resolution models degrade if the time-step cannot capture the rupture process accurately and, from this, infer when it is important to consider coupling of the two timescales and the level of accuracy required. To build upon these benchmarks, we undertake a generic study of a thrust fault in the crust with a prescribed geometry. It is found that lower crustal rheology affects the periodic time of characteristic earthquake cycles and the inter-seismic, free-surface deformation rate. In particular, the relaxation of the surface of a cratonic region (with a relatively strong lower crust) has a characteristic double-peaked uplift profile that persists for thousands of years after a major slip event. This pattern might be diagnostic of active faults in cratonic regions.



File | Purpose
--- | ---
`Example_Reynolds_Strouhal_CylinderFlow.ipynb` | Viscous-incertial flow (Section 3.1 in MS). 
`High-order_2ndOrd_RodWave_Maxwell.ipynb`| Vicoelastic wave Propation in a Maxwell Rod (Section 3.1 in MS). 
`BP5-FD_imcompressible.py` | Fully dynamic modelling with incompressible media   (Section 3.3 in MS).
`Thrust_2D.py` | 2D thrust fault model (case study in the MS).
`Thrust_3D.py` | 3D thrust fault model (case study in the MS).

Tests
-----
**_Please specify how your repository is tested for correctness._**
**_Tests are not required for `laboratory` tagged repositories, although still encouraged._**
**_All other repositories must include a test._**

The attained peak VRMs time is tested against an expected value. If it is outside a given tolerance, an exception is raised.

Parallel Safe
-------------
**_Please specify if your model will operate in parallel, and any caveats._**

Yes, test result should be obtained in both serial and parallel operation.

Check-list
----------
- [ ] (Required) Have you replaced the above sections with your own content? 
- [ ] (Required) Have you updated the Dockerfile to point to your required UW/UWG version? 
- [ ] (Required) Have you included a working Binder badge/link so people can easily run your model?
                 You probably only need to replace `template-project` with your repo name. 
- [ ] (Optional) Have you included an appropriate image for your model? 
