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
'Re40.gif' | flow pattern with Re=40
'Re400.gif' | flow pattern with Re=400
'Wave_Propapgation.gif' | Visco-ealsctic wave propagation (different BC from that in MS) 

Tests
-----

This study uses Underworld to reproduce the benchmark models provided by 

_Kolomenskiy, D. and Schneider, K., 2009. A Fourier spectral method for the Navier–Stokes equations with volume penalization for moving solid obstacles. Journal of Computational Physics, 228(16), pp.5687-5709._

_Lee, E.H. and Kanter, I., 1953. Wave propagation in finite rods of viscoelastic material. Journal of applied physics, 24(9), pp.1115-1122._

_Jiang, J., Erickson, B.A., Lambert, V.R., Ampuero, J.P., Ando, R., Barbot, S.D., Cattania, C., Zilio, L.D., Duan, B., Dunham, E.M. and Gabriel, A.A., 2022. Community‐driven code comparisons for three‐dimensional dynamic modeling of sequences of earthquakes and aseismic slip. Journal of Geophysical Research: Solid Earth, 127(3), p.e2021JB023519._

Better to take notebook file 'High-order_2ndOrd_RodWave_Maxwell.ipynb' as a start to check how different order FD approximaiton is implemented in the UW2 code

Parallel Safe
-------------
3D model results can be obtained in parallel operation.
