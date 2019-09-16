# Solving 1D PDEs with Neural Networks

Alejandro Francisco Queiruga  
Lawrence Berkeley National Lab, 2019

## The connection between 

The preprint can be found at
> Archive link

![networks](paper/CNN_FDM.png)

## Methodology

We use analytical solutions to generate datasets for the following equations:

1. The Heat Equation: Linear Parabolic
1. The Poisson Equation: Linear Elliptic
1. The Wave Equation: Linear Hyperbolic
1. Burger's Equation: Nonlinear conservative with shocks
1. Korteweg–de Vries (KdV) equation: Nonlinear with solitons
1. The Euler-Bernouli Beam Equation: Linear, biharmonic

The analytical solutions are all included in the `detest` testing framework should you wish to generate more data (or test a PDE solver.) The code to make the datasets is [analytical_solutions.ipynb](analytical_solutions.ipynb), and the outputs are .npz files in [data/](data).

## Hypotheses:

1. Purely advsererial training will work but slowly: **false**
1. Adding adverserial training will improve stability **unclear**
2. Need 2 history snapshots for the wave equation *todo*
3. Parameter-bottleneck autoencoder will not generalize to multiple
   trajectories *todo*
4. The CFL condition dictates connectivity and stability of the
   networks; need a U-net or a fully connected model for the heat
   equation and poisson equation. *todo*
   

