# Multi-Compartment Models For Rat Brain Diffusion

Main code is in the notebooks
1. Simulation Notebook
2. Visualization and Measurement

Requires installation of the source code, by running 
```
pip install -e .
```
from the root folder.

Notes for future packaging with installation of dependencies:
* Requires manual installation of dolfin, since dolfin does not play well with any python package manager.

## Installation
```
conda create -n multirat -c conda-forge fenics matplotlib jupyter
conda activate multirat
pip install meshio
pip install git+https://github.com/SVMTK/SVMTK.git
pip install -e . 
```
