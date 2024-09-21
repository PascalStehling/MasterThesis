# Homomorphic Post-Quantum Cryptography - Evaluation of Module Learning with Error in Homomorphic Cryptography

These are all related files for the Master Thesis. This Repo can run as a dev container, which makes it easier to set up the whole tooling for latex and python.

## Code
This folder contains all the Code that was used in the Master Thesis. The easiest way to run all the code is by using VS Code an installing all dependencies with [poetry](https://python-poetry.org/). The following files can be found:

- data/*: Contains the data that was generated to create the Plots in the Thesis
- BFV.py: The original R-LWE based BFV scheme
- LWE.ipynb: An script to learn and understand the basics of LWE and the different types. Everything in there is now explained even better in the Thesis. This was just a starting point
- ModuleBfv.py: the final Code with the Module BFV class. If this file is run directly, the three different Models are run and some statistics are printed.
- poetry.lock: The used versions of all libraries
- Polynomial.py: Contains the Polynomial and Polynomial class, which makes it possible to work with Ring Polynomials.
- pyproject.toml: contains the python definitions and all neccessary libraries
- Vis.ipynb: contains the code for creating all statistics and visualizing them.

## Expose
The original Expose that was handed in.

## Thesis
This file contains all Latex Files to recreate the Thesis. 