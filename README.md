# Homomorphic Post-Quantum Cryptography - Evaluation of Module Learning with Error in Homomorphic Cryptography

These are all related files for the Master Thesis. This Repo can run as a dev container, which makes it easier to set up the whole tooling for latex and python.

## Abstract

This thesis investigates the conversion of Ring-LWE (R-LWE)-based homomorphic encryption schemes to Module-LWE (M-LWE) and analyses the resulting performance differences. The advantage of M-LWE is that a fixed-sized polynomial degree can be utilized and the security of the system can be changed by increasing the vector/matrix dimension. This is the same concept that is utilized in the CRYSTALS-Kyber encryption scheme. The feasibility of transferring R-LWE to M-LWE is demonstrated based on the BFV homomorphic encryption scheme, showing that a functioning homomorphic encryption can be maintained. While the addition is straightforward, the multiplication necessitates the generation of multiple relinearization (evaluation) keys. It is demonstrated that the practical performance is only slightly inferior to that of R-LWE, with the advantage of smaller ciphertext sizes. However, there is still considerable scope for improvement in theoretical aspects, such as the study of security benefits and in practice, in enhancing the general performance.

## Folders
### Code
This folder contains all the Code that was used in the Master Thesis. The easiest way to run all the code is by using VS Code an installing all dependencies with [poetry](https://python-poetry.org/). The following files can be found:

- data/*: Contains the data that was generated to create the Plots in the Thesis
- BFV.py: The original R-LWE based BFV scheme
- LWE.ipynb: An script to learn and understand the basics of LWE and the different types. Everything in there is now explained even better in the Thesis. This was just a starting point
- ModuleBfv.py: the final Code with the Module BFV class. If this file is run directly, the three different Models are run and some statistics are printed.
- poetry.lock: The used versions of all libraries
- Polynomial.py: Contains the Polynomial and Polynomial class, which makes it possible to work with Ring Polynomials.
- pyproject.toml: contains the python definitions and all neccessary libraries
- Vis.ipynb: contains the code for creating all statistics and visualizing them.

### Expose
The original Expose that was handed in.

### Thesis
This file contains all Latex Files to recreate the Thesis. 