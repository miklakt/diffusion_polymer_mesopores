# diffusion_polymer_mesopores
Code showcase for modeling diffusion resistance of nanocolloids in polymer-filled mesopores. Uses SCFT-derived insertion free energy and position-dependent diffusion profiles, solved with the Smoluchowski equation on a cylindrical 2D lattice.

This repository provides a complete step-by-step demonstration of how to compute pore resistance for given pore geometries, polymer volume fraction distributions, particle size, and Flory–Huggins interaction parameters (χ).  

It bridges self-consistent field calculations (obtained via the **Namics** package) with analytical and numerical modeling.

## Original Study

This repository accompanies the manuscript **A polymer filling enhances the rate and selectivity of colloid permeation across mesopores**

**Abstract:**
Polymer-functionalised mesopores are an emerging technology for colloid separation, sensing and delivery.
Their potential is strikingly illustrated in living cells, where nuclear pore complexes (NPCs) control biocolloid transport between the nucleus and the cytosol.
Even colloids much smaller than the biopolymer-filled NPC channel are effectively blocked, but some larger colloids with distinct surface features rapidly permeate.
Simplistically, one may expect any polymer filling to obstruct and slow down colloid transport.
We demonstrate how a polymer filling that attracts colloids and extends beyond the mesopore, thus maximizing colloid capture, can instead increase permeation compared to a bare pore.
We also define how polymer-filled mesopores can effectively gate colloids according to their size and surface features.
Our findings provide a basic physical explanation for the exquisite permselectivity of NPCs, and a rational design strategy for novel mesopore-based separation, sensing, catalysis and drug delivery devices with enhanced performance features.

**Authors:** Mikhail Y. Laktionov, Frans A. M. Leermakers, Ralf P. Richter, Leonid I. Klushin, Oleg V. Borisov

**Reference:**  (submitted to ...).

## Repository Contents

- **`/src/`**  
  Python modules implementing the methods:
  1. Acquisition of volume fraction profiles from **Namics** output.  
  2.  Acquisition of insertion free energy profiles for cylindrical particles moving along the pore axis from **Namics** output. 
  3. Analytical model fitting to SCFT results.  
  4. Generalization to insertion free energies for spherical particles at arbitrary positions.  
  5. Definition of grids and boundary conditions for the Smoluchowski equation; construction of the matrix operator.  
  6. Solver for the Smoluchowski equation yielding total flux and pore resistance.  

- **`main.ipynb`**  
  Jupyter notebook showcasing the workflow, with explanatory text and figures.  

- **`/SCF/`**  
  Example input files and SCF data to reproduce selected results from the manuscript.  

- **`README.md`**  
  (this file).  

---

## Requirements

To run the code, you will need:

- **Python ≥ 3.9**  
- The following Python packages (see `requirements.txt` in the repo):
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `jupyter`
  - `ipympl`
  - `pandas`

- [**Namics** package](reference) or SFBox is used to calculate equilibrium volume fraction distributions of polymers and other chain molecules in inhomogeneous systems using Self-Consistent Field (SCF) theory

---

## Citation

If you use this code in your research, please cite:

