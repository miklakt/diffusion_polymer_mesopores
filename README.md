# diffusion_polymer_mesopores
Code showcase for modeling diffusion resistance of nanocolloids in polymer-filled mesopores. Uses SCFT-derived insertion free energy and position-dependent diffusion profiles, solved with the Smoluchowski equation on a cylindrical 2D lattice.

This repository provides a complete step-by-step demonstration of how to compute pore resistance for given pore geometries, polymer volume fraction distributions, particle size, and Flory-Huggins interaction parameters (χ).  

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
  2. Acquisition of insertion free energy profiles for cylindrical particles moving along the pore axis from **Namics** output.
  3. Analytical model fitting to SCFT results.  
  4. Generalization to insertion free energies for spherical particles at arbitrary positions.  
  5. Definition of grids and boundary conditions for the Smoluchowski equation; construction of the matrix operator.  
  6. Solver for the Smoluchowski equation yielding total flux and pore resistance.  

- **`main.ipynb`**  
  Jupyter notebook showcasing the workflow, with explanatory text and figures.  

- **`/SCF/`**  
  Example input files and SCF data to reproduce selected results from the manuscript, a dataset of precomputed via SF-SCF insertion free energies for small cylindrical colloids

- **`README.md`**  
  (this file).  

---

## Requirements

To run the code, you will need:

- **Python ≥ 3.9**  
- The following Python packages (see `requirements.txt` in the repo):
  - `numpy`
  - `scipy`
  - `jupyter`
  - `matplotlib`
  - `ipympl`
  - `pandas`
  - `joblib`

- [**Namics** package](https://github.com/leermakers/Namics.git) (or sfbox), which is used to calculate equilibrium volume fraction distributions of polymers and other chain molecules in inhomogeneous systems using Self-Consistent Field (SCF) theory.

---

## Installation

Below are the steps to set up the environment and run the code.  
Namics installation is only briefly mentioned, as compilation is system-specific and outside the scope of this guide.

### 1. Install Namics [Optional]

Download Namics from [GitHub](https://github.com/leermakers/Namics.git).  
You will need to compile it on your system following the instructions provided in the repository.
The installation is optional, if you want to relay on the precomputed results.

### 2. Install Python

- **Linux (e.g., Ubuntu, Debian, Fedora, Arch, …)**  : Python is usually preinstalled.  
- **Windows**: Use [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/) in PowerShell:  
  ```powershell
  winget install Python.Python.3.12
  ```

### 3. Clone this repository

This repository uses **Git LFS** (Large File Storage), so you must install it before cloning.

- **Linux (e.g., Ubuntu, Debian, Fedora, Arch, …)**  :  
  Install Git and Git LFS:  
  ```bash
  sudo apt update
  sudo apt install git git-lfs
  git lfs install
  ```  
  Then clone the repository:  
  ```bash
  git clone https://github.com/miklakt/diffusion_polymer_mesopores.git
  cd diffusion_polymer_mesopores
  ```

- **Windows**:  
  1. Install [Git for Windows](https://git-scm.com/download/win).  
  2. During installation, enable **Git LFS** (or install later with `git lfs install`).  
  3. After installation, run:  
     ```powershell
     git clone https://github.com/miklakt/diffusion_polymer_mesopores.git
     cd diffusion_polymer_mesopores
     ```  
  4. Alternatively, open the folder directly in [Visual Studio Code](https://code.visualstudio.com/).

---

### 4. Create a virtual environment

Run:
```bash
python -m venv .venv
```
Activate it:
- **Linux (e.g., Ubuntu, Debian, Fedora, Arch, …)**  :  
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**:  
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### 5. Install dependencies

With the virtual environment activated, run:
```bash
pip install -r requirements.txt
```

### 6. Run Jupyter Notebook

You can run the main notebook `main.ipynb` either in a browser or in Visual Studio Code:

- **Browser**:  
  ```bash
  jupyter notebook
  ```
  Then open `main.ipynb`.

- **Visual Studio Code**:  
  Install the following VS Code extensions:
  - *Python*
  - *Jupyter*

  Then simply open `main.ipynb` in VS Code.

---

## Citation

If you use this code in your research, please cite: