# CASpy (Chemical Absorption Solver for Python) v0.1.6

For detailed information on the package and an example input file, please refer to:
H. Mert Polat, Frederick de Meyer, Celine Houriez, Othonas A. Moultos and Thijs J.H. Vlugt
**Solving Chemical Absorption Equilibria Using Free Energy and Quantum Chemistry Calculations: Methodology, Limitations, and New Open-Source Software**, Journal of Chemical Theory and Computation, 2023

This package computes the speciation (and the absorption isotherm of the gaseous species) in a gas-liquid absorptive reaction system given the reaction stoichiometry, equilibrium constants of the reactions and/or the chemical potentials of the species.
An example input file can be found at: The supporting information of **Solving Chemical Absorption Equilibria Using Free Energy and Quantum Chemistry Calculations: Methodology, Limitations, and New Open-Source Software**, Journal of Chemical Theory and Computation, 2023

## How to use:

Install the package using the command:
```
pip install CASpy-ReactionEquilibria
```
Go to the directory that contains your input file using:
```
cd /path/to/your/input/file
```
Open a python instance on terminal:
```
python
```
Import the functions used in the library using:
```
from CASpy_ReactionEquilibria import caspy
```
Execute the code with your input file:
```
caspy.main("name_of_your_input_file")
```
The results are printed to a file called "caspy_output.log".
