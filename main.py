# Developed by H. Mert Polat, Frederick de Meyer, Celine Houriez, Othonas A. Moultos and Thijs J. H. Vlugt
# Delft,2022
# Please cite: H.M. Polat, F. de Meyer, C. Houriez, O.A. Moultos and T.J.H. Vlugt, 2023, 
# Solving Chemical Reaction Equilibrium Using Free Energy and Quantum Chemistry Calculations: Modeling and Limitations,
# Journal of Chemical Theory and Computation.
import numpy as np
# import the required functions from functions.py
from functions import read_input, calc_K0, calc_speciation, calc_speciation_imposePtotal


# Define constants
kB, NA = 1.38064852e-23, 6.022140857e+23
R = kB * NA
# The default volume is 1 dm3 (=1e-3 m3), this can be changed
# But concentrations should be adjusted accordingly.
V = 1e-3 # Volume in m3 (default=1e-3 m3)

# The function read_input(name_of_the_input_file) reads the input data
# The input file should be in the same folder with the solver
T,Nspecies,x0,names,charges,mu_0,mu_ex,gas_names,muex_gas, \
    cgas,Nreactions,stoichiometry,Nbalances,balances,lnK,compute_Kdes, \
        solvent_index,pure_density,Ngas_species,impose_Ptotal, \
            Ptotal,gas_composition = read_input("general_solver_mdea_lnk.txt")

# Compute initial amounts for balances
# Gases are treated differently since the concentration of gas species
# change in our calculations
if impose_Ptotal:
    init_amounts = np.zeros((Nbalances+Ngas_species,len(Ptotal)))
else:
    init_amounts = np.zeros((Nbalances, len(cgas[0,:])))

gas_indices = []
for i in gas_names:
    gas_indices.append(names.index(i))

if impose_Ptotal:
    for i in range(Nbalances):
        for j in range(len(Ptotal)):
            for k in range(len(balances[i])):
                init_amounts[i][j] = init_amounts[i][j] + x0[balances[i][k]]
    for i in range(Nbalances,Nbalances+Ngas_species):
        for j in range(len(Ptotal)):
            init_amounts[i][j] = (Ptotal[j]*gas_composition[i-Nbalances]) * V * \
                np.exp(-muex_gas[i-Nbalances]/(R*T)) / (R*T)
        
else:
    for i in range(Nbalances):
            for j in range(len(cgas[0,:])):
                for k in range(len(balances[i])):
                    if balances[i][k] in gas_indices:
                        ind = gas_names.index(names[balances[i][k]])
                        init_amounts[i][j] = init_amounts[i][j] + cgas[ind,j]
                    else:
                        init_amounts[i][j] = init_amounts[i][j] + x0[balances[i][k]]


# The function calc_K0 computes the desired equilibrium constants of the reactions in the system
K0 = calc_K0(T,lnK,x0,pure_density,stoichiometry,Nreactions,mu_0,mu_ex,V,solvent_index, compute_Kdes) # compute desired equilibrium constants
lnK0 = np.log(K0)
# The function calc_speciation computes the speciation as a function of gas concentration
if impose_Ptotal:
    spec = calc_speciation_imposePtotal(T, x0, K0, Nbalances, balances, init_amounts, 
                                        Nreactions, stoichiometry, cgas, gas_indices, gas_names, charges,  
                                        solvent_index, compute_Kdes, lnK, pure_density, mu_0, mu_ex, 
                                        V, names,Ptotal)
else:
    spec = calc_speciation(T, x0, K0, Nbalances, balances, init_amounts, Nreactions, 
                           stoichiometry, cgas, gas_names, charges, solvent_index, compute_Kdes,
                           lnK, pure_density, mu_0, mu_ex, V, names)
 

# This part prints the computed gas pressure as a function of gas loading
if impose_Ptotal:
    print("SPECIATION")
    print("Ptotal", Ptotal/1000, '/ [kPa]')
    for i in range(len(spec[:,0])):
        print(names[i],spec[i,:],'/ [mol/dm3]')
else:
    print("ISOTHERM")
    for i in range(len(gas_names)):
        print('C_%s / [mol/dm3], P_%s / [kPa]' % (gas_names[i], gas_names[i]))
        for j in range(len(cgas[i,:])):
            p_gas = (spec[names.index(gas_names[i]), j] * T * R) / (np.exp(-muex_gas[i] / (R * T)) / 1000) / 1000 #Compute gas partial pressure
            print('%.2e %.5e' % (cgas[i,j]/init_amounts[1,j], p_gas))
        
