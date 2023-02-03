# Developed by H. Mert Polat, Frederick de Meyer, Celine Houriez, Othonas A. Moultos and Thijs J. H. Vlugt
# Delft,2022
# Please cite: H.M. Polat, F. de Meyer, C. Houriez, O.A. Moultos and T.J.H. Vlugt, 2023, 
# Solving Chemical Reaction Equilibrium Using Free Energy and Quantum Chemistry Calculations: Modeling and Limitations,
# Journal of Chemical Theory and Computation.

# The functions.py file is a module that contains some functions required by the solver
# These functions are: read_input, compute_mole_fractions, sum_number_of_moles, sum_stoichiometric_coeffs,
# calc_K0, calc_speciation, objective_function, and write_output.
# read_input(): reads the input data from the input file
# The input file should be in the same folder with the solver
# compute_mole_fractions(): it computes the molefractions of the species given a list of concentrations.
# sum_number_of_moles():  it computes the sum of concentations of the species given a list of concentrations. 
# sum_stoichiometric_coeffs(): it computes the sum of the stoichiometric coeffs of solutes in a reaction.
# calc_K0(): Computes the desired equilibrium constants of the reactions.
# calc_speciation(): Computes speciation of the system given the input data.
# objective_function(): Computes the values in the objective function. The objective function is a list of:
# [(ln(K_act)-ln(K_des))/ln(K_des) for all reactions in the system,
#  (sum(N_initial_amounts)-sum(N_actual_amounts))/sum(N_initial_amounts) for all balances in the system,
#  net_charge/total_amount_of_charges for charge neutrality]
# write_output(): constructs a string to output. This is used by calc_speciation() to output data.

import numpy as np
from scipy.optimize import least_squares

 
def read_input(file_name):
    # Read the whole input_file
    input_file = open(file_name)
    input_data = input_file.readlines()
    input_file.close()
    
    # Define Avogadro's constant
    NA = 6.022140857e+23
    
    # Read input
    # the main for loop goes through all the lines in the input file.
    # In each line, we look for the string in the if statements.
    # If the string is in the line, we convert to integer of float if necessary and save the value
    for line in input_data:
        if "Temperature" in line:
            index = input_data.index(line)
            T = float(input_data[index+1])  # Temperature
        if "Number of Species" in line:
            index = input_data.index(line)
            Nspecies = int(input_data[index+1]) # Number of species
        if "C0" in line:
            index = input_data.index(line)
            x0 = np.zeros(Nspecies) # The initial guess in mol/dm3
            for i in range(Nspecies):
                if float(input_data[index+1].split()[i]) <= 0:
                    x0[i] = 1e-10
                else:
                    x0[i] = float(input_data[index+1].split()[i])
        if "Names of species" in line:
            index = input_data.index(line)
            names = [] # Names of the species
            for i in range(Nspecies):
               names.append(input_data[index+1].split()[i]) 
        if "Charges" in line:
            index = input_data.index(line)
            charges = np.zeros(Nspecies) # Net charges of the species, follows the indices in the the list of names
            for i in range(Nspecies):
                charges[i] = int(input_data[index+1].split()[i])
        if "mu^0 species" in line:
            index = input_data.index(line)
            mu_0 = np.zeros(Nspecies) # The ideal gas standard chemical potentials of the species in J/mol
            for i in range(Nspecies):
                mu_0[i] = float(input_data[index+1].split()[i]) * 1e3 # convert kJ/mol to J/mol and save the value
        if "mu^ex species" in line:
            index = input_data.index(line)
            mu_ex = np.zeros(Nspecies) # The excess chemical potentials of the species in J/mol
            for i in range(Nspecies):
                mu_ex[i] = float(input_data[index+1].split()[i]) * 1e3 # convert kJ/mol to J/mol and save the value
        if "Impose Ptotal and gas composition?" in line:
            index = input_data.index(line)
            true_answers = ['true','t']
            if input_data[index+1].split()[0].lower() in true_answers:
                impose_Ptotal = True
            else:
                impose_Ptotal = False
        if "Ptotal / [kPa]" in line:
            index = input_data.index(line)
            Ptotal = np.zeros(len(input_data[index+1].split()))
            for i in range(len(input_data[index+1].split())):
                Ptotal[i] = float(input_data[index+1].split()[i]) * 1e3
        if "Gas phase species" in line:
            index = input_data.index(line)
            Ngas = len(input_data[index+1].split())  # Number of species in the gas phase
            gas_names = [] # Names of the species in the gas phase
            for i in range(Ngas):
                gas_names.append(input_data[index+1].split()[i])
        if "Gas phase composition" in line:
            index = input_data.index(line)
            gas_composition = np.zeros(Ngas)
            for i in range(Ngas):
                gas_composition[i] = float(input_data[index+1].split()[i])
            # Normalize gas phase composition so the total mole fraction is 1.
            sum_gas_molefraction = sum(gas_composition)
            gas_composition = np.divide(gas_composition,sum_gas_molefraction)
        if "mu^ex gases" in line:
            index = input_data.index(line)
            muex_gas = np.zeros(Ngas) # The excess chemical potentials of the gas species in solvent at T in J/mol.
            for i in range(Ngas):
                muex_gas[i] = float(input_data[index+1].split()[i]) * 1e3 # convert kJ/mol to J/mol
        if "Ctotal,gas" in line:
            index = input_data.index(line)
            cgas = np.zeros((Ngas,len(input_data[index+1].split()))) # The concentrations of the gas species. This is a list.
            # The main loop of the solver uses this list. 
            # This means the speciation is computed as a function of the concentaration of the gas species in the solution.
            for i in range(Ngas):
                for j in range(len(cgas[i,:])):
                    cgas[i,j] = float(input_data[index+i+1].split()[j])
        if "Number of Reactions" in line:
            index = input_data.index(line)
            Nreactions = int(input_data[index+1]) # Number of reactions
        if "Stoichiometry" in line:
            index = input_data.index(line)
            stoichiometry = np.zeros((Nreactions,Nspecies)) # The matrix that contains the stoichiometry.
            for i in range(Nreactions):
                for j in range(Nspecies):
                    stoichiometry[i,j] = int(input_data[index+1+i].split()[j])
        if "Number of mass balance" in line:
            index = input_data.index(line)
            Nbalances = int(input_data[index+1]) # Number of mass balances in the system
        if "Balances" in line:
            index = input_data.index(line)
            balances = [[]] * Nbalances  # This list contains lists of the species included in each balance equation.
            for i in range(Nbalances):
                balances[i] = []
                for j in range(len(input_data[index+1+i].split())):
                    balances[i].append(int(input_data[index+1+i].split()[j])-1)
        if 'ln(K)' in line:
            index = input_data.index(line)
            lnK = np.zeros(Nreactions)  # this is list contains the ln(K) of each reaction. 
            compute_Kdes = np.full(Nreactions, False) # Create a list of booleans for all reactions (compute_Kdes)
            # If the boolean is True, then compute Kdes for that reaction using the derived expression (in calc_K0())
            # for the molefraction-based equilibrium constants
            # compute_K_des[i] is true if the input is:
            key = 'QMMC'
            for i in range(Nreactions):
                if input_data[index+1].split()[i]==key:
                    lnK[i] = int(0.0)
                    compute_Kdes[i] = True
                else:
                    lnK[i] = float(input_data[index+1].split()[i])
        if "Name of the solvent" in line:
            index = input_data.index(line)
            solvent_index = names.index(input_data[index+1].split('\n')[0])
        if "Pure Density" in line:
            index = input_data.index(line)
            pure_density = float(input_data[index+1]) * NA / 1e27 # convert from mol/dm3 to molecules per cubic angstrom
    return T,Nspecies,x0,names,charges,mu_0,mu_ex,gas_names,muex_gas, \
        cgas,Nreactions,stoichiometry,Nbalances,balances,lnK,compute_Kdes, \
        solvent_index,pure_density,Ngas,impose_Ptotal,Ptotal,gas_composition


def compute_mole_fractions(x):
    mole_fractions = np.divide(x, np.sum(x))
    return mole_fractions

def compute_mole_fractions_imposePtotal(x,Ngas,Nbalances,init_amounts,i):
    for j in range(Ngas):
        x = np.append(x,init_amounts[Nbalances+j,i])
    mole_fractions = np.divide(x, np.sum(x))
    return mole_fractions

def sum_number_of_moles(x):
    return sum(x)

def sum_stoichiometric_coeffs(stoichiometry, reaction_index,solvent_index):
    #prepare a list of indices without the solvent index
    list_indices = list(range(len(stoichiometry[reaction_index])))
    list_indices.remove(solvent_index)
    sum_of_coeffs = 0
    for i in list_indices:
        sum_of_coeffs = sum_of_coeffs + stoichiometry[reaction_index][i] # sum the stoichiometric coefficients of the solutes
    return sum_of_coeffs
    

def calc_K0(T,lnK,x0,pure_density,stoichiometry,Nreactions,mu_0,mu_ex,V,index_solvent,compute_Kdes):
    # Define constants
    kB, NA = 1.38064852e-23, 6.022140857e+23
    R = NA * kB
    K0 = np.zeros(Nreactions) # This list contains the desired equilibrium constant of each reaction. 
    V = V * 1e30 # Convert volume from m3 to A3
    for i in range(Nreactions):
        # If compute_Kdes[i] is False for the reaction, then
        # K0[i] is computed from the inputted ln(K)
        if compute_Kdes[i]==False:
            K0[i] = np.exp(lnK[i])
        # If compute_Kdes[i] is True for the reaction, then
        # K0[i] is computed using the expression we derived for the molefraction-based equilibrium constants
        else:
            sum_mu = 0
            for j in range(len(x0)):
                sum_mu = sum_mu + (stoichiometry[i][j] * (mu_0[j]+ mu_ex[j]))/(R*T)
            solvent_term = stoichiometry[i][index_solvent] * np.log(pure_density)
            xs = compute_mole_fractions(x0)[index_solvent]
            sum_number_of_molecules = sum_number_of_moles(x0) * NA
            K0[i] = np.exp(-(sum_mu + solvent_term) + (stoichiometry[i][index_solvent]*((1-xs)/xs))) * (xs**stoichiometry[i][index_solvent]) \
                    * (V/sum_number_of_molecules)**sum_stoichiometric_coeffs(stoichiometry,i,index_solvent)
    return K0


def calc_speciation(T, x0, K0, Nbalances, balances, init_amounts, 
                    Nreactions, stoichiometry, cgas, gas_names, charges,  
                    solvent_index, compute_Kdes, lnK, pure_density, mu_0, mu_ex, 
                    V, names):
    spec = np.zeros((len(x0), len(init_amounts[0,:]))) # This list contains the speciation in mol/dm3 as a function of gas loading.
    output_file = open('output.log', 'w') # Open the output.log file
    # Print a bunch of stuff before the calculations start
    output_file.write(write_output(2,None,None,None,None,None,None,
                                   None,None,names,Nreactions,Nbalances,None,solvent_index,
                                   pure_density,stoichiometry,balances,charges,T,V,False,None))
    # Numerical solution starts here
    for i in range(len(init_amounts[0,:])):
        residual = np.ones(2)
        solver_counter = 0 
        diff_Xs = 1
        K0_used = np.zeros(Nreactions)
        # if this is not the first data point computed,
        # use the solution from the previous data point point as the initial guess
        if i > 0: 
            x0 = spec[:,i-1]
        # Solver works until it finds a good solution
        # A good solution is found if max of abs(objective_function) < 1e-10 and
        # the difference between the old molefraction of the solvent and the new molefraction
        # of the solvent < 1e-3
        while max(abs(residual)) > 1e-10 or abs(diff_Xs) > 1e-3:
            Xs_old = compute_mole_fractions(x0)[solvent_index] # Compute the old molefraction of the solvent
            xScalingVec = x0
            x0 = np.divide(x0, xScalingVec) # The initial guess is scaled to unity and the scaling factor is saved to memory.
            # paramVec = [K0's for all reactions,initial amounts of species for balances]
            paramVec = np.zeros(Nreactions+Nbalances)
            for j in range(Nreactions): # Reactions
                paramVec[j] = K0[j]
                K0_used[j] = K0[j]
            for j in range(Nreactions,len(paramVec)): # Balances
                n = j - Nreactions
                paramVec[j] = init_amounts[n,i]
            # the "solution" variable runs the solver and saves the results
            # bounds are between 0 and inf, so the concentrations are never negative.
            solution = least_squares(objective_function, x0, bounds=(0.0, np.inf), method='trf', ftol=1e-15, 
                                     xtol=1e-15, gtol=1e-15, max_nfev=1000, verbose=0, 
                                     args=(paramVec,xScalingVec,Nreactions,Nbalances,stoichiometry,balances,charges))
            residual = solution.fun # the values of the objective function for this solution
            # Evaluate the solution
            x = solution.x # The solution
            x = np.multiply(x, xScalingVec) # Scale it back, using the scaling factor
            
            mole_fractions = compute_mole_fractions(x)  # Compute mole fractions
            Xs_new = mole_fractions[solvent_index] # compute the new molefraction of the solvent
            # Evaluate the solution by computing actual K for the reactions
            F_act = np.ones(Nreactions)
            for j in range(Nreactions):
                for k in range(len(mole_fractions)):
                    F_act[j] = F_act[j] * (mole_fractions[k]**stoichiometry[j][k])
                F_act[j] = np.log(F_act[j])

            diff_Xs = Xs_old-Xs_new # compute the difference between old solvent mole fraction and new solvent mole fraction (tolerance=1e-3)
            # compute K_des again for the new solution (because Xs changes)
            K0 = calc_K0(T, lnK, x, pure_density, stoichiometry, Nreactions, mu_0, mu_ex, V, solvent_index, compute_Kdes) 
            x0 = x
            solver_counter = solver_counter + 1 # Count the trials before a good solution (sum(objective_function)=0)
        output_file.write(write_output(1,solver_counter,residual,diff_Xs,x,K0_used,cgas,
                                       i,gas_names,names,Nreactions,Nbalances,F_act,None,
                                       None,None,None,None,None,None,False,None)) # write output for this data point
        # Store the speciation
        spec[:,i] = x # save the speciation
    output_file.close() # close the output file after all calculations are done.
    return spec

def calc_speciation_imposePtotal(T, x0, K0, Nbalances, balances, init_amounts, 
                                 Nreactions, stoichiometry, cgas, gas_indices, gas_names, charges,  
                                 solvent_index, compute_Kdes, lnK, pure_density, mu_0, mu_ex, 
                                 V, names,Ptotal):
    N_gasspecies = len(gas_indices)
    solvent_name = names[solvent_index]
    names_copy = names.copy()
    charges_copy = charges.copy()
    for i in reversed(gas_indices):
        x0 = np.delete(x0,i)
        names.pop(i)
        charges = np.delete(charges,i)
    for i in range(N_gasspecies):
        names.append(gas_names[i])
    for i in gas_indices:
        charges = np.append(charges,charges_copy[i])
    for i in range(Nbalances):
        for j in range(len(balances[i])):
            balances[i][j] = names.index(names_copy[balances[i][j]])
    for i in range(Nreactions):
        append = []
        a = stoichiometry[i]
        a = np.delete(a, gas_indices)
        for j in gas_indices:
            append.append(stoichiometry[i][j])
        a = np.append(a, append)
        stoichiometry[i] = a
    solvent_index = names.index(solvent_name)
    spec = np.zeros((len(x0)+2, len(init_amounts[0,:]))) # This list contains the speciation in mol/dm3 as a function of gas loading.
    output_file = open('output.log', 'w') # Open the output.log file
    # Print a bunch of stuff before the calculations start
    output_file.write(write_output(2,None,None,None,None,None,None,
                                   None,None,names,Nreactions,Nbalances,None,solvent_index,
                                   pure_density,stoichiometry,balances,charges,T,V,True,None))
    # Numerical solution starts here
    for i in range(len(init_amounts[0,:])):
        residual = np.ones(2)
        solver_counter = 0 
        K0_used = np.zeros(Nreactions)
        # if this is not the first data point computed,
        # use the solution from the previous data point point as the initial guess
        if i > 0: 
            x0 = spec[:,i-1]
            for j in reversed(gas_indices):
                x0 = np.delete(x0,j)
        # Solver works until it finds a good solution
        # A good solution is found if max of abs(objective_function) < 1e-10 and
        # the difference between the old molefraction of the solvent and the new molefraction
        # of the solvent < 1e-3
        while max(abs(residual)) > 1e-10:
            if solver_counter > 0:
                for j in reversed(gas_indices):
                    x0 = np.delete(x0,j)
            xScalingVec = x0
            x0 = np.divide(x0, xScalingVec) # The initial guess is scaled to unity and the scaling factor is saved to memory.
            # paramVec = [K0's for all reactions,initial amounts of species for balances]
            paramVec = np.zeros(Nreactions+Nbalances+N_gasspecies)
            for j in range(Nreactions): # Reactions
                paramVec[j] = K0[j]
                K0_used[j] = K0[j]
            for j in range(Nreactions,len(paramVec)): # Balances
                n = j - Nreactions
                paramVec[j] = init_amounts[n,i]
            # the "solution" variable runs the solver and saves the results
            # bounds are between 0 and inf, so the concentrations are always positive.
            solution = least_squares(objective_function_imposePtotal, x0, bounds=(0.0, np.inf), method='trf', ftol=1e-15, 
                                     xtol=1e-15, gtol=1e-15, max_nfev=1000, verbose=0, 
                                     args=(paramVec,xScalingVec,Nreactions,Nbalances,stoichiometry,
                                           balances,charges,N_gasspecies,init_amounts,i))
            residual = solution.fun # the values of the objective function for this solution
            # Evaluate the solution
            x = solution.x # The solution
            x = np.multiply(x, xScalingVec) # Scale it back, using the scaling factor
            for j in range(len(gas_indices)):
                x = np.append(x, init_amounts[Nbalances+j,i])
            mole_fractions = compute_mole_fractions(x)  # Compute mole fractions
            # Evaluate the solution by computing actual K for the reactions
            F_act = np.ones(Nreactions)
            for j in range(Nreactions):
                for k in range(len(mole_fractions)):
                    F_act[j] = F_act[j] * (mole_fractions[k]**stoichiometry[j][k])
                F_act[j] = np.log(F_act[j])

            # compute K_des again for the new solution (because Xs changes)
            K0 = calc_K0(T, lnK, x, pure_density, stoichiometry, Nreactions, mu_0, mu_ex, V, solvent_index, compute_Kdes)
            x0 = x
            solver_counter = solver_counter + 1 # Count the trials before a good solution (sum(objective_function)=0)
        output_file.write(write_output(1,solver_counter,residual,0.0,x,K0_used,cgas,
                                       i,gas_names,names,Nreactions,Nbalances,F_act,None,
                                       None,None,None,None,None,None,True,Ptotal[i])) # write output for this data point
        # Store the speciation
        spec[:,i] = x # save the speciation
    output_file.close() # close the output file after all calculations are done.
    return spec

def objective_function(x, paramVec, xScalingVec,Nreactions,Nbalances,stoichiometry,balances,charges):
    # Initialize variables
    F_act = np.ones(Nreactions)
    F = np.ones(Nreactions+Nbalances+1) # The list that contains the values of the objective function
    # Scale back the speciation array
    x = np.multiply(x, xScalingVec)
    mole_fractions = compute_mole_fractions(x) # For molefraction-based equilibrium constants 
    for i in range(Nreactions):  # Compute actual equilibrium constants for each reaction
        for j in range(len(x)):
            F_act[i] = F_act[i] * (mole_fractions[j]**stoichiometry[i][j])
        F_act[i] = np.log(F_act[i])
    actual_amounts = np.zeros(Nbalances) # compute actual amount of species for each mass balance equation
    for i in range(Nbalances):
        for j in balances[i]:
            actual_amounts[i] = actual_amounts[i] + x[j]
    # Reaction balances
    for i in range(Nreactions):
        F[i] = (np.log(paramVec[i]) - F_act[i]) / np.log(paramVec[i])
    # Mass balances
    for i in range(Nreactions,Nreactions+Nbalances):
        F[i] = (paramVec[i] - actual_amounts[i-Nreactions]) / paramVec[i]
    # Charge balance
    tot_charges = 0
    net_charge = 0
    for i in range(len(x)):
        tot_charges = tot_charges + x[i]*abs(charges[i])
        net_charge = net_charge + x[i]*charges[i]
    F[-1] = net_charge/tot_charges # charge neutrality
    return F

def objective_function_imposePtotal(x, paramVec, xScalingVec,Nreactions,Nbalances,stoichiometry,balances,charges,Ngas,init_amounts,data_point):
    # Initialize variables
    F_act = np.ones(Nreactions)
    F = np.ones(Nreactions+Nbalances+1) # The list that contains the values of the objective function
    # Scale back the speciation array
    x = np.multiply(x, xScalingVec)
    mole_fractions = compute_mole_fractions_imposePtotal(x,Ngas,Nbalances,init_amounts,data_point) # For molefraction-based equilibrium constants 
    for i in range(Ngas):
        x = np.append(x, init_amounts[Nbalances+i,data_point])
    for i in range(Nreactions):  # Compute actual equilibrium constants for each reaction
        for j in range(len(mole_fractions)):
            F_act[i] = F_act[i] * (mole_fractions[j]**stoichiometry[i][j])
        F_act[i] = np.log(F_act[i])
    actual_amounts = np.zeros(Nbalances) # compute actual amount of species for each mass balance equation
    for i in range(Nbalances):
        for j in balances[i]:
            actual_amounts[i] = actual_amounts[i] + x[j]
    # Reaction balances
    for i in range(Nreactions):
        F[i] = (np.log(paramVec[i]) - F_act[i]) / np.log(paramVec[i])
    # Mass balances
    for i in range(Nreactions,Nreactions+Nbalances):
        F[i] = (paramVec[i] - actual_amounts[i-Nreactions]) / paramVec[i]
    # Charge balance
    tot_charges = 0
    net_charge = 0
    for i in range(len(x)):
        tot_charges = tot_charges + x[i]*abs(charges[i])
        net_charge = net_charge + x[i]*charges[i]
    F[-1] = net_charge/tot_charges # charge neutrality
    return F

def write_output(choice,solver_counter,residual,diff_Xs,x,K0_used,cgas,i,
                 gas_names,names,Nreactions,Nbalances,F_act,solvent_index,
                 pure_density,stoichiometry,balances,charges,T,V,impose_Ptotal,Ptotal):
    # prepare a string to print to output.log file. a seperate function is used for this
    # to ensure the function calc_speciation() does not look cluttered and only composed for
    # the numerical solution
    if choice == 1: # choice = 1 to print solution after every data point
        mole_fractions = compute_mole_fractions(x)
        string_to_print = '------------------------------------------------Solution %d------------------------------------------------\n' % (i+1)
        if impose_Ptotal:
            string_to_print = string_to_print + "Ptotal = %.3e kPa \n" % Ptotal
        else:
            for j in range(len(gas_names)):
                string_to_print = string_to_print + "C_%s = %.3e mol/dm3 \n" % (gas_names[j],cgas[j,i])
        string_to_print = string_to_print + \
            "SOLUTION FOUND SUCCESSFULLY! [%d] iterations \nMaximum value in the objective function = [%.3e]\n" % (solver_counter, max(abs(residual))) + \
             'Names of the species: ['
        for j in names:
            string_to_print = string_to_print + "%s " % j
        string_to_print = string_to_print + '] \n' + 'Solution = ['
        for j in x:
            string_to_print = string_to_print + '%.3e ' % j
        string_to_print = string_to_print + '] / [mol/dm3]\n' + 'Mole fractions = [' 
        for j in mole_fractions:
            string_to_print = string_to_print + '%.3e ' % j
        string_to_print = string_to_print + ']\n' + 'ln(K_des) = ['
        for j in np.log(K0_used):
            string_to_print = string_to_print + '%.3f ' % j
        string_to_print = string_to_print + ']\n' + 'ln(K_act) = ['
        for j in F_act:
            string_to_print = string_to_print + '%.3f ' % j
        string_to_print = string_to_print + ']\n' + 'The values of mass balance equations and charge neutrality = ['
        for j in range(Nreactions,Nreactions+Nbalances):
            string_to_print = string_to_print + '%.3e ' % residual[j]
        string_to_print = string_to_print + "%.3e ]\n\n" % residual[-1]
    elif choice == 2: # choice = 2 to print initial conditions
        NA = 6.022140857e+23
        string_to_print = 'The chemical reaction equilibrium solver \n' + \
            "Developed by H. Mert Polat, Frederick de Meyer, Celine Houriez, Othonas A. Moultos and Thijs J. H. Vlugt \n" + \
            "Please cite:  H.M. Polat, F. de Meyer, C. Houriez, O.A. Moultos and T.J.H. Vlugt, 2023, \n" + \
            "Solving Chemical Reaction Equilibrium Using Free Energy and Quantum Chemistry Calculations: Modeling and Limitations,\n" + \
            "Journal of Chemical Theory and Computation.\n\n" + \
            "--------------------------------------------Initial conditions--------------------------------------------\n" + \
            "T = %.2f K, V = %.0e m3, Number of Species in liquid phase = %d\n\n" % (T,V,len(names)) + \
            "Solvent: %s, Pure Density of the Solvent = %.2f [mol/dm3]\n\n" % (names[solvent_index],pure_density*1e27/NA) + \
            "Number of reactions = %d\nReaction stoichiometry:\n" % Nreactions
        for x in range(Nreactions):
            string_to_print = string_to_print + "Reaction %d: " % (x+1)
            negative_indices = []
            positive_indices = []
            for y in range(len(names)):
                if stoichiometry[x,y] < 0:
                    negative_indices.append(y)
                if stoichiometry[x,y] > 0:
                    positive_indices.append(y)
            for y in negative_indices:
                if y == negative_indices[-1]:
                    string_to_print = string_to_print + '%d %s ' % (abs(stoichiometry[x,y]),names[y])
                else:
                    string_to_print = string_to_print + "%d %s + " % (abs(stoichiometry[x,y]),names[y])
            string_to_print = string_to_print + "<---> "
            for y in positive_indices:
                if y == positive_indices[-1]:
                    string_to_print = string_to_print + "%d %s " % (stoichiometry[x,y],names[y])
                else:
                    string_to_print = string_to_print + "%d %s + " % (stoichiometry[x,y],names[y])
            string_to_print = string_to_print + "\n"
        string_to_print = string_to_print + "\nNumber of mass balance equations = %d\nSpecies included in the mass balance equations:\n" % Nbalances
        for x in range(Nbalances):
            string_to_print = string_to_print + "Equation %d: " % (x+1)
            for y in range(len(balances[x])):
                string_to_print = string_to_print + "%s " % names[balances[x][y]]
            string_to_print = string_to_print + "\n"
        string_to_print = string_to_print + "\nThe net charges of the molecules: \n"
        for x in range(len(names)):
            string_to_print = string_to_print + "%s:%d " % (names[x],charges[x])
        string_to_print = string_to_print + "\n--------------------------------------------Initial conditions--------------------------------------------\n\n"
    return string_to_print