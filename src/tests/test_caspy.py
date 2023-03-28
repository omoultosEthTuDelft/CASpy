from CASpy_ReactionEquilibria import caspy

def test_results_for_A_B_case():
    """This function tests the results for a basic test case
    mu^0 and mu^ex of A and B = 0.0
    Initial conditions = 20 and 10 mol/dm3 for A and B, respectively
    Reaction --> A <--> B
    lnK for the reaction = 0.0
    The concentrations at equilibrium should be
    A = 15 mol/dm3, B = 15 mol/dm3"""
    caspy.main("src/tests/basic_test_A_B.txt")
    with open("caspy_output.log", 'r', encoding='utf-8') as f:
        data = f.readlines()
        for i in data:
            if "Solution =" in i:
                solution = i
    assert float(solution.split()[3]) == 15, 'Test for the basic case failed!'

def test_computed_lnK():
    """This function tests the computed lnK values
    Parameters:
    mu^0_A  = -5 kJ/mol
    mu^ex_A = -0.32 kJ/mol
    mu^0_B  = -7 kJ/mol
    mu^ex_B = -0.93 kJ/mol
    The computed lnK should be = 1.002"""
    caspy.main("src/tests/basic_test_A_B_lnK.txt")
    with open("caspy_output.log", 'r', encoding='utf-8') as f:
        data = f.readlines()
        for i in data:
            if "ln(K_des) =" in i:
                solution = i
    assert float(solution.split()[2].split('[')[1]) == 1.002, 'Test for the computed lnK failed!'
