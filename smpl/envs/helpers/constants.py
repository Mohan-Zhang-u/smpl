import numpy as np

# CONSTANTS

EPS = 1E-20  # very small number to avoid division by zero

NUM_COMPONENTS = 14

# names of components in the system
COMPONENTS = [
    "TROPINE", "DIMETHYLFORMAMIDE (DMF)", "PHENYLACETYLCHLORIDE", "INTERMEDIATE",
    "FORMALDEHYDE", "METHANOL", "SODIUM HYDROXIDE", "WATER", "ATROPINE",
    "APOATROPINE", "TROPINE ESTER", "SODIUM CHLORIDE", "BUFFER_SOLUTION",
    "TOLUENE"
]

# chemical formula of components in the system
COMPONENT_CHEMICAL_FORMULA = [
    "C8H15NO", "C3H7NO", "C8H7ClO", "C16H21O2NHCl", "CH2O", "CH3OH", "NaOH",
    "H2O", "C17H23NO3", "C17H21NO2", "C16H21O2N", "NaCl", "NH4Cl", "C7H8"
]

# mass density of component in the system
MASS_DENSITIES = np.array([
    1020.0, 944.0, 1169.0, 1094.5, 815.0, 792.0, 2130.0, 997.0, 1200.0, 1130.0,
    1100.0, 2160.0, 1530.0, 867.0
], dtype=object)  # /1000.0 # [g/L]

# molecular weight of components in the system
MOLECULAR_WEIGHT = np.array([
    141.21, 73.09, 154.59, 295.80, 30.03, 32.04, 40.00, 18.02, 289.37, 271.35,
    259.34, 58.44, 53.49, 92.14
])  # [g/mol]

# molar density of components in the system
MOLAR_DENSITY = MASS_DENSITIES / MOLECULAR_WEIGHT  # [mol/L]

# reaction rate constants
RATE_CONSTANTS = np.array([
    34206.0, 10000.0, 24.0, 43599.0
]) * 0.001  # [L/mol/min]

# reaction rate activation energies
ACTIVATION_ENERGIES = [
    1000.0, 100.0, 1819.0, 26207.0
]  # [J/mol]

# reaction mechanism for atropine production. the index correspond to the COMPONENT OR CHEMICAL FORMULA INDICES
"""
C0 + C2 ---> C3
C3 + C6 ---> C7 + C10 + C11
C4 + C10 ---> C8
C4 + C10 ---> C7 + C9
"""

# universal gas constant
GAS_CONSTANT = 8.314  # [J/mol/K]

# reaction stoichiometry matrix per reactor section
REACTION_STOICHIOMETRY = [
    [-1, 0, 0, 0],
    [0, 0, 0, 0],
    [-1, 0, 0, 0],
    [1, -1, 0, 0],
    [0, 0, -1, -1],
    [0, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 1, -1, -1],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

# liquid separation partition coefficient
SEPARATION_PARTITION_COEFFICIENTS = np.array([
    1E-10, 1E-10, 1E-10, 1E-10, 1E-10, 1E-10, 1E-10, 1E-10, 0.01, 1E-10, 1E-10, 1E-10, 1E-10, 1E-10
])

# steady states for the inputs
STEADY_STATE_Q_1 = 0.4078
STEADY_STATE_Q_2 = 0.1089
STEADY_STATE_Q_3 = 0.3888
STEADY_STATE_Q_4 = 0.2126
USS = np.array([
    STEADY_STATE_Q_1,
    STEADY_STATE_Q_2,
    STEADY_STATE_Q_3,
    STEADY_STATE_Q_4
])

# reference signals for the inputs
INPUT_REF_1 = 0
INPUT_REF_2 = 0
INPUT_REF_3 = 0
INPUT_REF_4 = 0
INPUT_REFS = [
    INPUT_REF_1,
    INPUT_REF_2,
    INPUT_REF_3,
    INPUT_REF_4
]
OUTPUT_REF = 0

# control system
SIM_TIME = 600
