import numpy as np
from casadi import reshape, vertcat, SX, integrator, mtimes, inf, nlpsol

from ..helpers.constants import NUM_COMPONENTS, MASS_DENSITIES, MOLECULAR_WEIGHT, EPS, REACTION_STOICHIOMETRY, \
    RATE_CONSTANTS, ACTIVATION_ENERGIES, GAS_CONSTANT, SEPARATION_PARTITION_COEFFICIENTS


class Stream:

    def __init__(self):
        self.total_volumetric_flowrate = 0.0  # [L/min]
        self.total_mass_flowrate = 0.0  # [g/min]
        self.temperature = 0.0  # [K]
        self.component_mass_flowrate = np.zeros(NUM_COMPONENTS, dtype=object)  # [g/min]
        self.component_volumetric_flowrate = np.zeros(NUM_COMPONENTS, dtype=object)  # [L/min]
        self.component_molar_concentration = np.zeros(NUM_COMPONENTS, dtype=object)  # [mol/L]
        self.component_mass_concentration = np.zeros(NUM_COMPONENTS, dtype=object)  # [g/L]
        self.component_mass_fraction = np.zeros(NUM_COMPONENTS, dtype=object)  # [dimensionless]
        self.average_mass_density = 0.0  # [g/L]

    def component_volumetric_flowrate_to_component_mass_flowrate(self):
        self.component_mass_flowrate = (self.component_volumetric_flowrate * MASS_DENSITIES)[:]  # [g/min]

    def evaluate_total_volumetric_flowrate(self):
        self.total_volumetric_flowrate = self.total_mass_flowrate / (self.average_mass_density + EPS)  # [L/min]

    def evaluate_average_mass_density(self):
        self.average_mass_density = 1.0 / np.sum(self.component_mass_fraction / MASS_DENSITIES)  # [g/L]

    def evaluate_total_mass_flowrate(self):
        self.total_mass_flowrate = np.sum(self.component_mass_flowrate)  # [g/min]

    def evaluate_component_molar_concentration(self):
        self.component_molar_concentration = self.component_mass_fraction \
                                             * self.average_mass_density \
                                             / MOLECULAR_WEIGHT  # [mol/L]

    def component_molar_concentration_to_component_mass_concentration(self):
        self.component_mass_concentration = self.component_molar_concentration * MOLECULAR_WEIGHT  # [g/L]

    def component_mass_concentration_to_component_mass_flowrate(self):
        self.component_mass_flowrate = (self.component_mass_concentration *
                                        self.total_volumetric_flowrate)[:]  # [g/min]

    def evaluate_component_mass_fraction(self):
        self.component_mass_fraction = self.component_mass_flowrate / (
                np.sum(self.component_mass_flowrate) + EPS)  # [dimensionless]

    def total_volumetric_flowrate_to_total_mass_flowrate(self):
        self.total_mass_flowrate = self.average_mass_density * self.total_volumetric_flowrate  # [g/min]

    def total_mass_flowrate_to_component_mass_flowrate(self):
        self.component_mass_flowrate = self.component_mass_fraction * self.total_mass_flowrate  # [g/min]

    def print_stream(self):
        print(f'Total volumetric flowrate: {self.total_volumetric_flowrate} mL/min')
        print(f'Total mass flowrate: {self.total_mass_flowrate} g/min')
        print(f'Component mass flowrate: {self.component_mass_flowrate} g/min')
        print(f'Component volumetric flowrate: {self.component_volumetric_flowrate} g/min')
        print(f'Component molar concentration: {self.component_molar_concentration} mol/mL')
        print(f'Component mass concentration: {self.component_mass_concentration} g/mL')
        print(f'Component mass fraction: {self.component_mass_fraction}')
        print(f'Average mass density: {self.average_mass_density} g/mL')


# mixer model
class Mixer:

    def __init__(
            self,
            ID: int,
            num_inputs: int
    ):
        self.ID = ID  # equipment ID. May be used for diagnostics. Currently not in use
        self.num_inputs = num_inputs  # number of inputs to the mixer

        # initialize streams
        self.input = [Stream() for _ in range(self.num_inputs)]  # input streams
        self.output = Stream()  # single output stream

    def mix_streams_mass(self):
        component_mass_flowrate = np.zeros(NUM_COMPONENTS, dtype=object)  # []
        # add the streams
        for stream in self.input:
            component_mass_flowrate += stream.component_mass_flowrate  # []

        self.output.total_mass_flowrate = np.sum(component_mass_flowrate)
        self.output.component_mass_flowrate = component_mass_flowrate[:]


# tubular reactor model
class TubularReactor:

    def __init__(
            self,
            ID: int,
            volume: float = 0.3,
            num_discretization_points: int = 5
    ):
        """diameter in mm"""
        self.ID = ID  # equipment ID. May be used for diagnostics. Currently not in use
        self.num_discretization_points = num_discretization_points  # [dimensionless]
        self.volume = volume  # [L]
        self.volume_per_section = self.volume / num_discretization_points  # [L]

        self.input = Stream()
        self.output = Stream()

    # xdot for the reactor
    def get_derivative(self, x, u, d):
        # x = states
        # u = inputs
        # d = disturbances [currently unused]

        # reshape array. currently using casadi reshape
        # function since the numpy reshape does not work for casadi variables
        x = reshape(x, (NUM_COMPONENTS, self.num_discretization_points))
        # create holder for state velocities. object datatype to allow for casadi variables
        xdot = np.zeros((NUM_COMPONENTS, self.num_discretization_points), dtype=object)

        T = u[1]  # reactor temperature
        Qtot = u[0]  # total volumetric flow rate [L/min]

        # start at 2nd discrete point because xdot[:,0] = 0
        for l in range(1, self.num_discretization_points):
            # Eq. (7)
            R1 = RATE_CONSTANTS[0] \
                 * np.exp(-ACTIVATION_ENERGIES[0] / (GAS_CONSTANT * T)) \
                 * x[0, l] \
                 * x[2, l]  # [mol/L/min]
            R2 = RATE_CONSTANTS[1] \
                 * np.exp(-ACTIVATION_ENERGIES[1] / (GAS_CONSTANT * T)) \
                 * x[3, l] \
                 * x[6, l]  # [mol/L/min]
            R3 = RATE_CONSTANTS[2] \
                 * np.exp(-ACTIVATION_ENERGIES[2] / (GAS_CONSTANT * T)) \
                 * x[4, l] \
                 * x[10, l]  # [mol/L/min]
            R4 = RATE_CONSTANTS[3] \
                 * np.exp(-ACTIVATION_ENERGIES[3] / (GAS_CONSTANT * T)) \
                 * x[4, l] \
                 * x[10, l]  # [mol/L/min]

            for i in range(NUM_COMPONENTS):
                # Eq. (4) and the reaction rate matrix r = S*R
                xdot[i, l] = -(Qtot / self.volume_per_section) \
                             * (x[i, l] - x[i, l - 1]) \
                             + (REACTION_STOICHIOMETRY[i][0] * R1) \
                             + (REACTION_STOICHIOMETRY[i][1] * R2) \
                             + (REACTION_STOICHIOMETRY[i][2] * R3) \
                             + (REACTION_STOICHIOMETRY[i][3] * R4)  # [mol/L/min]

        # reshape arrays to a vector
        xdot = reshape(xdot, (self.num_discretization_points * NUM_COMPONENTS, 1))

        return xdot


# liquid-liquid separator model
class LLSeparator:

    def __init__(
            self,
            ID: int,
            volume: float
    ):
        self.ID = ID  # ID of unit. currently used
        self.volume = volume  # volume of L-L-separator [L]
        self.input = Stream()  # one inlet
        self.output = [Stream() for _ in range(2)]  # two outlets

    # calculate the derivatives
    def get_derivative(self, x, z, u, d):
        # x = states
        # u = internal inputs
        # d = disturbances [currently unused]

        Qtot = u[0]  # total volumetric flowrate
        c_in = u[1:15]  # inlet concentration of the components from reactor 3

        # Eq. (10)
        xdot = (Qtot / self.volume) * (c_in - x)  # [mol/L]
        return xdot

    @staticmethod
    def get_algebraic(x, z, u, d):
        # x = states
        # u = internal inputs
        # d = disturbances [currently unused]

        Qtot = u[0]
        x_m = u[15:]  # mass fraction of components in inlet stream
        ci_avg = x

        F_OR = z[0: NUM_COMPONENTS]  # molar flow rate of organic phase. i.e. waste stream   [mol/min]
        F_AQ = z[NUM_COMPONENTS: NUM_COMPONENTS * 2]  # molar flow rate of acqueous phase i.e. product stream [mol/min]
        Q_OR = z[NUM_COMPONENTS * 2]  # volumetric flow rate of organic phase [L/min]
        Q_AQ = z[NUM_COMPONENTS * 2 + 1]  # volumetric flow rate of acqueous phase [L/min]

        # calculate algebraic equation residuals

        # Eq. (9)
        res1 = F_OR + F_AQ - Qtot * ci_avg
        # Eq. (11)
        res2 = (x_m[13] / (x_m[13] + x_m[7] + EPS)) * Qtot - Q_OR
        # Eq. (13)
        res3 = Q_OR * SEPARATION_PARTITION_COEFFICIENTS * F_AQ - Q_AQ * F_OR
        res4 = Q_OR + Q_AQ - Qtot

        return vertcat(res1, res2, res3, res4)  # merge the residuals. use np.concatenate when not using casadi


# overall plant model
class Plant:

    def __init__(self, ND1, ND2, ND3, V1, V2, V3, V4, dt):
        # ND1 = number of spatial discretization points for reactor 1
        # ND2 = number of spatial discretization points for reactor 2
        # ND3 = number of spatial discretization points for reactor 3
        # V1 = volume of reactor 1
        # V2 = volume of reactor 2
        # V3 = volume of reactor 3
        # V4 = volume of liquid-liquid separator
        # dt = sampling time

        # six external input streams and 2 external output streams
        self.input = [Stream() for _ in range(6)]
        self.output = [Stream() for _ in range(2)]

        # ::: ::: Mixer: 1. has two inlets
        self.mixer1 = Mixer(1, 2)
        # connect S1 & S2 to Mixer 1
        self.mixer1.input[:] = self.input[0:2]

        # ::: ::: Tubular reactor: 1
        self.reactor1 = TubularReactor(2, V1, ND1)
        # connect outlet of mixer 1 to inlet of reactor 1
        self.reactor1.input = self.mixer1.output

        # ::: ::: Mixer: 2. has 3 inlets
        self.mixer2 = Mixer(3, 3)
        # connect S3 & S4 to mixer inlets
        self.mixer2.input[0:2] = self.input[2:4]
        # connect outlet from reactor 1 to the 3rd inlet of mixer 2
        self.mixer2.input[2] = self.reactor1.output

        # ::: ::: Tubular reactor: 2
        self.reactor2 = TubularReactor(4, V2, ND2)
        # connect outlet of mixer 2 to inlet of reactor 2
        self.reactor2.input = self.mixer2.output

        # ::: ::: Mixer: 3. has 3 inlets
        self.mixer3 = Mixer(5, 3)
        # connect S5 & S6 to the inlets of mixer 3
        self.mixer3.input[0:2] = self.input[4:6]
        # connect the outlet of reactor 2 to the inlet of mixer 3
        self.mixer3.input[2] = self.reactor2.output

        # ::: ::: Tubular reactor: 3
        self.reactor3 = TubularReactor(6, V3, ND3)
        # connect the outlet of mixer 3 to the inlet of reactor 3
        self.reactor3.input = self.mixer3.output

        # ::: ::: Liquid-Liquid separator: 1
        self.llseparator = LLSeparator(7, V4)
        # connect outlet of reactor 3 to inlet of llseparator
        self.llseparator.input = self.reactor3.output
        # connect the outlets of the llseparator to the two outlets of the overall process
        self.llseparator.output[0:2] = self.output[0:2]

        # define sizes of variables
        self.Nx = NUM_COMPONENTS * (ND1 + ND2 + ND3 + 1)  # state variables
        self.Nz = 30  # algebraic variables
        self.Nu = 32  # internal inputs

        # create casadi symbolic variables
        x = SX.sym('x', self.Nx)
        u = SX.sym('u', self.Nu)
        z = SX.sym('z', self.Nz)

        # create integrator
        opts = {"tf": dt, "abstol": 1E-10}  # interval length
        # dictionary for the integrator
        dae = {'x': x, 'z': z, 'p': u,
               'ode': self.get_derivative(x, z, u),
               'alg': self.get_algebraic(x, z, u)}
        # create the dae integrator
        self.F = integrator('F', 'idas', dae, opts)

    # this calculates the environmental factor. all components are considered harmless except
    # water and atropine in product stream
    @staticmethod
    def calculate_Efactor(z):
        # z = algebraic states

        # component mass flow rate of organic stream
        F_mass_OR = z[0: NUM_COMPONENTS] * MOLECULAR_WEIGHT
        # component mass flow rate of aqueous stream
        F_mass_AQ = z[NUM_COMPONENTS: NUM_COMPONENTS * 2] * MOLECULAR_WEIGHT

        # calculate E-factor
        Efactor = (np.sum(np.delete(F_mass_OR, [7]))
                   + np.sum(np.delete(F_mass_AQ, [7, 8]))) \
                  / F_mass_AQ[8]

        return Efactor.full().ravel().item()

    # this function simulates one time step.
    def simulate(self, x, z, u):
        # x = the state of the plant
        # z = an initial guess of the algebraic state for the algebraic equation solver.
        # a good guess results in faster computation
        # u = the external inputs i.e. volumetric flow rates for streams 1--4

        # update the inlet concentrations of the reactors with the outlets from the mixers
        # and return the updated states as well as all the internal inputs
        x_system, u_system = self.mix_and_get_initial_condition(x, u)
        # compute the states at the next time step
        x_next = self.F(x0=x_system, z0=z, p=u_system)
        # return the updated initial system state, next system state and the algebraic states
        return x_system, x_next["xf"], x_next["zf"]

    # compute the mixer equations and update the initial state concentration of the reactors
    # with the outlets of the mixers

    def mix_and_get_initial_condition(self, x, u):
        ND1 = self.reactor1.num_discretization_points
        ND2 = self.reactor2.num_discretization_points
        ND3 = self.reactor3.num_discretization_points

        # mixer 1
        # stream 1: 2M TROPINE in DMF
        self.mixer1.input[0].component_mass_fraction[0] = 0.293  # approximate mass fraction of 2M Tropine in DMF
        self.mixer1.input[0].component_mass_fraction[1] = 0.707  # approximate mass fraction of 2M Tropine in DMF
        self.mixer1.input[0].evaluate_average_mass_density()  # approximate average mass density of 2M Tropine in DMF
        self.mixer1.input[0].total_volumetric_flowrate = u[0] / 1000.0  # set total volumetric flowrate of tropine
        self.mixer1.input[0].total_volumetric_flowrate_to_total_mass_flowrate()
        self.mixer1.input[0].total_mass_flowrate_to_component_mass_flowrate()
        # stream 2: PHENYLACETYLCHLORIDE (pure or neat)
        self.mixer1.input[1].component_volumetric_flowrate[2] = u[1] / 1000.0
        self.mixer1.input[1].component_volumetric_flowrate_to_component_mass_flowrate()
        # mix streams
        self.mixer1.mix_streams_mass()
        # auxilliary calculations
        self.mixer1.output.evaluate_component_mass_fraction()
        self.mixer1.output.evaluate_average_mass_density()
        self.mixer1.output.evaluate_total_volumetric_flowrate()
        self.mixer1.output.evaluate_component_molar_concentration()

        # select the states for reactor 1
        x_reactor1 = x[0: NUM_COMPONENTS * ND1]
        # reshape the states into a 2D array. casadi reshape function is used for consistency
        x_reactor1 = reshape(x_reactor1, (NUM_COMPONENTS, ND1))
        # inlet dirichlet boundary condition
        x_reactor1[:, 0] = self.reactor1.input.component_molar_concentration[:]
        # set the outlet concentration of reactor 1 to the last state concentrations
        self.reactor1.output.component_molar_concentration = x_reactor1[:, -1]
        # set reactor1 input (internal) to the total volumetric flow rate of Mixer 1
        u_reactor1 = self.mixer1.output.total_volumetric_flowrate
        # reshape back to vector
        x_reactor1 = reshape(x_reactor1, (ND1 * NUM_COMPONENTS, 1))

        # assumption of constant total volumetric flowrate in the axial direction
        # => total volumetric flow rate does not change inside the reactor
        self.reactor1.output.total_volumetric_flowrate = self.reactor1.input.total_volumetric_flowrate

        # mixer 2
        # from reactor 1
        self.mixer2.input[2].component_molar_concentration_to_component_mass_concentration()
        self.mixer2.input[2].component_mass_concentration_to_component_mass_flowrate()
        # stream 3:37 wt% FOMALDEHYDE
        self.mixer2.input[0].component_mass_fraction[4] = 0.37
        self.mixer2.input[0].component_mass_fraction[7] = 0.63
        self.mixer2.input[0].evaluate_average_mass_density()
        self.mixer2.input[0].total_volumetric_flowrate = u[2] / 1000.0
        self.mixer2.input[0].total_volumetric_flowrate_to_total_mass_flowrate()
        self.mixer2.input[0].total_mass_flowrate_to_component_mass_flowrate()
        # stream 4: 4M NaOH solution
        self.mixer2.input[1].component_mass_fraction[6] = 0.138
        self.mixer2.input[1].component_mass_fraction[7] = 0.862
        self.mixer2.input[1].evaluate_average_mass_density()
        self.mixer2.input[1].total_volumetric_flowrate = u[3] / 1000.0
        self.mixer2.input[1].total_volumetric_flowrate_to_total_mass_flowrate()
        self.mixer2.input[1].total_mass_flowrate_to_component_mass_flowrate()
        # mix the three streams
        self.mixer2.mix_streams_mass()
        # auxilliary calculations
        self.mixer2.output.evaluate_component_mass_fraction()
        self.mixer2.output.evaluate_average_mass_density()
        self.mixer2.output.evaluate_total_volumetric_flowrate()
        self.mixer2.output.evaluate_component_molar_concentration()

        # reactor 2
        x_reactor2 = x[NUM_COMPONENTS * ND1: NUM_COMPONENTS * (ND1 + ND2)]
        x_reactor2 = reshape(x_reactor2, (NUM_COMPONENTS, ND2))  # reshape
        # inlet dirichlet boundary condition
        x_reactor2[:, 0] = self.reactor2.input.component_molar_concentration[:]
        self.reactor2.output.component_molar_concentration = x_reactor2[:, -1]
        x_reactor2 = reshape(x_reactor2, (ND2 * NUM_COMPONENTS, 1))
        u_reactor2 = self.mixer2.output.total_volumetric_flowrate

        # constant total volumetric flowrate in the axial direction
        self.reactor2.output.total_volumetric_flowrate = self.reactor2.input.total_volumetric_flowrate

        # mixer 3
        # from reactor 2
        self.mixer3.input[2].component_molar_concentration_to_component_mass_concentration()
        self.mixer3.input[2].component_mass_concentration_to_component_mass_flowrate()
        # stream 5
        self.mixer3.input[0].component_mass_fraction[12] = 0.022
        self.mixer3.input[0].component_mass_fraction[7] = 0.978
        self.mixer3.input[0].evaluate_average_mass_density()
        self.mixer3.input[0].total_volumetric_flowrate = 0.2 / 1000.0  # 0.3 # 0.2 # [mL/min] Fixed
        self.mixer3.input[0].total_volumetric_flowrate_to_total_mass_flowrate()
        self.mixer3.input[0].total_mass_flowrate_to_component_mass_flowrate()
        # stream 6
        self.mixer3.input[1].component_volumetric_flowrate[13] = 0.5 / 1000.0  # 0.5 # [mL/min] Fixed
        self.mixer3.input[1].component_volumetric_flowrate_to_component_mass_flowrate()
        # mix the 3 streams
        self.mixer3.mix_streams_mass()
        # auxilliary calculations
        self.mixer3.output.evaluate_component_mass_fraction()
        self.mixer3.output.evaluate_average_mass_density()
        self.mixer3.output.evaluate_total_volumetric_flowrate()
        self.mixer3.output.evaluate_component_molar_concentration()

        # reactor 3
        x_reactor3 = x[NUM_COMPONENTS * (ND1 + ND2):NUM_COMPONENTS * (ND1 + ND2 + ND3)]
        x_reactor3 = reshape(x_reactor3, (NUM_COMPONENTS, ND3))  # reshape
        # inlet dirichlet boundary condition
        x_reactor3[:, 0] = self.reactor3.input.component_molar_concentration[:]
        self.reactor3.output.component_molar_concentration = x_reactor3[:, -1]
        x_reactor3 = reshape(x_reactor3, (ND3 * NUM_COMPONENTS, 1))
        u_reactor3 = self.mixer3.output.total_volumetric_flowrate

        # constant total volumetric flowrate in the axial direction
        self.reactor3.output.total_volumetric_flowrate = self.reactor3.input.total_volumetric_flowrate
        self.reactor3.output.component_molar_concentration_to_component_mass_concentration()
        self.reactor3.output.component_mass_concentration_to_component_mass_flowrate()
        self.reactor3.output.evaluate_component_mass_fraction()

        # separator 1
        x_llseparator = x[NUM_COMPONENTS * (ND1 + ND2 + ND3):]
        q_llseparator = self.reactor3.output.total_volumetric_flowrate
        component_molar_concentration_llseparator = self.reactor3.output.component_molar_concentration[:]
        component_mass_fraction_llseparator = self.reactor3.output.component_mass_fraction[:]
        u_llseparator = vertcat(q_llseparator, component_molar_concentration_llseparator,
                                component_mass_fraction_llseparator)

        return vertcat(x_reactor1, x_reactor2, x_reactor3, x_llseparator), \
               vertcat(u_reactor1, u_reactor2, u_reactor3, u_llseparator)

    def get_derivative(self, x, z, u):
        # some variables
        ND1 = self.reactor1.num_discretization_points
        ND2 = self.reactor2.num_discretization_points
        ND3 = self.reactor3.num_discretization_points

        # Reactor 1
        x_reactor1 = x[0: NUM_COMPONENTS * ND1]
        T_reactor1 = 373.15  # [K] Fixed reactor temperature
        q_reactor1 = u[0]
        u_reactor1 = [q_reactor1, T_reactor1]
        # reactor 1 state velocities
        xdot_reactor1 = self.reactor1.get_derivative(x_reactor1, u_reactor1, 0)

        # Reactor 2
        x_reactor2 = x[NUM_COMPONENTS * ND1: NUM_COMPONENTS * (ND1 + ND2)]
        T_reactor2 = 373.15  # [K] Fixed reactor temperature
        q_reactor2 = u[1]
        u_reactor2 = [q_reactor2, T_reactor2]
        # reactor 2 state velocities
        xdot_reactor2 = self.reactor2.get_derivative(x_reactor2, u_reactor2, 0)

        # Reactor 3
        x_reactor3 = x[NUM_COMPONENTS * (ND1 + ND2): NUM_COMPONENTS * (ND1 + ND2 + ND3)]
        T_reactor3 = 323.15  # [K] Fixed reactor temperature
        q_reactor3 = u[2]
        u_reactor3 = [q_reactor3, T_reactor3]
        # reactor 3 state velocities
        xdot_reactor3 = self.reactor3.get_derivative(x_reactor3, u_reactor3, 0)

        # Separator 1
        x_llseparator = x[NUM_COMPONENTS * (ND1 + ND2 + ND3):]
        z_llseparator = z[:]
        u_llseparator = u[3:]
        xdot_llseparator = self.llseparator.get_derivative(x_llseparator, z_llseparator, u_llseparator, 0)

        xdot = vertcat(xdot_reactor1, xdot_reactor2, xdot_reactor3, xdot_llseparator)

        return xdot

    def get_algebraic(self, x, z, u):
        ND1 = self.reactor1.num_discretization_points
        ND2 = self.reactor2.num_discretization_points
        ND3 = self.reactor3.num_discretization_points

        x_llseparator = x[NUM_COMPONENTS * (ND1 + ND2 + ND3):]
        z_llseparator = z[:]
        u_llseparator = u[3:]
        res_llseparator = self.llseparator.get_algebraic(x_llseparator, z_llseparator, u_llseparator, 0)
        return res_llseparator


# simulate model for one time step and get next states and output
def atropine_sim_model(x, u, A, B, C, D):
    # x = state
    # u =  input.
    u = u / 1000  # this has been scaled in the optimization problem. hence the division by 1000 to unscale it
    xplus = np.matmul(A, x) + np.matmul(B, u)  #
    y = np.matmul(C, x) + np.matmul(D, u)
    return xplus, y


# MPC/EMPC controller
def atropine_mpc_controller(x0, N, Nx, Nu, uss, ur, yr, A, B, C, D):
    # controller weights. different values may lead to different controller performance or cause instability
    R = np.identity(4) * 0.1  # weight on inputs
    Q = 10  # weight on output

    # start with empty NLP
    w = []  # decision variables
    w0 = []  # decision variables initial guess
    lbw = []  # lower bound on decision variables
    ubw = []  # upper bound on decision variables
    J = 0  # initial cost
    g = []  # state constraint
    lbg = []  # lower bound on state constraint
    ubg = []  # upper bound on state constraint

    # formulate NLP
    xk = x0  # fix initial state from state estimator

    for k in range(N):
        Uk = SX.sym('U_' + str(k), Nu)  # create input variable at time step k
        w.append(Uk)  # store in decision variable list
        lbw.append(- uss * 1000)  # lower bound on inputs at time k
        ubw.append(uss * 1000)  # upper bound on inputs at time k
        w0.append([0] * Nu)  # initial input guess

        # simulate model to get x(k+1) and y(k)
        xk, yk = atropine_sim_model(xk, Uk, A, B, C, D)
        # calculate the cost
        J += Q * (yk - yr) ** 2 \
             + mtimes((Uk - ur).T, mtimes(R, (Uk - ur)))
        # add state constraints. currently the states are unconstrained since the
        # identified model states have no physical meaning
        g.append(xk)
        lbg.append([-inf] * Nx)
        ubg.append([inf] * Nx)

    # Create an NLP solver
    opts = {}
    opts["verbose"] = False
    opts["ipopt.print_level"] = 0
    opts["print_time"] = 0
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob, opts)

    # Solve the NLP
    sol = solver(
        x0=vertcat(*w0),
        lbx=vertcat(*lbw),
        ubx=vertcat(*ubw),
        lbg=vertcat(*lbg),
        ubg=vertcat(*ubg)
    )
    # get solution
    w_opt = sol['x']
    u_opt = w_opt[0:Nu]
    return u_opt.full().ravel()
