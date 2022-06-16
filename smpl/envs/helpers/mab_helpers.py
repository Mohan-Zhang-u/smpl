import numpy as np
import mpctools as mpc
import casadi as cs
import copy
from tqdm import tqdm
from scipy import integrate
import matplotlib.pyplot as plt
import os


class ControllerHelper:
    def __init__(self, nx_list,
                 nu_list,
                 num_sim,
                 dt_itgl,
                 dt_spl,
                 xscale,
                 uscale,
                 xss,
                 uss,
                 solver_opts = None,
                 casadi_opts = None):
        self.nx_list = nx_list
        self.nu_list = nu_list
        self.Nx_up = nx_list[0]; self.Nx_buffer = nx_list[1]; self.Nx_down = nx_list[2]
        self.Nx = sum(nx_list)
        self.Nu_up = nu_list[0]; self.Nu_buffer = nu_list[1]; self.Nu_down = nu_list[2]
        self.Nu = sum(nu_list)
        self.num_sim = num_sim
        self.dt_itgl = dt_itgl
        self.dt_spl = dt_spl
        self.dt_ratio = int(self.dt_spl/self.dt_itgl)
        self.xscale = xscale
        self.uscale = uscale
        self.xss = xss
        self.uss = uss
        self.solver_opts = solver_opts
        self.casadi_opts = casadi_opts
        self.init_model_helper()

        self.sat = 20000000

    def init_model_helper(self):
        self.model_helper = ModelHelper(nx_list=self.nx_list,
                                           nu_list=self.nu_list,
                                           xscale=self.xscale,
                                           uscale=self.uscale,
                                           xss=self.xss,
                                           uss=self.uss)
        self.up_model_helper = UpModelHelper(Nx=self.Nx_up,
                                             Nu=self.Nu_up,
                                             xss=self.xss[:self.Nx_up],
                                             uss=self.uss[:self.Nu_up],
                                             xscale=self.xscale[:self.Nx_up],
                                             uscale=self.uscale[:self.Nu_up])

    # ------------------------------------------------------------------------------------------------------------------
    # Function and auxiliary functions for constructing simulators
    # ------------------------------------------------------------------------------------------------------------------
    def prepare_simulators(self, Q=None, R=None, N=None, solver_opts=None, casadi_opts=None):
        """
        Creates simulators based on the demand of the study.
        For examples, if we want to compare empc and mpc, we can create each of them.
        If we want to study the effects of N on mpc, we can create several mpcs with different values of N.
        If we only want to study the model itself, we just ignore the controllers.
        """
        # Plant. Consider this as the plant in reality. It is not used in controller
        self.plant = self._build_plant()
        # MPC
        self.mpc_cont = self._build_mpc_up(Q, R, N, self.dt_spl, self.xss[:self.Nx_up], self.uss[:self.Nu_up], self.xscale[:self.Nx_up], self.uscale[:self.Nu_up])
        self.mpc_cont.initialize(casadioptions=casadi_opts, solveroptions=solver_opts)
        # EMPC
        self.empc_cont = self._build_empc_up(N, self.dt_spl, self.xss[:self.Nx_up], self.uss[:self.Nu_up], self.xscale[:self.Nx_up], self.uscale[:self.Nu_up])
        self.empc_cont.initialize(casadioptions=casadi_opts, solveroptions=solver_opts)
        return self.plant, self.mpc_cont, self.empc_cont

    def _build_plant(self):
        xdot_scale = self.model_helper.xdot_scale  # Excluding the time information
        self.plant = mpc.DiscreteSimulator(xdot_scale, self.dt_itgl, [self.Nx, self.Nu], ['x', 'u'])
        return self.plant

    def _build_mpc_up(self, Q, R, N, dt, xss, uss, xscale, uscale):
        Nx = xss.size
        Nu = uss.size
        xdot_scale = self.up_model_helper.xdot_scale  # Upstream model only

        def stage_cost(x, u):
            xd = x - xss / xscale
            ud = u - uss / uscale
            return mpc.mtimes(xd.T, Q, xd) + mpc.mtimes(ud.T, R, ud)

        lfunc = mpc.getCasadiFunc(stage_cost, [Nx, Nu], ['x', 'u'], 'lfunc')

        xlb = xss * 0.8 / xscale  # np.ones(self.Nx)*1e-10
        xlb[16] = 33.0 / 37.0

        xub = 1.2 * xss / xscale
        xub[7] = 1
        xub[15] = 1
        xub[16] = 1

        ulb = 0.8 * uss / uscale
        uub = uscale / uscale

        contargs = dict(
            N={"t": N, "x": Nx, "u": Nu, "c": 3},  # , "e":Ns, "s":Ns},
            verbosity=0,
            l=lfunc,
            # e=zfunc,
            x0=xss / xss,
            ub={"u": uub, "x": xub},  # Change upper bounds
            lb={"u": ulb, "x": xlb},  # Change lower bounds
            guess={
                "x": xss / xscale,
                "u": uss / uscale  # [:Nu]
            }
        )

        ctrl = mpc.nmpc(f=xdot_scale,
                        Delta=dt, timelimit=120,
                        discretel=False,
                        **contargs,
                        )

        return ctrl

    def _build_empc_up(self, N, dt, xss, uss, xscale, uscale):
        Nx = xss.size
        Nu = uss.size
        xdot_scale = self.up_model_helper.xdot_scale  # Upstream model only

        def stage_cost(x, u):
            xx = x * cs.DM(xscale)
            uu = u * cs.DM(uscale)
            return - xx[6] * uu[1] - x[14] * uu[3]

        lfunc = mpc.getCasadiFunc(stage_cost, [Nx, Nu], ['x', 'u'], 'lfunc')

        xlb = xss * 0.8 / xscale  # np.ones(self.Nx)*1e-10
        xlb[16] = 33.0 / 37.0

        xub = 1.2 * xss / xscale
        xub[7] = 1
        xub[15] = 1
        xub[16] = 1

        ulb = 0.8 * uss / uscale
        uub = uscale / uscale

        contargs = dict(
            N={"t": N, "x": Nx, "u": Nu, "c": 3},  # , "e":Ns, "s":Ns},
            verbosity=0,
            l=lfunc,
            # e=zfunc,
            x0=xss / xscale,
            ub={"u": uub, "x": xub},  # Change upper bounds
            lb={"u": ulb, "x": xlb},  # Change lower bounds
            guess={
                "x": xss / xscale,
                "u": uss / uscale  # [:Nu]
            }
        )

        ctrl = mpc.nmpc(f=xdot_scale,
                        Delta=dt, timelimit=120,
                        discretel=False,
                        **contargs,
                        )

        return ctrl

    # ------------------------------------------------------------------------------------------------------------------
    # Function and auxiliary functions for run simulations
    # ------------------------------------------------------------------------------------------------------------------
    def run(self, Xm, Um, Xe, Ue, Xs, Us, Xi, Xie, t, u0, cl):
        uk = copy.deepcopy(u0)
        print("part 1")
        for k in tqdm(range(0, self.num_sim)):
            # Time
            t += [k * self.dt_spl]

            if cl:
                # Control
                uk = self._control(Xm[k], uk, 'mpc', k)

            # Advance one sampling time step.
            # Consider this as we obtain the measurements/observations from the plant after one sampling time
            for i in range(0, self.dt_ratio):
                xk = self._simulation(Xi[k * self.dt_ratio + i], uk)
                Xi += [copy.deepcopy(xk)]

            # Store solution
            Um += [copy.deepcopy(uk)]
            Xm += [copy.deepcopy(xk)]

            Xs += [copy.deepcopy(self.xss)]
            Us += [copy.deepcopy(self.uss)]

        # EMPC
        print("part 2 with empc")
        uk = copy.deepcopy(u0)  # Reinitialize uk to uss
        for k in tqdm(range(0, self.num_sim)):
            # Time
            # t += [k * dt_spl]

            if cl:
                # Control
                uk = self._control(Xe[k], uk, 'empc', k)

            # Advance one sampling time step.
            # Consider this as we obtain the measurements/observations from the plant after one sampling time
            for i in range(0, self.dt_ratio):
                xk = self._simulation(Xie[k * self.dt_ratio + i], uk)
                Xie += [copy.deepcopy(xk)]

            # Store solution
            Ue += [copy.deepcopy(uk)]
            Xe += [copy.deepcopy(xk)]
        return Xm, Um, Xe, Ue, Xs, Us, Xi, Xie, t

    def _simulation(self, xk, uk):
        # Switch configuration
        if uk[-1] == 1:  # 1 means switch configuration #TODO: change to switch configuration. relax requirements! when switch, t=0 is both full and empty (A is full B is empty. A and B are both valid data points at t=0)
            print(xk[-1] * uk[7], 'mg of mAb is captured. Switching the column')
            xk[19:-1] = 0  # (1950). Set state vector of downstream capture column to 0
            xk[-1] = 0  # In new column, accumulated mab is 0
            uk[-1] = 0  # Reset
        # Integrator
        xkp1 = self.plant.sim(xk, uk)
        # update accumulated mAb
        xkp1[-1] += self.dt_itgl * (
                    xkp1[19] - xkp1[-14])  # difference between inlet concentration and outlet concentration
        return xkp1

    def _control(self, x, u, control, k):
        # Separate x for each controller
        x_up = x[0:17]  # 17
        x_buffer = x[17:19]  # 2
        x_down = x[19:]  # 1951
        # Upstream controller
        if control == 'mpc':
            ###### mpc
            # compute control action: mpc
            self.mpc_cont.fixvar("x", 0, x_up)
            self.mpc_cont.solve()

            # Print status and make sure solver didn't fail.
            print("%5s %d: %s" % ("MPC", k, self.mpc_cont.stats["status"]))
            if self.mpc_cont.stats["status"] != "Solve_Succeeded":
                0  # break
            else:
                self.mpc_cont.saveguess()

            u[:7] = np.squeeze(self.mpc_cont.var["u", 0])
        if control == 'empc':
            #### empc
            # compute control action: empc
            self.empc_cont.fixvar("x", 0, x_up)
            self.empc_cont.solve()

            # Print status and make sure solver didn't fail.
            print("%5s %d: %s" % ("EMPC", k, self.empc_cont.stats["status"]))
            if self.empc_cont.stats["status"] != "Solve_Succeeded":
                0  # break
            else:
                self.empc_cont.saveguess()

            u[:7] = np.squeeze(self.empc_cont.var["u", 0])

        # Level P controller
        u[7] = self._pcontroller(x_buffer[0])
        # Downstream controller
        u[8] = self._switcher(x_down)
        return u

    def _pcontroller(self, y):
        # Parameters
        Kc = -1 / 60  #
        p_ss = self.uss[7] / self.uscale[7]  # Steady state of outlet flow rate
        y_ss = self.xss[17] / self.xscale[17]  # Steady state of liquid level

        # Implement P controller
        p = p_ss + Kc * (y_ss - y)
        return p

    def _switcher(self, x_down): # change when Cout has the danger concentration.
        if x_down[-14] >= self.xss[18] * 0.01:  # 1% of breakthrough curve
            switch = 1  # Column is saturated, now switch column
        else:
            switch = 0
        return switch

    def _switcher2(self, x_down): # change when full.
        if x_down[-1] >= self.sat:
            switch = 1  # Column is saturated, now switch column
        else:
            switch = 0
        return switch


# model helper
class ModelHelper:
    def __init__(self, nx_list, nu_list, xscale, uscale, xss, uss):
        self.Nx_up = nx_list[0]; self.Nx_buffer = nx_list[1]; self.Nx_down = nx_list[2]
        self.Nx = sum(nx_list)
        self.Nu_up = nu_list[0]; self.Nu_buffer = nu_list[1]; self.Nu_down = nu_list[2]
        self.Nu = sum(nu_list)
        self.xscale = xscale
        self.uscale = uscale
        self.xss = xss
        self.uss = uss

        self.sat = 200000000

        # # Capture column loading process
        num_z_cap_load = 150  # Number of discretization nodes in capture column
        num_r = 10  # Number of discretization nodes in radius of beads
        self.num_z_cap_load = num_z_cap_load  # discretization points in Z direction
        self.num_r = num_r  # discretization points in r direction
        # Define state number
        num_c_cap_load = self.num_z_cap_load  # Number of c states
        num_cp_cap_load = self.num_z_cap_load * self.num_r  # Number of cp states
        num_q1_cap_load = self.num_z_cap_load  # Number of q1 states
        num_q2_cap_load = self.num_z_cap_load  # Number of q2 states
        self.num_x_cap_load = num_c_cap_load + num_cp_cap_load + num_q1_cap_load + num_q2_cap_load  # Total number of states
        self.vol_cap = 2*50000/1000  # L
        self.len_cap = 5*4/10  # dm

        # Defined at the outside of __init__
        self.plant = None
        self.upmodel_helper = None
        self.empc_cont = None
        self.mpc_cont = None

    def xdot(self, x, u):
        """
        :param self:
        :param x:
        :param u:
        :return:
        """
        """Parameters"""
        # Upstream model
        K_damm = 1.76  # ammonia constant for cell death (mM)
        K_dgln = 9.6e-03 / 60  # constant for glutamine degradation (min^(-1))
        K_glc = 0.75  # Monod constant for glucose (mM)
        K_gln = 0.075  # Monod constant for glutamine (mM)
        KI_amm = 28.48  # Monod constant for ammonia (mM)
        KI_lac = 171.76  # Monod constant for lactate (mM)
        m_glc = 4.9e-14 / 60  # maintenance coefficients of glucose (mmol/cell/min)
        n = 2  # constant represents the increase of specific death as ammonia concentration increases (-)
        Y_ammgln = 0.45  # yield of ammonia from glutamine (mmol/mmol)
        Y_lacglc = 2.0  # yield of lactate from glucose (mmol/mmol)
        Y_Xglc = 2.6e8  # yield of cells on glucose (cell/mmol)
        Y_Xgln = 8e8  # yield of cells on glutamine (cell/mmol)
        alpha1 = 3.4e-13 / 60  # constants of glutamine maintenance coefficient (mM L/cell/min)
        alpha2 = 4  # constants of glutamine maintenance coefficient (mM)
        mu_max = (0.0016 * x[16] - 0.0308) * 5 / 60  # 0.058 * 5  # maximum specific growth rate (min^(-1))
        mu_dmax = 0.06/60  # (-0.0045 * x[16] + 0.1682) / 60  # 0.06    # maximum specific death rate (min^(-1))
        m_mabx = 6.59e-10 / 60  # constant production of mAb by viable cells (mg/Cell/min)

        # Add new parameters -- Ben
        Delta_H = 5e5  # 4.4e4# 1e-8
        rho = 1560.0  # density of mixture is assumed to be density of glucose [g/L]
        cp_up = 1.244  # specific heat capacity of mixture is assumed to be that of glucose [J/g/dC]
        U = 4e2
        T_in = 37.0

        # Buffer tank
        D_bf = 5  # Diameter of the tank (dm)
        L_bf = D_bf*5  # Diameter of the tank (dm)
        Ac_bf = np.pi * (D_bf / 2) ** 2  # Cross-sectional area (dm^2)
        V_bf = Ac_bf * L_bf  # Volume of the tank (L)

        # Capture column
        qmax1 = 36.45 * 1000  # column capacity (mg/L)
        k1 = 0.704 / 1000  # kinetic constant (L/(mg min))
        qmax2 = 77.85 * 1000  # column capacity (mg/L)
        k2 = 0.021 / 1000  # kinetic constant (L/(mg min))
        K = 15.3 / 1000  # Langmuir equilibrium constant (L/mg)
        Deff = 7.6e-5 / 100  # effective pore diffusivity (dm2/min)

        rp = 4.25e-3 / 10  # particle radius (dm)
        L = self.len_cap  # * column length (dm)
        V = self.vol_cap  # * column volume (L)
        column_r = np.sqrt(V / (np.pi * L))  # column radius (dm)
        v = u[7] / (np.pi * column_r ** 2)  # * velocity (dm/min). Calculated using inlet flow rate/harvest flow rate
        # ** v：interstitial velocity (*double check how to calculate it*)

        Dax = 0.55 * v / 10  # axial dispersion coefficient (dm2/min)
        kf = 0.067 * v ** 0.58  # mass transfer coefficient (dm/min)
        epsilon_c = 0.31  # extra-particle column void (-)
        epsilon_p = 0.94  # particle porosity (-)

        dz = L / (self.num_z_cap_load + 1)  # distance delta (dm)
        dr = rp / (self.num_r + 1)  # delta radius (dm)

        """Initialize ode list"""
        dxdt = []

        """Inputs"""
        # Upstream model
        F_in = u[0]  # inlet flow rate (L/min)
        F_1 = u[1]  # outlet flow rate of bioreactor (L/min)
        F_r = u[2]  # u[1]-u[3] # u[2]      # recycle flow rate (L/min)
        F_2 = u[3]  # Outlet flow rate of separator (L/min)
        GLC_in = u[4]  # inlet glucose concentration (mM)
        GLN_in = u[5]  # inlet glutamine concentration (mM)
        Tc = u[6]  # coolant temperature (d C)
        # Buffer tank
        F_out_bf = u[7]  # Outlet flow rate of the buffer tank (L/min)
        F_in_bf = u[3]  # Inlet flow rate of the buffer tank (L/min)
        c_in_bf = x[14]  # Inlet concentration of mAb (mg/L)
        # Capture column
        F = u[7]  # Inlet flow rate/harvest flow rate (L/min)
        cF = x[18]  # mAb concentration in mobile phase/harvest mAb concentration (mg/L)

        """States"""
        # Upstream model
        Xv1 = x[0]  # concentration of viable cells in reactor(cell/L)
        Xt1 = x[1]  # total concentration of cells in reactor(cell/L)
        GLC1 = x[2]  # glucose concentration in reactor(mM)
        GLN1 = x[3]  # glutamine concentration in reactor(mM)
        LAC1 = x[4]  # lactate concentration in reactor(mM)
        AMM1 = x[5]  # ammonia concentration in reactor(mM)
        mAb1 = x[6]  # mAb concentration in reactor(mg/L)
        V1 = x[7]  # volume in reactor (L)
        Xv2 = x[8]  # concentration of viable cells in separator(cell/L)
        Xt2 = x[9]  # total concentration of cells in separator(cell/L)
        GLC2 = x[10]  # glucose concentration in separator(mM)
        GLN2 = x[11]  # glutamine concentration in separator(mM)
        LAC2 = x[12]  # lactate concentration in separator(mM)
        AMM2 = x[13]  # ammonia concentration in separator(mM)
        mAb2 = x[14]  # mAb concentration in separator(mg/L)
        V2 = x[15]  # volume in separator (L)
        # temperature as a state -- Ben
        T = x[16]  # (d C)
        # Buffer tank
        h = x[17]  # Liquid level in the buffer tank (dm)
        c_bf = x[18]  # Concentration of mAb in the buffer tank (mg/L)
        # All states of capture column
        x_column = x[19:-1]
        x_reshape = cs.reshape(x_column, (1 + self.num_r + 1 + 1, self.num_z_cap_load))  # Convert from Nx * 1  to Nz *  ( c + cp + q1 + q2)
        x_reshape = x_reshape.T  # Transpose to get correct shape
        # Capture column
        c = x_reshape[:, 0]  # mAb concentration in the mobile phase (mg/L)
        cp = x_reshape[:, 1:self.num_r + 1]  # mAb concentration inside the particle (mg/L)
        q1 = x_reshape[:, self.num_r + 1:self.num_r + 2]  # adsorbed mAb concentration (mg/L)
        q2 = x_reshape[:, self.num_r + 2:self.num_r + 3]
        # Separator: assumptions: 92% cell recycle rate and 20% mab retention
        Xvr = 0.92 * Xv1 * F_1 / F_r  # concentration of viable cells in recycle stream(cell/L)
        Xtr = 0.92 * Xt1 * F_1 / F_r  # total concentration of cells in recycle stream(cell/L)
        GLCr = 0.2 * GLC1 * F_1 / F_r  # glucose concentration in recycle stream(mM)
        GLNr = 0.2 * GLN1 * F_1 / F_r  # glutamine concentration in recycle stream(mM)
        LACr = 0.2 * LAC1 * F_1 / F_r  # lactate concentration in recycle stream(mM)
        AMMr = 0.2 * AMM1 * F_1 / F_r  # ammonia concentration in recycle stream(mM)
        mAbr = 0.2 * mAb1 * F_1 / F_r  # mAb concentration in recycle stream (mg/L)

        """ODEs"""
        # Upstream model
        f_lim = (GLC1 / (K_glc + GLC1)) * (GLN1 / (K_gln + GLN1))
        f_inh = (KI_lac / (KI_lac + LAC1)) * (KI_amm / (KI_amm + AMM1))
        mu = mu_max * f_lim * f_inh
        mu_d = mu_dmax / (1 + (K_damm / AMM1) ** n)

        dxdt += [mu * Xv1 - mu_d * Xv1 - (F_in / V1) * Xv1 + (F_r / V1) * (Xvr - Xv1)]
        dxdt += [mu * Xv1 + (F_r / V1) * (Xtr - Xt1) - (F_in / V1) * Xt1]

        Q_glc = mu / Y_Xglc + m_glc
        dxdt += [-Q_glc * Xv1 + (F_in / V1) * (GLC_in - GLC1) + (F_r / V1) * (GLCr - GLC1)]

        m_gln = (alpha1 * GLN1) / (alpha2 + GLN1)
        Q_gln = mu / Y_Xgln + m_gln
        dxdt += [-Q_gln * Xv1 - K_dgln * GLN1 + (F_in / V1) * (GLN_in - GLN1) + (F_r / V1) * (GLNr - GLN1)]

        Q_lac = Y_lacglc * Q_glc
        dxdt += [Q_lac * Xv1 - (F_in / V1) * LAC1 + (F_r / V1) * (LACr - LAC1)]

        Q_amm = Y_ammgln * Q_gln
        dxdt += [Q_amm * Xv1 + K_dgln * GLN1 - (F_in / V1) * AMM1 + (F_r / V1) * (AMMr - AMM1)]
        dxdt += [m_mabx * Xv1 - (F_in / V1) * mAb1 + (F_r / V1) * (mAbr - mAb1)]

        dxdt += [F_in + F_r - F_1]

        dxdt += [(F_1 / V2) * (Xv1 - Xv2) - (F_r / V2) * (Xvr - Xv2)]
        dxdt += [(F_1 / V2) * (Xt1 - Xt2) - (F_r / V2) * (Xtr - Xt2)]
        dxdt += [(F_1 / V2) * (GLC1 - GLC2) - (F_r / V2) * (GLCr - GLC2)]
        dxdt += [(F_1 / V2) * (GLN1 - GLN2) - (F_r / V2) * (GLNr - GLN2)]
        dxdt += [(F_1 / V2) * (LAC1 - LAC2) - (F_r / V2) * (LACr - LAC2)]
        dxdt += [(F_1 / V2) * (AMM1 - AMM2) - (F_r / V2) * (AMMr - AMM2)]
        dxdt += [(F_1 / V2) * (mAb1 - mAb2) - (F_r / V2) * (mAbr - mAb2)]
        dxdt += [F_1 - F_2 - F_r]

        # Temperature dynamics -- Ben
        # Average mass of an mAb cell is 150 kDa = 2.4908084e-19 g
        dxdt += [
            (F_in / V1) * (T_in - T) + (Delta_H / (rho * cp_up)) * mu * Xv1 * 2.4908084e-19 + (
                        U / (V1 * rho * cp_up)) * (
                    Tc - T)]

        # Buffer tank:
        dhdt = 1 / Ac_bf * (F_in_bf - F_out_bf)
        dcdt_bf = F_in_bf / (Ac_bf * h) * (c_in_bf - c_bf)
        dxdt += [dhdt]
        dxdt += [dcdt_bf]

        # Capture column
        # mAb concentration in mobile phase (c)
        dc = cs.SX.zeros(self.num_z_cap_load + 1)
        k = np.arange(1, self.num_z_cap_load)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc[k] = (c[k] - c[k - 1]) / dz
        # Top boundary condition: stands for difference between top and point [0]
        dc[0] = v / (epsilon_c * Dax) * (c[0] - cF)  # check c[0] or c[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc[-1] = 0

        # Second order discretization
        dc2 = cs.SX.zeros(self.num_z_cap_load)
        k = np.arange(0, self.num_z_cap_load)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2[k] = (dc[k + 1] - dc[k]) / dz
        # from point 0 to point 49 (total 50 points)
        dc_true = dc[1:self.num_z_cap_load + 1]
        # Cp value at r=rp point
        cp_rp = cp[:, -1]  # *
        # calculate dcdt value
        dcdt = Dax * dc2 - (v / epsilon_c) * dc_true - (1 - epsilon_c) / epsilon_c * 3 / (rp) * kf * (c - cp_rp)

        # mAb concentration inside the particle
        dcp = cs.SX.zeros((self.num_z_cap_load, self.num_r + 1))
        k = np.arange(1, self.num_r)
        # forward difference: @ point 0 = c[1] -c[0],
        # dcp[1] stands for point 0 here.
        dcp[:, k] = (cp[:, k] - cp[:, k - 1]) / dr
        # Top boundary: dcp[0] is difference between r=0 and point 0
        dcp[:, 0] = 0
        # Bottom boundary: dcp[30] stands for the difference between point 29 and r=rp
        dcp[:, -1] = kf / Deff * (c[:] - cp_rp)

        # second order derivation
        dcp2 = cs.SX.zeros((self.num_z_cap_load, self.num_r))
        k = np.arange(0, self.num_r + 1)
        r = cs.SX.zeros((1, self.num_r + 1))
        r[:, k] = (k + 1) * dr  # particle radius
        r_repmat = cs.repmat(r, self.num_z_cap_load, 1)

        k = np.arange(0, self.num_r)
        dcp2[:, k] = (dcp[:, k + 1] * (r_repmat[:, k + 1] ** 2) - dcp[:, k] * (r_repmat[:, k] ** 2)) / dr
        # dcp2[:, k] = (dcp[:, k + 1] - dcp[:, k]) / dr
        cp_rp = cs.reshape(cp_rp, (self.num_z_cap_load, 1))

        # # absorbed mAb concentration (mg/ml)
        dq1dt = k1 * ((qmax1 - q1) * cp_rp - q1 / K)  # q1: quick adsorption rete
        dq2dt = k2 * ((qmax2 - q2) * cp_rp - q2 / K)  # q2: slow adsorption rete

        dcpdt = cs.SX.zeros((self.num_z_cap_load, self.num_r))
        k = np.arange(0, self.num_r)
        dcpdt[:, k] = Deff * (1 / (r_repmat[:, k] ** 2)) * dcp2[:, k] - (1 / epsilon_p) * (dq1dt[:] + dq2dt[:])

        dxdt_column = cs.horzcat(dcdt, dcpdt, dq1dt, dq2dt)
        dxdt_column = dxdt_column.T
        dxdt_column = cs.reshape(dxdt_column, (self.num_x_cap_load, 1))

        dxdt += [dxdt_column]
        dxdt += [0]  # The accumulated mAb is not calculated within integrator, so here we set the ode to 0
        # Add constraints on each state
        return dxdt

    def xdot_scale(self, x, u):
        return cs.vertcat(*self.xdot(x * cs.DM(self.xscale), u * cs.DM(self.uscale))) / cs.DM(self.xscale)

class UpModelHelper:
    def __init__(self, Nx, Nu, xss, uss, xscale, uscale):
        self.Nx = Nx
        self.Nu = Nu
        self.xss = xss
        self.uss = uss
        self.xscale = xscale
        self.uscale = uscale

    def xdot(self, x, u):
        # Parameter values
        K_damm = 1.76  # ammonia constant for cell death (mM)
        K_dgln = 9.6e-03 / 60  # constant for glutamine degradation (min^(-1))
        K_glc = 0.75  # Monod constant for glucose (mM)
        K_gln = 0.075  # Monod constant for glutamine (mM)
        KI_amm = 28.48  # Monod constant for ammonia (mM)
        KI_lac = 171.76  # Monod constant for lactate (mM)
        m_glc = 4.9e-14 / 60  # maintenance coefficients of glucose (mmol/cell/min)
        n = 2  # constant represents the increase of specific death as ammonia concentration increases (-)
        Y_ammgln = 0.45  # yield of ammonia from glutamine (mmol/mmol)
        Y_lacglc = 2.0  # yield of lactate from glucose (mmol/mmol)
        Y_Xglc = 2.6e8  # yield of cells on glucose (cell/mmol)
        Y_Xgln = 8e8  # yield of cells on glutamine (cell/mmol)
        alpha1 = 3.4e-13 / 60  # constants of glutamine maintenance coefficient (mM L/cell/min)
        alpha2 = 4  # constants of glutamine maintenance coefficient (mM)
        mu_max = (0.0016 * x[16] - 0.0308) * 5 / 60  # 0.058 * 5  # maximum specific growth rate (min^(-1))
        mu_dmax = 0.06/60  #(-0.0045 * x[16] + 0.1682) / 60  # 0.06    # maximum specific death rate (min^(-1))
        m_mabx = 6.59e-10 / 60  # constant production of mAb by viable cells (mg/Cell/min)

        # Add new parameters -- Ben
        Delta_H = 5e5  # 4.4e4# 1e-8
        rho = 1560.0  # density of mixture is assumed to be density of glucose [g/L]
        cp = 1.244  # specific heat capacity of mixture is assumed to be that of glucose [J/g/dC]
        U = 4e2
        T_in = 37.0

        # Initialize ode list
        dxdt = []

        # Inputs:
        F_in = u[0]  # inlet flow rate (L/min)
        F_1 = u[1]  # outlet flow rate of bioreactor (L/min)
        F_r = u[2]  # u[1]-u[3] # u[2]      # recycle flow rate (L/min)
        F_2 = u[3]  # Outlet flow rate of separator (L/min)
        GLC_in = u[4]  # inlet glucose concentration (mM)
        GLN_in = u[5]  # inlet glutamine concentration (mM)
        Tc = u[6]  # coolant temperature (d C)

        # States:
        Xv1 = x[0]  # concentration of viable cells in reactor(cell/L)
        Xt1 = x[1]  # total concentration of cells in reactor(cell/L)
        GLC1 = x[2]  # glucose concentration in reactor(mM)
        GLN1 = x[3]  # glutamine concentration in reactor(mM)
        LAC1 = x[4]  # lactate concentration in reactor(mM)
        AMM1 = x[5]  # ammonia concentration in reactor(mM)
        mAb1 = x[6]  # mAb concentration in reactor(mg/L)
        V1 = x[7]  # volume in reactor (L)
        Xv2 = x[8]  # concentration of viable cells in separator(cell/L)
        Xt2 = x[9]  # total concentration of cells in separator(cell/L)
        GLC2 = x[10]  # glucose concentration in separator(mM)
        GLN2 = x[11]  # glutamine concentration in separator(mM)
        LAC2 = x[12]  # lactate concentration in separator(mM)
        AMM2 = x[13]  # ammonia concentration in separator(mM)
        mAb2 = x[14]  # mAb concentration in separator(mg/L)
        V2 = x[15]  # volume in separator (L)
        # temperature as a state -- Ben
        T = x[16]  # (d C)

        # Assumptions: 92% cell recycle rate and 20% mab retention
        Xvr = 0.92 * Xv1 * F_1 / F_r  # concentration of viable cells in recycle stream(cell/L)
        Xtr = 0.92 * Xt1 * F_1 / F_r  # total concentration of cells in recycle stream(cell/L)
        GLCr = 0.2 * GLC1 * F_1 / F_r  # glucose concentration in recycle stream(mM)
        GLNr = 0.2 * GLN1 * F_1 / F_r  # glutamine concentration in recycle stream(mM)
        LACr = 0.2 * LAC1 * F_1 / F_r  # lactate concentration in recycle stream(mM)
        AMMr = 0.2 * AMM1 * F_1 / F_r  # ammonia concentration in recycle stream(mM)
        mAbr = 0.2 * mAb1 * F_1 / F_r  # mAb concentration in recycle stream (mg/L)

        # ODEs
        f_lim = (GLC1 / (K_glc + GLC1)) * (GLN1 / (K_gln + GLN1))
        f_inh = (KI_lac / (KI_lac + LAC1)) * (KI_amm / (KI_amm + AMM1))
        mu = mu_max * f_lim * f_inh
        mu_d = mu_dmax / (1 + (K_damm / AMM1) ** n)

        dxdt += [mu * Xv1 - mu_d * Xv1 - (F_in / V1) * Xv1 + (F_r / V1) * (Xvr - Xv1)]
        dxdt += [mu * Xv1 + (F_r / V1) * (Xtr - Xt1) - (F_in / V1) * Xt1]

        Q_glc = mu / Y_Xglc + m_glc
        dxdt += [-Q_glc * Xv1 + (F_in / V1) * (GLC_in - GLC1) + (F_r / V1) * (GLCr - GLC1)]

        m_gln = (alpha1 * GLN1) / (alpha2 + GLN1)
        Q_gln = mu / Y_Xgln + m_gln
        dxdt += [-Q_gln * Xv1 - K_dgln * GLN1 + (F_in / V1) * (GLN_in - GLN1) + (F_r / V1) * (GLNr - GLN1)]

        Q_lac = Y_lacglc * Q_glc
        dxdt += [Q_lac * Xv1 - (F_in / V1) * LAC1 + (F_r / V1) * (LACr - LAC1)]

        Q_amm = Y_ammgln * Q_gln
        dxdt += [Q_amm * Xv1 + K_dgln * GLN1 - (F_in / V1) * AMM1 + (F_r / V1) * (AMMr - AMM1)]
        dxdt += [m_mabx * Xv1 - (F_in / V1) * mAb1 + (F_r / V1) * (mAbr - mAb1)]

        dxdt += [F_in + F_r - F_1]

        dxdt += [(F_1 / V2) * (Xv1 - Xv2) - (F_r / V2) * (Xvr - Xv2)]
        dxdt += [(F_1 / V2) * (Xt1 - Xt2) - (F_r / V2) * (Xtr - Xt2)]
        dxdt += [(F_1 / V2) * (GLC1 - GLC2) - (F_r / V2) * (GLCr - GLC2)]
        dxdt += [(F_1 / V2) * (GLN1 - GLN2) - (F_r / V2) * (GLNr - GLN2)]
        dxdt += [(F_1 / V2) * (LAC1 - LAC2) - (F_r / V2) * (LACr - LAC2)]
        dxdt += [(F_1 / V2) * (AMM1 - AMM2) - (F_r / V2) * (AMMr - AMM2)]
        dxdt += [(F_1 / V2) * (mAb1 - mAb2) - (F_r / V2) * (mAbr - mAb2)]
        dxdt += [F_1 - F_2 - F_r]

        # Temperature dynamics -- Ben
        # Average mass of an mAb cell is 150 kDa = 2.4908084e-19 g
        dxdt += [
            (F_in / V1) * (T_in - T) + (Delta_H / (rho * cp)) * mu * Xv1 * 2.4908084e-19 + (U / (V1 * rho * cp)) * (
                    Tc - T)]

        return dxdt

    def xdot_scale(self, x, u):
        return cs.vertcat(*self.xdot(x * cs.DM(self.xscale), u * cs.DM(self.uscale))) / cs.DM(self.xscale)


# utilities helper
xscale = np.concatenate((np.array([3e10, 5e10, 5e3, 50, 200, 50, 150, 3000, 2e10, 3e10,5000, 100,
                   250, 100, 200, 3000, 37.0]), np.array([3, 200]), np.ones(1951)))
uscale = np.concatenate((np.array([3000/60, 3000/60, 3000/60, 3000/60, 2000, 2000, 50]), np.array([3000/60]), np.ones(1)))


class UtilsHelper:
    def load_ss(self, res_dir='smpl/datasets/mabenv'):
        xss = np.load(os.path.join(res_dir, 'xss.npy'))
        uss = np.load(os.path.join(res_dir, 'uss.npy'))
        return xss, uss
    
    def load_bounds(self, res_dir='smpl/datasets/mabenv'):
        ulb = np.load(os.path.join(res_dir, 'ulb.npy'))
        uub = np.load(os.path.join(res_dir, 'uub.npy'))
        xlb = np.load(os.path.join(res_dir, 'xlb.npy'))
        xub = np.load(os.path.join(res_dir, 'xub.npy'))
        return ulb, uub, xlb, xub
    
    def modify_ss(self, xss, uss):
        """ If the modification is not required, simply return xss and uss as they are """
        c_inss = xss[14]  # Inlet concentration of mAb (mg/L)
        F_inss = uss[3]  # Inlet flow rate of buffer tank (L/min)
        # Steady state values of buffer tank are determined by the upstream
        F_outss = F_inss  # Outlet flow rate of buffer tank (L/min)
        xss_b = np.array([15, c_inss])  # Liquid height (dm), concentration of mAb (mg/L)
        uss_b = np.array([F_outss])
        # Steady state values for the integrated model
        xss_modify = np.concatenate((xss, xss_b, np.zeros(1951)))  # Downstream SS is 0, since it does not have SS
        uss_modify = np.concatenate((uss, uss_b, np.zeros(1)))  # Downstream SS is 0, since it does not have SS
        return xss_modify, uss_modify

    def prepare_ss(self, res_dir='data'):
        # Load steady state values of upstream
        xss, uss = self.load_ss(res_dir)
        # Modify the steady state values. Project specific. Not necessarily useful for other projects
        xss_modify, uss_modify = self.modify_ss(xss, uss)
        return xss_modify, uss_modify

    def prepare_init(self, xss, uss):
        """ Modify this function based on the demand """
        # Initialize lists for data storage
        t = []
        Xm = []  # MPC
        Um = []
        Xe = []  # EMPC
        Ue = []
        Xs = []  # Steady state
        Us = []
        Xi = []  # Plant integration
        Xie = []
        # Initial states and inputs
        x0 = xss/xscale*0.95 # stochastic! up and down 30% *0.7 or *1.3 #TODO:
        u0 = uss/uscale
        Xm += [x0]  # Scaled
        Xe += [x0]  # Scaled
        Xs += [xss] # Unscaled
        Xi += [x0]
        Xie += [x0]
        return Xm, Um, Xe, Ue, Xs, Us, Xi, Xie, u0, t

    def plot_results(self, Xm, Um, Xe, Ue, Xs, Us, t, num_sim, dt_spl, cl):
        if cl:
            # convert results to numpy arrays
            t += [num_sim * dt_spl]
            Xm = np.array(Xm) * xscale
            Um = np.array(Um) * uscale
            Xe = np.array(Xe) * xscale
            Ue = np.array(Ue) * uscale
            Xs = np.array(Xs)
            Us = np.array(Us)
            self.compare_controllers(t, Xm, Um, Xe, Ue, Xs, Us)
            return Xm, Um, Xe, Ue, Xs, Us, t

        else:
            # convert results to numpy arrays
            t += [num_sim * dt_spl]
            Xm = np.array(Xm) * xscale
            Um = np.array(Um) * uscale
            Xs = np.array(Xs)
            Us = np.array(Us)
            X, U = self.process_results(Xm, Um)  # Convert results to numpy arrays
            # utils.save_results(t, X, U)  # Save results
            self.visualize_results_all(t, X, U)  # Visualize results
            return Xm, Um, Xe, Ue, Xs, Us, t

    def compare_controllers(self, t, Xm, Um, Xe, Ue, Xs, Us):
        t2 = t[:-1]  # Time list for inputs
        # create figure (fig), and array of axes (ax)
        plt.close("all")

        # Upstream
        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, Xm[:, 0])
        plt.plot(t, Xe[:, 0])
        plt.plot(t, Xs[:, 0], linestyle="--")
        plt.ylabel('$X_{v, reactor}$ (cell/L)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 8])
        plt.plot(t, Xe[:, 8])
        plt.plot(t, Xs[:, 8], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$X_{v, separator}$ (cell/L)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_state1.png")
        plt.show()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(t, Xm[:, 1])
        plt.plot(t, Xe[:, 1])
        plt.plot(t, Xs[:, 1], linestyle="--")
        plt.ylabel('$X_{t, reactor}$ (cell/L)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 9])
        plt.plot(t, Xe[:, 9])
        plt.plot(t, Xs[:, 9], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$X_{t, separator}$ (cell/L)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_state2.png")
        plt.show()

        plt.figure(3)
        plt.subplot(211)
        plt.plot(t, Xm[:, 2])
        plt.plot(t, Xe[:, 2])
        plt.plot(t, Xs[:, 2], linestyle="--")
        plt.ylabel('$[GLC]_{reactor}$ (mM)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 10])
        plt.plot(t, Xe[:, 10])
        plt.plot(t, Xs[:, 10], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$[GLC]_{separator}$ (mM)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_state3.png")
        plt.show()

        plt.figure(4)
        plt.subplot(211)
        plt.plot(t, Xm[:, 3])
        plt.plot(t, Xe[:, 3])
        plt.plot(t, Xs[:, 3], linestyle="--")
        plt.ylabel('$[GLN]_{reactor}$ (mM)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 11])
        plt.plot(t, Xe[:, 11])
        plt.plot(t, Xs[:, 11], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$[GLN]_{separator}$ (mM)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_state4.png")
        plt.show()

        plt.figure(5)
        plt.subplot(211)
        plt.plot(t, Xm[:, 4])
        plt.plot(t, Xe[:, 4])
        plt.plot(t, Xs[:, 4], linestyle="--")
        plt.ylabel('$[LAC]_{reactor}$ (mM)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 12])
        plt.plot(t, Xe[:, 12])
        plt.plot(t, Xs[:, 12], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$[LAC]_{separator}$ (mM)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_state5.png")
        plt.show()

        plt.figure(6)
        plt.subplot(211)
        plt.plot(t, Xm[:, 5])
        plt.plot(t, Xe[:, 5])
        plt.plot(t, Xs[:, 5], linestyle="--")
        plt.xlabel('')
        plt.ylabel('$[AMM]_{reactor}$ (mM)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 13])
        plt.plot(t, Xe[:, 13])
        plt.plot(t, Xs[:, 13], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$[AMM]_{separator}$ (mM)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_state6.png")
        plt.show()

        plt.figure(7)
        plt.subplot(211)
        plt.plot(t, Xm[:, 6])
        plt.plot(t, Xe[:, 6])
        plt.plot(t, Xs[:, 6], linestyle="--")
        plt.xlabel('')
        plt.ylabel('$[mAb]_{reactor}$ (mg/L)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 14])
        plt.plot(t, Xe[:, 14])
        plt.plot(t, Xs[:, 14], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$[mAb]_{separator}$ (mg/L)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_state7.png")
        plt.show()

        plt.figure(8)
        plt.subplot(211)
        plt.plot(t, Xm[:, 7])
        plt.plot(t, Xe[:, 7])
        plt.plot(t, Xs[:, 7], linestyle="--")
        plt.xlabel('')
        plt.ylabel('$V_{reactor}$ (L)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 15])
        plt.plot(t, Xe[:, 15])
        plt.plot(t, Xs[:, 15], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$V_{separator}$ (L)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_state8.png")
        plt.show()

        plt.figure(9)
        plt.subplot(211)
        plt.plot(t, Xm[:, 16])
        plt.plot(t, Xe[:, 16])
        plt.plot(t, Xs[:, 16], linestyle="--")
        plt.ylabel('T ($^\circ$C)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.step(t2, Um[:, 6], where='post')
        plt.step(t2, Ue[:, 6], where='post')
        plt.plot(t2, Us[:, 6], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$T_c$ ($^\circ$C)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_state9.png")
        plt.show()

        plt.figure(10)
        plt.subplot(211)
        plt.step(t2, Um[:, 0], where='post')
        plt.step(t2, Ue[:, 0], where='post')
        plt.plot(t2, Us[:, 0], linestyle="--")
        plt.ylabel('$F_{in}$ (L/min)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.step(t2, Um[:, 1], where='post')
        plt.step(t2, Ue[:, 1], where='post')
        plt.plot(t2, Us[:, 1], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$F_1$ (L/min)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_input1.png")
        plt.show()

        plt.figure(11)
        plt.subplot(211)
        plt.step(t2, Um[:, 2], where='post')
        plt.step(t2, Ue[:, 2], where='post')
        plt.plot(t2, Us[:, 2], linestyle="--")
        plt.ylabel('$F_r$ (L/min)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.step(t2, Um[:, 3], where='post')
        plt.step(t2, Ue[:, 3], where='post')
        plt.plot(t2, Us[:, 3], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$F_2$ (L/min)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_input2.png")
        plt.show()

        plt.figure(12)
        plt.subplot(211)
        plt.step(t2, Um[:, 4], where='post')
        plt.step(t2, Ue[:, 4], where='post')
        plt.plot(t2, Us[:, 4], linestyle="--")
        plt.ylabel('$[GLC]_{in}$ (mM)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.step(t2, Um[:, 5], where='post')
        plt.step(t2, Ue[:, 5], where='post')
        plt.plot(t2, Us[:, 5], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$[GLN]_{in}$ (mM)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/upstream_input3.png")
        plt.show()

        # Buffer tank
        plt.figure(13)
        plt.subplot(211)
        plt.step(t2, Um[:, 3], where='post')
        plt.step(t2, Ue[:, 3], where='post')
        plt.plot(t2, Us[:, 3], linestyle="--")
        plt.ylabel('$F_{in, buffer tank}$ (L/min)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.step(t2, Um[:, 7], where='post')
        plt.step(t2, Ue[:, 7], where='post')
        plt.plot(t2, Us[:, 7], linestyle='--')
        plt.xlabel('Time (min)')
        plt.ylabel('$F_{out, buffer tank}$ (L/min)')
        plt.legend(['PID-MPC', 'PID-EMPC', 'SS'])
        plt.tight_layout()
        plt.savefig("results/integratedmodel/buffer_input1.png")
        plt.show()

        plt.figure(14)
        plt.subplot(211)
        plt.plot(t, Xm[:, 17])
        plt.plot(t, Xe[:, 17])
        plt.plot(t, Xs[:, 17], linestyle="--")
        plt.ylabel('$h_{buffertank}$ (dm)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 18])
        plt.plot(t, Xe[:, 18])
        plt.plot(t, Xs[:, 18], linestyle="--")
        plt.xlabel('Time (min)')
        plt.ylabel('$[mAb]_{buffertank}$ (mg/L)')
        plt.tight_layout()
        plt.savefig("results/integratedmodel/buffer_state1.png")
        plt.show()

        # Downstream
        plt.figure(11)
        plt.subplot(231)
        plt.plot(t, Xm[:, 19+13 * 1 + 1])
        plt.plot(t, Xm[:, 19+13 * 75 + 1])
        plt.plot(t, Xm[:, 19+13 * 149 + 1])
        plt.plot(t, Xe[:, 19+13 * 1 + 1], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 75 + 1], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 149 + 1], linestyle='--')
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$c_p$ (mg/L) r=1")
        plt.xlabel("Time (min)")
        plt.subplot(232)
        plt.plot(t, Xm[:, 19+13 * 1 + 5])
        plt.plot(t, Xm[:, 19+13 * 75 + 5])
        plt.plot(t, Xm[:, 19+13 * 149 + 5])
        plt.plot(t, Xe[:, 19+13 * 1 + 5], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 75 + 5], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 149 + 5], linestyle='--')
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$c_p$ (mg/L) r=5")
        plt.xlabel("Time (min)")
        plt.subplot(233)
        plt.plot(t, Xm[:, 19+13 * 1 + 10])
        plt.plot(t, Xm[:, 19+13 * 75 + 10])
        plt.plot(t, Xm[:, 19+13 * 149 + 10])
        plt.plot(t, Xe[:, 19+13 * 1 + 10], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 75 + 10], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 149 + 10], linestyle='--')
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$c_p$ (mg/L) r=10")
        plt.xlabel("Time (min)")
        plt.subplot(234)
        plt.plot(t, Xm[:, 19+13 * 1 + 0])
        plt.plot(t, Xm[:, 19+13 * 75 + 0])
        plt.plot(t, Xm[:, 19+13 * 149 + 0])
        plt.plot(t, Xe[:, 19+13 * 1 + 0], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 75 + 0], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 149 + 0], linestyle='--')
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$[mAb]_{column}$ (mg/L)")
        plt.xlabel("Time (min)")
        plt.subplot(235)
        plt.plot(t, Xm[:, 19+13 * 1 + 11])
        plt.plot(t, Xm[:, 19+13 * 75 + 11])
        plt.plot(t, Xm[:, 19+13 * 149 + 11])
        plt.plot(t, Xe[:, 19+13 * 1 + 11], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 75 + 11], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 149 + 11], linestyle='--')
        # plt.ylim((32, 35))
        plt.ylabel("$q_1$ (mg/L)")
        plt.xlabel("Time (min)")
        plt.subplot(236)
        plt.plot(t, Xm[:, 19+13 * 1 + 12])
        plt.plot(t, Xm[:, 19+13 * 75 + 12])
        plt.plot(t, Xm[:, 19+13 * 149 + 12])
        plt.plot(t, Xe[:, 19+13 * 1 + 12], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 75 + 12], linestyle='--')
        plt.plot(t, Xe[:, 19+13 * 149 + 12], linestyle='--')
        # plt.ylim((71, 74))
        plt.ylabel("$q_2$ (mg/L)")
        plt.xlabel("Time (min)")
        plt.legend(['MPC level 1', 'level 25', 'level 49', 'EMPC level 1', 'level 25', 'level 49'])
        plt.tight_layout()
        plt.savefig("results/integratedmodel/downstream_state1.png")
        plt.show()

    def visualize_results_all(self, t, X, U):
        t2 = t[:-1]  # Time list for inputs
        plt.close("all")

        # States
        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, X[:, 0])
        plt.ylabel('$X_{v, reactor}$ (cell/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 8])
        plt.xlabel('Time (min)')
        plt.ylabel('$X_{v, separator}$ (cell/L)')
        plt.savefig("results/integratedmodel_ol/upstream_state1.png")
        plt.show()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(t, X[:, 1])
        plt.ylabel('$X_{t, reactor}$ (cell/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 9])
        plt.xlabel('Time (min)')
        plt.ylabel('$X_{t, separator}$ (cell/L)')
        plt.savefig("results/integratedmodel_ol/upstream_state2.png")
        plt.show()

        plt.figure(3)
        plt.subplot(211)
        plt.plot(t, X[:, 2])
        plt.ylabel('$[GLC]_{reactor}}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 10])
        plt.xlabel('Time (min)')
        plt.ylabel('$[GLC]_{separator}}$ (mM)')
        plt.savefig("results/integratedmodel_ol/upstream_state3.png")
        plt.show()

        plt.figure(4)
        plt.subplot(211)
        plt.plot(t, X[:, 3])
        plt.ylabel('$[GLN]_{reactor}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 11])
        plt.xlabel('Time (min)')
        plt.ylabel('$[GLN]_{separator}$ (mM)')
        plt.savefig("results/integratedmodel_ol/upstream_state4.png")
        plt.show()

        plt.figure(5)
        plt.subplot(211)
        plt.plot(t, X[:, 4])
        plt.ylabel('$[LAC]_{reactor}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 12])
        plt.xlabel('Time (min)')
        plt.ylabel('$[LAC]_{separator}$ (mM)')
        plt.savefig("results/integratedmodel_ol/upstream_state5.png")
        plt.show()

        plt.figure(6)
        plt.subplot(211)
        plt.plot(t, X[:, 5])
        plt.ylabel('$[AMM]_{reactor}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 13])
        plt.xlabel('Time (min)')
        plt.ylabel('$[AMM]_{separator}$ (mM)')
        plt.savefig("results/integratedmodel_ol/upstream_state6.png")
        plt.show()

        plt.figure(7)
        plt.subplot(211)
        plt.plot(t, X[:, 6])
        plt.ylabel('$[mAb]_{reactor}$ (mg/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 14])
        plt.xlabel('Time (min)')
        plt.ylabel('$[mAb]_{separator}$ (mg/L)')
        plt.savefig("results/integratedmodel_ol/upstream_state7.png")
        plt.show()

        plt.figure(8)
        plt.subplot(211)
        plt.plot(t, X[:, 7])
        plt.ylabel('$V_{reactor}$ (L)')
        plt.subplot(212)
        plt.plot(t, X[:, 15])
        plt.xlabel('Time (min)')
        plt.ylabel('$V_{separator}$ (L)')
        plt.savefig("results/integratedmodel_ol/upstream_state8.png")
        plt.show()

        plt.figure(9)
        plt.subplot(211)
        plt.plot(t, X[:, 16])
        plt.ylabel('T ($^\circ$C)')
        plt.subplot(212)
        plt.step(t2, U[:, 6], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$T_c$ ($^\circ$C)')
        plt.savefig("results/integratedmodel_ol/upstream_state9.png")
        plt.show()

        plt.figure(10)
        plt.subplot(211)
        plt.plot(t, X[:, 17])
        plt.ylabel('$h_{buffer tank}$ (dm)')
        plt.subplot(212)
        plt.plot(t, X[:, 18])
        plt.xlabel('Time (min)')
        plt.ylabel('$[mAb]_{buffer tank}$ (mg/L)')
        plt.savefig("results/integratedmodel_ol/buffer_state1.png")
        plt.show()

        plt.figure(11)
        plt.subplot(231)
        plt.plot(t, X[:, 19+13 * 1 + 1])
        plt.plot(t, X[:, 19+13 * 75 + 1])
        plt.plot(t, X[:, 19+13 * 149 + 1])
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$c_p$ (mg/L) r=1")
        plt.xlabel("t (min)")
        plt.subplot(232)
        plt.plot(t, X[:, 19+13 * 1 + 5])
        plt.plot(t, X[:, 19+13 * 75 + 5])
        plt.plot(t, X[:, 19+13 * 149 + 5])
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$c_p$ (mg/L) r=5")
        plt.xlabel("t (min)")
        plt.subplot(233)
        plt.plot(t, X[:, 19+13 * 1 + 10])
        plt.plot(t, X[:, 19+13 * 75 + 10])
        plt.plot(t, X[:, 19+13 * 149 + 10])
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$c_p$ (mg/L) r=10")
        plt.xlabel("t (min)")
        plt.subplot(234)
        plt.plot(t, X[:, 19+13 * 1 + 0])
        plt.plot(t, X[:, 19+13 * 75 + 0])
        plt.plot(t, X[:, 19+13 * 149 + 0])
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$[mAb]_{column}$ (mg/L)")
        plt.xlabel("t (min)")
        plt.subplot(235)
        plt.plot(t, X[:, 19+13 * 1 + 11])
        plt.plot(t, X[:, 19+13 * 75 + 11])
        plt.plot(t, X[:, 19+13 * 149 + 11])
        # plt.ylim((32, 35))
        plt.ylabel("$q_1$ (mg/L)")
        plt.xlabel("t (min)")
        plt.subplot(236)
        plt.plot(t, X[:, 19+13 * 1 + 12])
        plt.plot(t, X[:, 19+13 * 75 + 12])
        plt.plot(t, X[:, 19+13 * 149 + 12])
        # plt.ylim((71, 74))
        plt.ylabel("$q_2$ (mg/L)")
        plt.xlabel("t (min)")
        plt.legend(['level 1', 'level 25', 'level 49'])
        plt.tight_layout()
        plt.savefig("results/integratedmodel_ol/downstream_state1.png")
        plt.show()

        # Inputs
        plt.figure(12)
        plt.subplot(211)
        plt.step(t2, U[:, 0], where='post')
        plt.ylabel('$F_{in}$ (L/min)')
        plt.subplot(212)
        plt.step(t2, U[:, 1], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$F_1$ (L/min)')
        plt.savefig("results/integratedmodel_ol/upstream_input1.png")
        plt.show()

        plt.figure(13)
        plt.subplot(211)
        plt.step(t2, U[:, 2], where='post')
        plt.ylabel('$F_r$ (L/min)')
        plt.subplot(212)
        plt.step(t2, U[:, 3], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$F_2$ (L/min)')
        plt.savefig("results/integratedmodel_ol/upstream_input2.png")
        plt.show()

        plt.figure(14)
        plt.subplot(211)
        plt.step(t2, U[:, 4], where='post')
        plt.ylabel('$[GLC]_{in}$ (mM)')
        plt.subplot(212)
        plt.step(t2, U[:, 5], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$[GLN]_{in}$ (mM)')
        plt.savefig("results/integratedmodel_ol/upstream_input3.png")
        plt.show()

        plt.figure(16)
        plt.subplot(211)
        plt.step(t2, U[:, 3], where='post')
        plt.ylabel('$F_{in, buffer tank}$ (L/min)')
        plt.subplot(212)
        plt.step(t2, U[:, 7], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$F_{out, buffer tank}$ (L/min)')
        plt.savefig("results/integratedmodel_ol/buffer_input1.png")
        plt.show()

    def process_results(self, X, U):
        X = np.array(X)
        U = np.array(U)
        return X, U
