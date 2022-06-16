import mpctools as mpc
import casadi as cs
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import os

class MabModelHelper():
    def __init__(self, xscale, uscale, Nx, Nu, dt, xss, uss):
        self.xscale = xscale
        self.uscale = uscale
        self.Nx = Nx
        self.Nu = Nu
        self.dt = dt
        self.xss = xss
        self.uss = uss

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

    def reactor_tank_column(self, x, u):
        """
        An integrated model including the upstream, the buffer tank, and capture column in the downstream.
        :param self:
        :param x: States vector (1969, 1)
        :param u: Input vector (8, 1)
        :return: ODEs (1969, 1)
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
        mu_dmax = (-0.0045 * x[16] + 0.1682) / 60  # 0.06    # maximum specific death rate (min^(-1))
        m_mabx = 6.59e-10 / 60  # constant production of mAb by viable cells (mg/Cell/min)

        # Add new parameters -- Ben
        Delta_H = 5e5  # 4.4e4# 1e-8
        rho = 1560.0  # density of mixture is assumed to be density of glucose [g/L]
        cp_up = 1.244  # specific heat capacity of mixture is assumed to be that of glucose [J/g/dC]
        U = 4e2
        T_in = 37.0

        # Buffer tank
        D_bf = 5  # Diameter of the tank (dm)
        L_bf = D_bf  # Diameter of the tank (dm)
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
        x_column = x[19:]  # All states of capture column
        x_reshape = np.reshape(x_column, (
        self.num_z_cap_load, 1 + self.num_r + 1 + 1))  # Convert from Nx * 1  to Nz *  ( c + cp + q1 + q2)
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
        cp_rp = np.reshape(cp_rp, [self.num_z_cap_load, 1])

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

        # Add constraints on each state
        return dxdt

    def reactor_tank(self, x, u):
        """
        An integrated model including upstream and buffer tank only.
        :param x: State vector (19,1)
        :param u: Input vector (8,1)
        :return: ODEs (19,1)
        """
        # Parameter values of upstream model
        K_damm = 1.76     # ammonia constant for cell death (mM)
        K_dgln = 9.6e-03/60  # constant for glutamine degradation (min^(-1))
        K_glc = 0.75      # Monod constant for glucose (mM)
        K_gln = 0.075     # Monod constant for glutamine (mM)
        KI_amm = 28.48    # Monod constant for ammonia (mM)
        KI_lac = 171.76   # Monod constant for lactate (mM)
        m_glc = 4.9e-14/60   # maintenance coefficients of glucose (mmol/cell/min)
        n = 2             # constant represents the increase of specific death as ammonia concentration increases (-)
        Y_ammgln = 0.45   # yield of ammonia from glutamine (mmol/mmol)
        Y_lacglc = 2.0    # yield of lactate from glucose (mmol/mmol)
        Y_Xglc = 2.6e8    # yield of cells on glucose (cell/mmol)
        Y_Xgln = 8e8      # yield of cells on glutamine (cell/mmol)
        alpha1 = 3.4e-13/60  # constants of glutamine maintenance coefficient (mM L/cell/min)
        alpha2 = 4        # constants of glutamine maintenance coefficient (mM)
        mu_max = (0.0016*x[16] - 0.0308)*5/60 # 0.058 * 5  # maximum specific growth rate (min^(-1))
        mu_dmax = (-0.0045*x[16] + 0.1682)/60 # 0.06    # maximum specific death rate (min^(-1))
        m_mabx = 6.59e-10/60 # constant production of mAb by viable cells (mg/Cell/min)

        # Add new parameters -- Ben
        Delta_H = 5e5     # 4.4e4# 1e-8
        rho = 1560.0      # density of mixture is assumed to be density of glucose [g/L]
        cp = 1.244        # specific heat capacity of mixture is assumed to be that of glucose [J/g/dC]
        U = 4e2
        T_in = 37.0

        # Parameters of buffer tank
        D = 5  # Diameter of the tank (dm)
        L = D  # Diameter of the tank (dm)
        Ac = np.pi*(D/2)**2  # Cross-sectional area (dm^2)
        V = Ac*L  # Volume of the tank (L)

        # Initialize ode list
        dxdt = []

        # Inputs of upstream model:
        F_in = u[0]    # inlet flow rate (L/min)
        F_1 = u[1]     # outlet flow rate of bioreactor (L/min)
        F_r = u[2]     # u[1]-u[3] # u[2]      # recycle flow rate (L/min)
        F_2 = u[3]     # Outlet flow rate of separator (L/min)
        GLC_in = u[4]  # inlet glucose concentration (mM)
        GLN_in = u[5]  # inlet glutamine concentration (mM)
        Tc = u[6]      # coolant temperature (d C)
        # Inputs of buffer tank:
        F_out_bf = u[7]  # Outlet flow rate of the buffer tank (L/min)
        F_in_bf = u[3]   # Inlet flow rate of the buffer tank (L/min)
        c_in_bf = x[14]   # Inlet concentration of mAb (mg/L)

        # States:
        Xv1 = x[0]   # concentration of viable cells in reactor(cell/L)
        Xt1 = x[1]   # total concentration of cells in reactor(cell/L)
        GLC1 = x[2]  # glucose concentration in reactor(mM)
        GLN1 = x[3]  # glutamine concentration in reactor(mM)
        LAC1 = x[4]  # lactate concentration in reactor(mM)
        AMM1 = x[5]  # ammonia concentration in reactor(mM)
        mAb1 = x[6]  # mAb concentration in reactor(mg/L)
        V1 = x[7]    # volume in reactor (L)
        Xv2 = x[8]    # concentration of viable cells in separator(cell/L)
        Xt2 = x[9]    # total concentration of cells in separator(cell/L)
        GLC2 = x[10]  # glucose concentration in separator(mM)
        GLN2 = x[11]  # glutamine concentration in separator(mM)
        LAC2 = x[12]  # lactate concentration in separator(mM)
        AMM2 = x[13]  # ammonia concentration in separator(mM)
        mAb2 = x[14]  # mAb concentration in separator(mg/L)
        V2 = x[15]    # volume in separator (L)
        # temperature as a state -- Ben
        T = x[16]     # (d C)
        # buffer tank
        h = x[17]  # Liquid level in the tank (dm)
        c = x[18]  # Concentration of mAb in the tank (mg/L)

        # Assumptions: 92% cell recycle rate and 20% mab retention
        Xvr = 0.92 * Xv1 * F_1 / F_r   # concentration of viable cells in recycle stream(cell/L)
        Xtr = 0.92 * Xt1 * F_1 / F_r   # total concentration of cells in recycle stream(cell/L)
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

        # ODEs of buffer tank:
        dhdt = 1/Ac*(F_in_bf-F_out_bf)
        dcdt = F_in_bf/(Ac*h)*(c_in_bf-c)
        dxdt += [dhdt]
        dxdt += [dcdt]

        # Add constraints on each state

        return dxdt

    def pcontroller(self, y):
        """
        A P controller taking y (liquid level of buffer tank) as input and returning p (outlet flow rate)
        :param y: input (liquid level of buffer tank)
        :return: output (outlet flow rate of buffer tank)
        """
        # Parameters
        Kc = -1/60  #
        p_ss = self.uss[7]  # Steady state of outlet flow rate
        y_ss = self.xss[17]  # Steady state of liquid level

        # Implement P controller
        p = p_ss + Kc*(y_ss-y)  # The objective of controller is to maintain liquid level at y_ss
        return p


class UpModelHelper():
    def __init__(self, xscale, uscale, Nx, Nu, dt):
        self.xscale = xscale
        self.uscale = uscale
        self.Nx = Nx
        self.Nu = Nu
        self.dt = dt

    def reactor(self, x, u):
        # Parameter values
        K_damm = 1.76     # ammonia constant for cell death (mM)
        K_dgln = 9.6e-03  # constant for glutamine degradation (h^(-1))
        K_glc = 0.75      # Monod constant for glucose (mM)
        K_gln = 0.075     # Monod constant for glutamine (mM)
        KI_amm = 28.48    # Monod constant for ammonia (mM)
        KI_lac = 171.76   # Monod constant for lactate (mM)
        m_glc = 4.9e-14   # maintenance coefficients of glucose (mmol/cell/h)
        n = 2             # constant represents the increase of specific death as ammonia concentration increases (-)
        Y_ammgln = 0.45   # yield of ammonia from glutamine (mmol/mmol)
        Y_lacglc = 2.0    # yield of lactate from glucose (mmol/mmol)
        Y_Xglc = 2.6e8    # yield of cells on glucose (cell/mmol)
        Y_Xgln = 8e8      # yield of cells on glutamine (cell/mmol)
        alpha1 = 3.4e-13  # constants of glutamine maintenance coefficient (mM L/cell/h)
        alpha2 = 4        # constants of glutamine maintenance coefficient (mM)
        mu_max = (0.0016*x[16] - 0.0308)*5 # 0.058 * 5  # maximum specific growth rate (h^(-1))
        mu_dmax = (-0.0045*x[16] + 0.1682) # 0.06    # maximum specific death rate (h^(-1))
        m_mabx = 6.59e-10 # constant production of mAb by viable cells (mg/Cell/h)

        # Add new parameters -- Ben
        Delta_H = 5e5     # 4.4e4# 1e-8
        rho = 1560.0      # density of mixture is assumed to be density of glucose [g/L]
        cp = 1.244        # specific heat capacity of mixture is assumed to be that of glucose [J/g/dC]
        U = 4e2
        T_in = 37.0

        # Initialize ode list
        dxdt = []

        # Inputs:
        F_in = u[0]    # inlet flow rate (L/h)
        F_1 = u[1]     # outlet flow rate of bioreactor (L/h)
        F_r = u[2]     # u[1]-u[3] # u[2]      # recycle flow rate (L/h)
        F_2 = u[3]     # Outlet flow rate of separator (L/h)
        GLC_in = u[4]  # inlet glucose concentration (mM)
        GLN_in = u[5]  # inlet glutamine concentration (mM)
        Tc = u[6]      # coolant temperature (d C)

        # States:
        Xv1 = x[0]   # concentration of viable cells in reactor(cell/L)
        Xt1 = x[1]   # total concentration of cells in reactor(cell/L)
        GLC1 = x[2]  # glucose concentration in reactor(mM)
        GLN1 = x[3]  # glutamine concentration in reactor(mM)
        LAC1 = x[4]  # lactate concentration in reactor(mM)
        AMM1 = x[5]  # ammonia concentration in reactor(mM)
        mAb1 = x[6]  # mAb concentration in reactor(mg/L)
        V1 = x[7]    # volume in reactor (L)
        Xv2 = x[8]    # concentration of viable cells in separator(cell/L)
        Xt2 = x[9]    # total concentration of cells in separator(cell/L)
        GLC2 = x[10]  # glucose concentration in separator(mM)
        GLN2 = x[11]  # glutamine concentration in separator(mM)
        LAC2 = x[12]  # lactate concentration in separator(mM)
        AMM2 = x[13]  # ammonia concentration in separator(mM)
        mAb2 = x[14]  # mAb concentration in separator(mg/L)
        V2 = x[15]    # volume in separator (L)
        # temperature as a state -- Ben
        T = x[16]

        # Assumptions: 92% cell recycle rate and 20% mab retention
        Xvr = 0.92 * Xv1 * F_1 / F_r   # concentration of viable cells in recycle stream(cell/L)
        Xtr = 0.92 * Xt1 * F_1 / F_r   # total concentration of cells in recycle stream(cell/L)
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

    def create_plant(self):
        self.simulator = mpc.DiscreteSimulator(self.reactor, self.dt, [self.Nx,self.Nu], ["x","u"])

    def xdot_scale(self, x, u):
        return cs.vertcat(*self.reactor(x * cs.DM(self.xscale), u * cs.DM(self.uscale))) / cs.DM(self.xscale)

    def ss_optimization(self):
        x = cs.SX.sym('x', self.Nx)
        u = cs.SX.sym('u', self.Nu)

        def lecost(x, u):
            xx = x * self.xscale
            uu = u * self.uscale
            return - xx[6] * uu[1] - x[14] * uu[3]

        dvar = cs.vertcat(x, u)
        le = lecost(x, u)
        xdot = cs.vertcat(self.xdot_scale(x, u))
        xlcon = cs.DM(np.ones(self.Nx) * 1e-30)
        xlcon[16] = 33.0 / 37.0

        xucon = cs.DM(self.xscale / self.xscale) * cs.inf
        xucon[16] = 37.0 / self.xscale[-1]
        xucon[15] = 1
        xucon[7] = 1
        ulcon = cs.DM(self.uscale * 0.0001 / self.uscale)
        uucon = cs.DM(self.uscale / self.uscale)

        dvarlcon = cs.vertcat(xlcon, ulcon)
        dvarucon = cs.vertcat(xucon, uucon)

        gcon = cs.DM(np.zeros(self.Nx))

        prob = {'x': dvar, 'f': le, 'g': xdot}

        # NLP solver options
        opts = {}
        opts["expand"] = True
        # uncomment this line to let ipopt suprress any outputs
        # opts["ipopt.print_level"] = 0
        opts["verbose"] = False
        opts["ipopt.linear_solver"] = "mumps" # use ma27 for faster and more robust computer. This needs to be installed on the system
        # opts["hessian_approximation"] = "limited-memory"

        nlp = cs.nlpsol('ss_optimization', 'ipopt', prob, opts)
        res = nlp(lbx=dvarlcon, ubx=dvarucon, lbg=gcon, ubg=gcon,
                  x0=cs.vertcat(1 * self.xscale / self.xscale, 1 * self.uscale / self.uscale))
        sol = res['x'].full().ravel()
        print("Steady-state complete")
        return sol[:self.Nx] * self.xscale, sol[self.Nx:] * self.uscale

    def build_mpc_controller(self, xss, uss, Q, R, N, dt):
        Nx = self.Nx
        Nu = self.Nu

        def stage_cost(x, u):
            xd = x - xss / self.xscale
            ud = u - uss / self.uscale
            return mpc.mtimes(xd.T, Q, xd) + mpc.mtimes(ud.T, R, ud)

        lfunc = mpc.getCasadiFunc(stage_cost, [Nx, Nu], ['x', 'u'], 'lfunc')

        xlb = xss * 0.8 / self.xscale  # np.ones(self.Nx)*1e-10
        xlb[16] = 33.0 / 37.0

        xub = 1.2 * xss / self.xscale
        xub[7] = 1
        xub[15] = 1
        xub[16] = 1

        ulb = 0.8 * uss / self.uscale
        uub = self.uscale / self.uscale

        contargs = dict(
            N={"t": N, "x": Nx, "u": Nu, "c": 3},  # , "e":Ns, "s":Ns},
            verbosity=0,
            l=lfunc,
            # e=zfunc,
            x0=xss / xss,
            ub={"u": uub, "x": xub},  # Change upper bounds
            lb={"u": ulb, "x": xlb},  # Change lower bounds
            guess={
                "x": xss / self.xscale,
                "u": uss / self.uscale  # [:Nu]
            }
        )

        ctrl = mpc.nmpc(f=self.xdot_scale,
                        Delta=dt, timelimit=120,
                        discretel=False,
                        **contargs,
                        )

        return ctrl

    def build_empc_controller(self, xss, uss, N, dt):
        Nx = self.Nx
        Nu = self.Nu

        def stage_cost(x, u):
            xx = x * cs.DM(self.xscale)
            uu = u * cs.DM(self.uscale)
            return - xx[6] * uu[1] - x[14] * uu[3]

        lfunc = mpc.getCasadiFunc(stage_cost, [Nx, Nu], ['x', 'u'], 'lfunc')

        xlb = xss * 0.8 / self.xscale  # np.ones(self.Nx)*1e-10
        xlb[16] = 33.0 / 37.0

        xub = 1.2 * xss / self.xscale
        xub[7] = 1
        xub[15] = 1
        xub[16] = 1

        ulb = 0.8 * uss / self.uscale
        uub = self.uscale / self.uscale

        contargs = dict(
            N={"t": N, "x": Nx, "u": Nu, "c": 3},  # , "e":Ns, "s":Ns},
            verbosity=0,
            l=lfunc,
            # e=zfunc,
            x0=xss / self.xscale,
            ub={"u": uub, "x": xub},  # Change upper bounds
            lb={"u": ulb, "x": xlb},  # Change lower bounds
            guess={
                "x": xss / self.xscale,
                "u": uss / self.uscale  # [:Nu]
            }
        )

        ctrl = mpc.nmpc(f=self.xdot_scale,
                        Delta=dt, timelimit=120,
                        discretel=False,
                        **contargs,
                        )

        return ctrl


class BufferTankHelper():
    """
    A buffer tank is added in between upstream and downstream. The objective is to improve the robustness of control of
    downstream
    """
    def __init__(self, Nx, Nu, dt, xss, uss):
        self.Nx = Nx
        self.Nu = Nu
        self.dt = dt
        self.xss = xss
        self.uss = uss

    def tank(self, x, u):
        # Parameters
        D = 5  # Diameter of the tank (dm)
        L = D  # Diameter of the tank (dm)
        Ac = np.pi*(D/2)**2  # Cross-sectional area (dm^2)
        V = Ac*L  # Volume of the tank (L)

        # Initialize ode list
        dxdt = []

        # Inputs:
        F_out = u[0]  # Outlet flow rate of the buffer tank
        F_in = u[1]   # Inlet flow rate of the buffer tank
        c_in = u[2]   # Inlet concentration of mAb

        # States:
        h = x[0]  # Liquid level in the tank
        c = x[1]  # Concentration of mAb in the tank

        # ODEs
        dhdt = 1/Ac*(F_in-F_out)
        dcdt = F_in/(Ac*h)*(c_in-c)

        dxdt += [dhdt]
        dxdt += [dcdt]

        return dxdt

    def pcontroller(self, y):
        # Parameters
        Kc = -1  #
        p_ss = self.uss[0]  # Steady state of outlet flow rate
        y_ss = self.xss[0]  # Steady state of liquid level

        # Implement P controller
        p = p_ss + Kc*(y_ss-y)
        return p


class DownModelHelper():
    def __init__(self, num_z_cap_load,
                 num_z_cap_elu,
                 num_z_cex_load,
                 num_z_cex_elu,
                 num_z_loop,
                 num_z_aex,
                 num_r,
                 num_sim,
                 delta_t,
                 init_states,
                 inputs):
        # Define discretization points
        self.num_z_cap_load = num_z_cap_load  # discretization points in Z direction
        self.num_r = num_r  # discretization points in r direction
        self.num_z_cap_elu = num_z_cap_elu  # discretization points in Z direction
        self.num_z_cex_load = num_z_cex_load  # discretization points in Z direction
        self.num_z_cex_elu = num_z_cex_elu  # discretization points in Z direction for
        self.num_z_loop = num_z_loop  # discretization points in Z direction
        self.num_z_aex = num_z_aex  # discretization points in Z direction
        self.num_sim = num_sim  # number of simulation points
        self.delta_t = delta_t  # time step size [min]
        self.x0 = init_states  # initial states
        self.u = inputs  # input
        self.period = 7.2*60  # [min] The nominal period

        # # Capture column loading process
        # Define state number
        num_c_cap_load = self.num_z_cap_load  # Number of c states
        num_cp_cap_load = self.num_z_cap_load * self.num_r  # Number of cp states
        num_q1_cap_load = self.num_z_cap_load  # Number of q1 states
        num_q2_cap_load = self.num_z_cap_load  # Number of q2 states
        self.num_x_cap_load = num_c_cap_load + num_cp_cap_load + num_q1_cap_load + num_q2_cap_load  # Total number of states

        # # Capture column elution process
        # Define state number
        num_c_cap_elu = self.num_z_cap_elu * 1  # Number of c states
        num_q_cap_elu = self.num_z_cap_elu * 1  # Number of q states
        num_cs_cap_elu = self.num_z_cap_elu * 1  # Number of cs states
        self.num_x_cap_elu = num_c_cap_elu + num_q_cap_elu + num_cs_cap_elu  # Total number of states

        # # CEX loading process
        # Define state number
        num_c_cex_load = self.num_z_cex_load
        num_cp_cex_load = self.num_z_cex_load * self.num_r
        num_q1_cex_load = self.num_z_cex_load
        num_q2_cex_load = self.num_z_cex_load
        self.num_x_cex_load = num_c_cex_load + num_cp_cex_load + num_q1_cex_load + num_q2_cex_load # Total number of states

        # # CEX elution process
        # Define state number
        num_c_cex_elu = self.num_z_cex_elu * 1
        num_q_cex_elu = self.num_z_cex_elu * 1
        num_cs_cex_elu = self.num_z_cex_elu * 1
        self.num_x_cex_elu = num_c_cex_elu + num_q_cex_elu + num_cs_cex_elu  # Total number of states

        # # Hold-up loop process
        # Define state number
        num_c_loop = self.num_z_loop * 1
        self.num_x_loop = num_c_loop

        # # AEX process
        # Define state number
        num_c_aex = self.num_z_aex * 1  # State number for variable c
        self.num_x_aex = num_c_aex  # Total number of states for AEX

        # # Define total state number
        self.num_x_total = self.num_x_cap_load+self.num_x_cap_elu+self.num_x_loop*2+self.num_x_cex_load+self.num_x_cex_elu+self.num_x_loop*2+self.num_x_aex
        self.num_x_1 = self.num_x_cap_load
        self.num_x_2 = self.num_x_cap_elu+self.num_x_loop
        self.num_x_3 = self.num_x_loop+self.num_x_cex_load
        self.num_x_4 = self.num_x_cex_elu+self.num_x_loop
        self.num_x_5 = self.num_x_loop+self.num_x_aex
        self.s1 = 0
        self.e1 = self.num_x_1
        self.e2 = self.e1 + self.num_x_2
        self.e3 = self.e2 + self.num_x_3
        self.e4 = self.e3 + self.num_x_4
        self.e5 = self.e4 + self.num_x_5

        # # Define sizes of columns
        self.vol_cap = 2*50000  # ml
        self.len_cap = 5*4  # cm
        self.vol_cex = 1*50000  # ml
        self.len_cex = 2.5*4  # cm
        self.vol_loop = 10*50000  # ml
        self.len_loop = 150*4  # cm
        self.vol_aex = 1*50000  # ml
        self.len_aex = 2.5*4  # cm

    def capture_load(self, t, x, u):
        # # Capture model
        # # State and input arrays for the process
        x_reshape = np.reshape(x, (self.num_z_cap_load, 1 + self.num_r + 1 + 1))  # Convert from Nx * 1  to Nz *  ( C + CP + q1 + q2)
        c = x_reshape[:, 0]  # mAb concentration in the mobile phase (mg/ml)
        cp = x_reshape[:, 1:self.num_r + 1]  # mAb concentration inside the particle (mg/ml)
        q1 = x_reshape[:, self.num_r + 1:self.num_r + 1 + 1]  # adsorbed mAb concentration (mg/ml)
        q2 = x_reshape[:, self.num_r + 2:1 + self.num_r + 2]

        F = u[0]  # Inlet flow rate/harvest flow rate (ml/min)
        cF = u[1]  # mAb concentration in mobile phase/harvest mAb concentration (mg/ml)

        # # Parameter values
        qmax1 = 36.45  # column capacity (mg/ml)
        k1 = 0.704  # kinetic constant (ml/(mg min))
        qmax2 = 77.85  # column capacity (mg/ml)
        k2 = 0.021  # kinetic constant (ml/(mg min))
        K = 15.3  # Langmuir equilibrium constant (ml/mg)
        Deff = 7.6e-5  # effective pore diffusivity (cm2/min)

        rp = 4.25e-3  # particle radius (cm)
        L = self.len_cap  # * column length (cm)
        V = self.vol_cap  # * column volume (ml)
        column_r = np.sqrt(V / (np.pi * L))  # column radius
        v = F / (np.pi * column_r ** 2)  # * velocity (cm/min)
        # ** v：interstitial velocity (*double check how to calculate it*)

        Dax = 0.55 * v  # axial dispersion coefficient (cm2/min)
        kf = 0.067 * v ** 0.58  # mass transfer coefficient (cm/min)
        epsilon_c = 0.31  # extra-particle column void (-)
        epsilon_p = 0.94  # particle porosity (-)

        dz = L / (self.num_z_cap_load + 1)  # distance delta
        dr = rp / (self.num_r + 1)  # delta radius

        # # ODEs
        # # mAb concentration in mobile phase (c)
        dc = np.zeros(self.num_z_cap_load + 1)
        k = np.arange(1, self.num_z_cap_load)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc[k] = (c[k] - c[k - 1]) / dz
        # Top boundary condition: stands for difference between top and point [0]
        dc[0] = v / (epsilon_c * Dax) * (c[0] - cF)  # check C[0] or C[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc[-1] = 0

        # Second order discretization
        dc2 = np.zeros(self.num_z_cap_load)
        k = np.arange(0, self.num_z_cap_load)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2[k] = (dc[k + 1] - dc[k]) / dz
        # from point 0 to point 49 (total 50 points)
        dc_true = dc[1:self.num_z_cap_load + 1]
        # Cp value at r=rp point
        cp_rp = cp[:, -1]  #*
        # calculate dcdt value
        dcdt = Dax * dc2 - (v / epsilon_c) * dc_true - (1 - epsilon_c) / epsilon_c * 3 / (rp) * kf * (c - cp_rp)
        dcdt = np.reshape(dcdt, [self.num_z_cap_load, 1])

        # # mAb concentration inside the particle
        dcp = np.zeros((self.num_z_cap_load, self.num_r + 1))
        k = np.arange(1, self.num_r)
        # forward difference: @ point 0 = c[1] -c[0],
        # dcp[1] stands for point 0 here.
        dcp[:, k] = (cp[:, k] - cp[:, k - 1]) / dr
        # Top boundary: dcp[0] is difference between r=0 and point 0
        dcp[:, 0] = 0
        # Bottom boundary: dcp[30] stands for the difference between point 29 and r=rp
        dcp[:, -1] = kf / Deff * (c[:] - cp_rp)

        # second order derivation
        dcp2 = np.zeros((self.num_z_cap_load, self.num_r))
        k = np.arange(0, self.num_r)
        r = np.zeros((1, self.num_r))
        r[:, k] = (k + 1) * dr  # particle radius

        dcp2[:, k] = ((((k + 2) * dr) ** 2) * dcp[:, k + 1] - (
                ((k + 1) * dr) ** 2 * dcp[:, k])) / dr
        cp_rp = np.reshape(cp_rp, [self.num_z_cap_load, 1])

        # # absorbed mAb concentration (mg/ml)
        dq1dt = k1 * ((qmax1 - q1) * cp_rp - q1 / K)  # q1: quick adsorption rete
        dq2dt = k2 * ((qmax2 - q2) * cp_rp - q2 / K)  # q2: slow adsorption rete

        dcpdt = np.zeros((self.num_z_cap_load, self.num_r))
        k = np.arange(0, self.num_r)
        dcpdt[:, k] = Deff * (1 / (r[:, k] ** 2)) * dcp2[:, k] - (1 / epsilon_p) * (dq1dt[:] + dq2dt[:])

        dxdt = np.concatenate((dcdt, dcpdt, dq1dt, dq2dt), axis=1)
        dxdt = np.reshape(dxdt, (self.num_x_cap_load))

        return dxdt

    def cap_vi(self, t, x, u):
        x1 = x[0:self.num_z_cap_elu*3]
        x2 = x[self.num_z_cap_elu*3:]
        # # Capture column - elution mode ------------------------------------------------------------------------------
        x_reshape = np.reshape(x1, (self.num_z_cap_elu, 1 + 1 + 1))
        c = x_reshape[:, 0]
        q = x_reshape[:, 1]
        cs = x_reshape[:, 2]

        F = u[0]  # flow rate of elution buffer
        cF = c[-1]  # u[1]
        cS = 0.9  #*

        # # Parameters
        qmax_elu = 114.3
        k_elu = 0.64
        H0_elu = 2.2e-2
        beta_elu = 0.2
        L = self.len_cap  # column length [cm]
        V = self.vol_cap  # column volume [ml]
        column_r = np.sqrt(V / (np.pi * L))  # column radius
        v = F / (np.pi * column_r ** 2)  # velocity
        # ** v：interstitial velocity (*check how to calculate it*)

        Dax = 0.55 * v  # axial dispersion coefficient (cm2/min)
        epsilon_c = 0.31  # extra-particle column void (-)
        epsilon_p = 0.94  # particle porosity (-)
        epsilon = 0.32  # (epsilon_c + epsilon_p) * 1  # total column void

        # # ODEs
        dz = L / (self.num_z_cap_elu + 1)  # distance delta for elution and loading

        # # Mab concentration in mobile phase (c)
        dc = np.zeros(self.num_z_cap_elu + 1)
        k = np.arange(1, self.num_z_cap_elu)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc[k] = (c[k] - c[k - 1]) / dz
        # Top boundary condition: stands for difference between top and point [0]
        dc[0] = v / (epsilon_c * Dax) * (c[0] - 0)  # check C[0] or C[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc[-1] = v / (epsilon_c * Dax) * (cF - c[-1])
        # Second order discretization
        dc2 = np.zeros(self.num_z_cap_elu)
        k = np.arange(0, self.num_z_cap_elu)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2[k] = (dc[k + 1] - dc[k]) / dz
        # from point 0 to point 49 (total 50 points)
        dc_true = dc[1:self.num_z_cap_elu + 1]

        # # Modifier concentration (M)
        # first order discretization
        dcs = np.zeros(self.num_z_cap_elu + 1)
        k = np.arange(1, self.num_z_cap_elu)
        # point 0: dcs[0] = cs[1] - cs[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dcs[k] = (cs[k] - cs[k - 1]) / dz
        # Top boundary condition: stands for difference between top and point [0]
        dcs[0] = v / (epsilon_c * Dax) * (cs[0] - cS)  # check C[0] or C[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dcs[-1] = v / (epsilon_c * Dax) * (cs[-1] - cs[-1])
        # Second order discretization
        dcs2 = np.zeros(self.num_z_cap_elu)
        k = np.arange(0, self.num_z_cap_elu)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dcs2[k] = (dcs[k + 1] - dcs[k]) / dz
        # from point 0 to point 49 (total 50 points)
        dcs_true = dcs[1:self.num_z_cap_elu + 1]

        # dcsdt = Dax*dcs2 - v/epsilon_c*dcs_true
        dcsdt = Dax * dcs2 - v / epsilon * dcs_true

        dqdt = k_elu * (H0_elu * (cs ** (-beta_elu)) * (1 - (q / qmax_elu)) * c - q)

        # calculate dcdt value
        # dcdt = D_ax * dc2 - v/epsilon_c * dc_true + (1-epsilon_c)/epsilon_c * dqdt
        dcdt = Dax * dc2 - v / epsilon * dc_true - (1 - epsilon_c) / epsilon * dqdt

        dcsdt = np.reshape(dcsdt, [self.num_z_cap_elu, 1])
        dqdt = np.reshape(dqdt, [self.num_z_cap_elu, 1])
        dcdt = np.reshape(dcdt, [self.num_z_cap_elu, 1])

        dxdt = np.concatenate((dcdt, dqdt, dcsdt), axis=1)
        dxdt = np.reshape(dxdt, (self.num_x_cap_elu))

        # # Hold-up Loops: ---------------------------------------------------------------------------------------------
        x_reshape_ = np.reshape(x2, (self.num_z_loop, 1))
        c_ = x_reshape_[:, 0]

        F_ = u[0]
        cF_ = c[-1]  # u[1]

        # # Parameters
        L_ = self.len_loop
        V_ = self.vol_loop
        column_r_ = np.sqrt(V_ / (np.pi * L_))  # column radius
        v_ = F_ / (np.pi * column_r_ ** 2)  # velocity
        Dax_ = 290 * v_

        dz_ = L_ / (self.num_z_loop + 1)  # distance delta

        # # Mab concentration in mobile phase (c):
        dc_ = np.zeros(self.num_z_loop + 1)
        k_ = np.arange(1, self.num_z_loop)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc_[k_] = (c_[k_] - c_[k_ - 1]) / dz_
        # Top boundary condition: stands for difference between top and point [0]
        dc_[0] = v_ / (Dax_) * (c_[0] - cF_)  # check C[0] or C[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc_[-1] = 0  # v_ / (Dax_) * (0 - c_[-1])

        # Second order discretization
        dc2_ = np.zeros(self.num_z_loop)
        k_ = np.arange(0, self.num_z_loop)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2_[k_] = (dc_[k_ + 1] - dc_[k_]) / dz_
        # from point 0 to point 49 (total 50 points)
        dc_true_ = dc_[1:self.num_z_loop + 1]

        # calculate dcdt value
        dcdt_ = Dax_ * dc2_ - v_ * dc_true_

        dxdt_ = np.concatenate((dxdt, dcdt_))

        return dxdt_

    def vi_cex(self, t, x, u):
        x1 = x[0:self.num_z_loop]
        x2 = x[self.num_z_loop:]
        # # Hold-up Loops: -------------------------------------------------------------------
        x_reshape_ = np.reshape(x1, (self.num_z_loop, 1))
        c_ = x_reshape_[:, 0]

        F_ = u[0]
        cF_ = c_[-1]  # u[1]

        # # Parameters
        L_ = self.len_loop
        V_ = self.vol_loop
        column_r_ = np.sqrt(V_ / (np.pi * L_))  # column radius
        v_ = F_ / (np.pi * column_r_ ** 2)  # velocity
        Dax_ = 290 * v_

        dz_ = L_ / (self.num_z_loop + 1)  # distance delta

        # # Mab concentration in mobile phase (c):
        dc_ = np.zeros(self.num_z_loop + 1)
        k_ = np.arange(1, self.num_z_loop)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc_[k_] = (c_[k_] - c_[k_ - 1]) / dz_
        # Top boundary condition: stands for difference between top and point [0]
        dc_[0] = v_ / (Dax_) * (c_[0] - 0)
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc_[-1] = v_ / (Dax_) * (cF_ - c_[-1])  # check C[0] or C[-1]

        # Second order discretization
        dc2_ = np.zeros(self.num_z_loop)
        k_ = np.arange(0, self.num_z_loop)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2_[k_] = (dc_[k_ + 1] - dc_[k_]) / dz_
        # from point 0 to point 49 (total 50 points)
        dc_true_ = dc_[1:self.num_z_loop + 1]

        # calculate dcdt value
        dcdt_ = Dax_ * dc2_ - v_ * dc_true_

        # # CEX model ---------------------------------------------------------------------
        # # State and input arrays for the process
        x_reshape = np.reshape(x2, (
        self.num_z_cap_load, 1 + self.num_r + 1))  # sandra # Convert from Nx * 1  to Nz *  ( C + CP + q1 + q2)
        c = x_reshape[:, 0]  # mAb concentration in the mobile phase (mg/ml)
        cp = x_reshape[:, 1:self.num_r + 1]  # mAb concentration inside the particle (mg/ml)
        q1 = x_reshape[:, self.num_r + 1:self.num_r + 1 + 1]  # adsorbed mAb concentration (mg/ml)
        # q2 = x_reshape[:, self.num_r + 2:1 + self.num_r + 2] #sandra

        F = u[0]  # Inlet flow rate/harvest flow rate (ml/min)
        cF = c_[-1]  # u[1]  # mAb concentration in mobile phase/harvest mAb concentration (mg/ml)

        # # Parameter values
        # qmax1 = 36.45  # column capacity (mg/ml) #sandra
        k1 = 0.704  # kinetic constant (ml/(mg min))
        # qmax2 = 77.85  # column capacity (mg/ml) #sandra
        # k2 = 0.021  # kinetic constant (ml/(mg min)) #sandra
        K = 15.3  # Langmuir equilibrium constant (ml/mg)
        Deff = 7.6e-5  # effective pore diffusivity (cm2/min)

        rp = 4.25e-3  # particle radius (cm)
        L = self.len_cex  # * column length (cm)
        V = self.vol_cex  # * column volume (ml)
        column_r = np.sqrt(V / (np.pi * L))  # column radius
        v = F / (np.pi * column_r ** 2)  # * velocity (cm/min)
        # ** v：interstitial velocity (*double check how to calculate it*)

        Dax = 0.11 * v  # axial dispersion coefficient (cm2/min) #sandra
        kf = 0.067 * v ** 0.58  # mass transfer coefficient (cm/min)
        epsilon_c = 0.31  # extra-particle column void (-)
        epsilon_p = 0.94  # particle porosity (-)
        epsilon = 0.34  # sandra
        dz = L / (self.num_z_cap_load + 1)  # distance delta
        dr = rp / (self.num_r + 1)  # delta radius

        # # ODEs
        # # mAb concentration in mobile phase (c)
        dc = np.zeros(self.num_z_cap_load + 1)
        k = np.arange(1, self.num_z_cap_load)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc[k] = (c[k] - c[k - 1]) / dz
        # Top boundary condition: stands for difference between top and point [0]
        dc[0] = v / (epsilon_c * Dax) * (c[0] - cF)  # check C[0] or C[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc[-1] = 0

        # Second order discretization
        dc2 = np.zeros(self.num_z_cap_load)
        k = np.arange(0, self.num_z_cap_load)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2[k] = (dc[k + 1] - dc[k]) / dz
        # from point 0 to point 49 (total 50 points)
        dc_true = dc[1:self.num_z_cap_load + 1]
        # Cp value at r=rp point
        cp_rp = cp[:, -1]  # *

        # # adsorbed mAb concentration (mg/ml)
        dq1dt = k1 * (cp - q1 / K)  # q1: adsorption rate #sandra
        # dq1dt = k1 * ( cp_rp - q1 / K)  # q1: adsorption rate #sandra
        # calculate dcdt value
        dcdt = Dax * dc2 - (v / epsilon) * dc_true - (1 - epsilon_c) / epsilon * (dq1dt[:, -1])  # sandra
        dcdt = np.reshape(dcdt, [self.num_z_cap_load, 1])

        # # mAb concentration inside the particle
        dcp = np.zeros((self.num_z_cap_load, self.num_r + 1))
        k = np.arange(1, self.num_r)
        # forward difference: @ point 0 = c[1] -c[0],
        # dcp[1] stands for point 0 here.
        dcp[:, k] = (cp[:, k] - cp[:, k - 1]) / dr
        # Top boundary: dcp[0] is difference between r=0 and point 0
        dcp[:, 0] = 0
        # Bottom boundary: dcp[30] stands for the difference between point 29 and r=rp
        dcp[:, -1] = kf / Deff * (c[:] - cp_rp)

        # second order derivation
        dcp2 = np.zeros((self.num_z_cap_load, self.num_r))
        k = np.arange(0, self.num_r)
        r = np.zeros((1, self.num_r))
        r[:, k] = (k + 1) * dr  # particle radius

        dcp2[:, k] = ((((k + 2) * dr) ** 2) * dcp[:, k + 1] - (
                ((k + 1) * dr) ** 2 * dcp[:, k])) / dr
        cp_rp = np.reshape(cp_rp, [self.num_z_cap_load, 1])

        # # absorbed mAb concentration (mg/ml)
        # dq1dt = k1 * ((qmax1 - q1) * cp_rp - q1 / K)  # q1: quick adsorption rate #sandra
        # dq2dt = k2 * ((qmax2 - q2) * cp_rp - q2 / K)  # q2: slow adsorption rate  #sandra

        dcpdt = np.zeros((self.num_z_cap_load, self.num_r))
        k = np.arange(0, self.num_r)
        dcpdt[:, k] = Deff * (1 / (r[:, k] ** 2)) * dcp2[:, k] - (1 / epsilon_p) * (dq1dt[:])  # sandra
        dq1dt_new = np.reshape(dq1dt[:, -1],
                               [150, 1])  # dq1dt was reshaped from a 1d array(150, ) to a 2D array (150x1) #sandra
        dxdt = np.concatenate((dcdt, dcpdt, dq1dt_new), axis=1)
        dxdt = np.reshape(dxdt, (
                    self.num_x_cap_load - 150))  # size of dxdt changed from 150x13 to 150x12 because q2 was removed  #sandra

        dxdt_ = np.concatenate((dcdt_, dxdt))
        return dxdt_

    def cex_hl(self, t, x, u):
        x1 = x[0:self.num_z_cap_elu*3]
        x2 = x[self.num_z_cap_elu*3:]
        # # CEX elution:
        x_reshape = np.reshape(x1, (self.num_z_cex_elu, 1 + 1 + 1))
        c = x_reshape[:, 0]
        q = x_reshape[:, 1]
        cs = x_reshape[:, 2]

        F = u[0]
        cF = c[-1]
        cS = 0.9

        # # Parameters
        qmax = 150.2
        k_cex = 0.99
        H0_elu = 6.9e-4
        beta_elu = 8.5
        L = self.len_cex  # column length [cm]
        V = self.vol_cex  # column volume [ml]
        column_r = np.sqrt(V / (np.pi * L))  # column radius
        v = F / (np.pi * column_r ** 2)  # velocity
        # ** v：interstitial velocity (*check how to calculate it*)

        Dax = 0.11 * v  # axial dispersion coefficient (cm2/min)
        epsilon_c = 0.34  # extra-particle column void (-)
        epsilon = 0.34  #sandra

        dz = L / (self.num_z_cex_elu + 1)  # distance delta

        # # Mab concentration in mobile phase (c):
        dc = np.zeros(self.num_z_cex_elu + 1)
        k = np.arange(1, self.num_z_cex_elu)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc[k] = (c[k] - c[k - 1]) / dz
        # Top boundary condition: stands for difference between top and point [0]
        dc[0] = v / (epsilon_c * Dax) * (c[0] - 0)
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc[-1] = v / (epsilon_c * Dax) * (cF - c[-1])  # check C[0] or C[-1]

        # Second order discretization
        dc2 = np.zeros(self.num_z_cex_elu)
        k = np.arange(0, self.num_z_cex_elu)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2[k] = (dc[k + 1] - dc[k]) / dz
        # from point 0 to point 49 (total 50 points)
        dc_true = dc[1:self.num_z_cex_elu + 1]

        # # Modifier concentration (M)
        # first order discretization
        dcs = np.zeros(self.num_z_cex_elu + 1)
        k = np.arange(1, self.num_z_cex_elu)
        # point 0: dcs[0] = cs[1] - cs[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dcs[k] = (cs[k] - cs[k - 1]) / dz
        # Top boundary condition: stands for difference between top and point [0]
        dcs[0] = v / (epsilon_c * Dax) * (cs[0] - cS)  # check C[0] or C[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dcs[-1] = v / (epsilon_c * Dax) * (cS - cs[-1])  # check C[0] or C[-1]

        # Second order discretization
        dcs2 = np.zeros(self.num_z_cex_elu)
        k = np.arange(0, self.num_z_cex_elu)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dcs2[k] = (dcs[k + 1] - dcs[k]) / dz
        # from point 0 to point 49 (total 50 points)
        dcs_true_cex = dcs[1:self.num_z_cex_elu + 1]

        dcsdt = Dax * dcs2 - v / epsilon * dcs_true_cex
        dqdt = k_cex * (H0_elu * (cs ** (-beta_elu)) * (1 - (q / qmax)) * c - q)
        # calculate dcdt value
        dcdt = Dax * dc2 - v / epsilon * dc_true - (1 - epsilon_c) / epsilon * dqdt  # *

        dcsdt = np.reshape(dcsdt, [self.num_z_cex_elu, 1])
        dqdt = np.reshape(dqdt, [self.num_z_cex_elu, 1])
        dcdt = np.reshape(dcdt, [self.num_z_cex_elu, 1])

        dxdt = np.concatenate((dcdt, dqdt, dcsdt), axis=1)
        dxdt = np.reshape(dxdt, (self.num_x_cex_elu))

        # # Hold-up Loops: ---------------------------------------------------------------------------------------------
        x_reshape_ = np.reshape(x2, (self.num_z_loop, 1))
        c_ = x_reshape_[:, 0]

        F_ = u[0]
        cF_ = c[-1]  # u[1]

        # # Parameters
        L_ = self.len_loop
        V_ = self.vol_loop
        column_r_ = np.sqrt(V_ / (np.pi * L_))  # column radius
        v_ = F_ / (np.pi * column_r_ ** 2)  # velocity
        Dax_ = 290 * v_

        dz_ = L_ / (self.num_z_loop + 1)  # distance delta

        # # Mab concentration in mobile phase (c):
        dc_ = np.zeros(self.num_z_loop + 1)
        k_ = np.arange(1, self.num_z_loop)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc_[k_] = (c_[k_] - c_[k_ - 1]) / dz_
        # Top boundary condition: stands for difference between top and point [0]
        dc_[0] = v_ / (Dax_) * (c_[0] - cF_)  # check C[0] or C[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc_[-1] = 0  # v_ / (Dax_) * (0 - c_[-1])

        # Second order discretization
        dc2_ = np.zeros(self.num_z_loop)
        k_ = np.arange(0, self.num_z_loop)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2_[k_] = (dc_[k_ + 1] - dc_[k_]) / dz_
        # from point 0 to point 49 (total 50 points)
        dc_true_ = dc_[1:self.num_z_loop + 1]

        # calculate dcdt value
        dcdt_ = Dax_ * dc2_ - v_ * dc_true_

        dxdt_ = np.concatenate((dxdt, dcdt_))

        return dxdt_

    def hl_aex(self, t, x, u):
        x1 = x[0:self.num_z_loop]
        x2 = x[self.num_z_loop:]
        # # Hold-up Loops: ---------------------------------------------------------------------------------------------
        x_reshape_ = np.reshape(x1, (self.num_z_loop, 1))
        c_ = x_reshape_[:, 0]

        F_ = u[0]
        cF_ = c_[-1]  # u[1]

        # # Parameters
        L_ = self.len_loop
        V_ = self.vol_loop
        column_r_ = np.sqrt(V_ / (np.pi * L_))  # column radius
        v_ = F_ / (np.pi * column_r_ ** 2)  # velocity
        Dax_ = 290 * v_

        dz_ = L_ / (self.num_z_loop + 1)  # distance delta

        # # Mab concentration in mobile phase (c):
        dc_ = np.zeros(self.num_z_loop + 1)
        k_ = np.arange(1, self.num_z_loop)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc_[k_] = (c_[k_] - c_[k_ - 1]) / dz_
        # Top boundary condition: stands for difference between top and point [0]
        dc_[0] = v_ / (Dax_) * (c_[0] - 0)  # check C[0] or C[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc_[-1] = v_ / (Dax_) * (cF_ - c_[-1])

        # Second order discretization
        dc2_ = np.zeros(self.num_z_loop)
        k_ = np.arange(0, self.num_z_loop)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2_[k_] = (dc_[k_ + 1] - dc_[k_]) / dz_
        # from point 0 to point 49 (total 50 points)
        dc_true_ = dc_[1:self.num_z_loop + 1]

        # calculate dcdt value
        dcdt_ = Dax_ * dc2_ - v_ * dc_true_

        # # AEX ---------------------------------------------------------------------------------------------
        x_reshape = np.reshape(x2, (self.num_z_aex, 1))
        c = x_reshape[:, 0]

        F = u[0]
        cF = c_[-1]

        # # Parameters
        L = self.len_aex  # column length [cm]
        V = self.vol_aex  # column volume [ml]
        column_r = np.sqrt(V / (np.pi * L))  # column radius
        v = F / (np.pi * column_r ** 2)  # velocity
        # ** v：interstitial velocity (*check how to calculate it*)

        Dax = 0.16 * v  # axial dispersion coefficient (cm2/min)
        epsilon_c = 0.34  # extra-particle column void (-)
        epsilon = 0.32

        # Flow-through AEX column (product did not adsorb)
        dz = L / (self.num_z_aex + 1)  # distance delta

        # # Mab concentration in mobile phase (c):
        dc = np.zeros(self.num_z_aex + 1)
        k = np.arange(1, self.num_z_aex)
        # point 0: dc[0] = c[1] - c[0]; point [0] means dc[1] here because dc[0] is the top boundary
        dc[k] = (c[k] - c[k - 1]) / dz
        # Top boundary condition: stands for difference between top and point [0]
        dc[0] = v / (epsilon_c * Dax) * (c[0] - cF)  # check C[0] or C[-1]
        # Bottom boundary condition: stands for difference between point [49] and bottom
        dc[-1] = v / (epsilon_c * Dax) * (c[-1] - c[-1])

        # Second order discretization
        dc2 = np.zeros(self.num_z_aex)
        k = np.arange(0, self.num_z_aex)
        # central difference: dc2[k] = (c[k + 1] - 2 * c[k] + c[k - 1]) / (dz ** 2)
        dc2[k] = (dc[k + 1] - dc[k]) / dz
        # from point 0 to point 49 (total 50 points)
        dc_true_aex = dc[1:self.num_z_aex + 1]

        # calculate dcdt value
        dcdt = Dax * dc2 - v / epsilon * dc_true_aex

        dxdt = np.concatenate((dcdt_, dcdt))
        return dxdt

    def init_simulator(self):
        self.num_x = self.x0.size
        self.num_u = self.u.size

        self.x_all = np.zeros((self.num_sim + 1, self.num_x))
        self.u_all = np.zeros((self.num_sim, self.num_u))
        self.x_all[0, :] = tuple(self.x0)  # Record the initial state
        self.u_all[:, :] = tuple(self.u)  # NI (Need improvement: check if u is uniform or dynamic over time)

    def run(self, model):
        for i in range(1, self.num_sim + 1):
            sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, u=self.u_all[i - 1]),
                                      t_span=[0, self.delta_t], y0=tuple(self.x_all[i - 1, :]))
            xk = sol['y'][:, -1]
            self.x_all[i, :] = xk
        return self.x_all

    def run_cap(self, model):
        for i in range(1, self.num_sim + 1):
            sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, u=self.u_all[i - 1]),
                                      t_span=[0, self.delta_t], y0=tuple(self.x_all[i - 1, :]))
            xk = sol['y'][:, -1]
            self.x_all[i, :] = xk

            switchConfig = self.controller2(np.concatenate((xk,self.u_all[i-1],np.array([i*self.delta_t]))))
            print(i)
            if switchConfig == True:
                self.x_all = self.x_all[:i+1,:]  # Truncate the redundant rows
                return self.x_all

    def controller(self, time):
        if time >= self.period:
            switchConfig = True
        else:
            switchConfig = False
        return switchConfig

    def controller2(self, x):
        capacity = 2708225  # [mg] Here we use the saturation value. In reality, we should use a capacity way less than this value.
        currentCapacity = (20000*self.delta_t+x[-1])*x[-2]/2*x[-3]
        if currentCapacity >= capacity:
            switchConfig = True
        else:
            switchConfig = False
        return switchConfig

    def compute_reward(self, previous_observation, action, current_observation, criteria='compute_recover_rate'):
        r = self.compute_recover_rate(current_observation)  # input shape (2+1950,)
        return r

    def compute_recover_rate(self,x):  # Recovery yield = purified product/harvested mAb
        harvested = x[-3]*x[-2]*x[-1]
        bottom = np.max([20000*self.delta_t, x[-1]])
        purified = (20000*self.delta_t+bottom)*x[-2]/2*x[-3]
        return purified/harvested

    def recovered_mAb(self, x, u, utils):
        x_downstream = x[19:]/1000  # Select states of downstream (mg/ml)
        x0_cap_elu = utils.load_to_elu(x_downstream)  # Convert final state of loading mode to the initial state of elution mode
        F = u[7]*1000  # Feed flow rate (ml/min)
        cF = x[18]/1000  # Concentration of mAb in feed flow (mg/ml)
        X_d = []  # Initialize an empty list

        # Convert the units
        # Capture column - elution mode + Holdup loop - simulator -----------------------------------------------
        self.x0 = np.concatenate((x0_cap_elu, np.zeros(self.num_z_loop)))  # The initial state is the final state of caputre column at previous configuration
        self.u = np.array([F/16.7*60, cF])  # From the reference (ml/min) (mg/ml)
        self.num_sim = int(6000 * 0.104)
        self.init_simulator()
        X_cap_vi = self.run(self.cap_vi)
        X_d.append(X_cap_vi)
        x0_vi = X_cap_vi[-1, -self.num_z_loop:]
        # utils.visualize_results_down(X_cap_vi, 'cap_vi')
        Ad_vi = self.cal2(X_cap_vi[:, -1], self.u[0], X_cap_vi[:, 447])  # downstream.u is a 1x2 array.
        print("Adsorbed mAb in VI loop is " + str(Ad_vi) + ' mg')

        # VI + CEX load mode - simulator -----------------------------------------------
        self.x0 = np.concatenate((x0_vi, np.zeros(self.num_z_cex_load * 12)))
        self.u = np.array([F/16.7*11.1, cF])
        self.num_sim = int(6000 * 3.6)
        self.init_simulator()
        X_vi_cex = self.run(self.vi_cex)
        X_d.append(X_vi_cex)
        x0_cex_elu = utils.load_to_elu_cex(x_load=X_vi_cex[-1,
                                                 self.num_z_loop:])  # Convert final state of loading column to initial state of elution column
        # utils.visualize_results_down(X_vi_cex, 'vi_cex')
        Ad_cex = self.cal2(X_vi_cex[:, -12], self.u[0], X_vi_cex[:, 149])
        print("Adsorbed mAb in CEX is " + str(Ad_cex) + ' mg')

        # CEX elution mode + Holdup loop - simulator -----------------------------------------------
        self.x0 = np.concatenate((x0_cex_elu, np.zeros(self.num_z_loop)))
        self.u = np.array([F/16.7*5.6, cF])
        self.num_sim = int(6000 * 0.333)
        self.init_simulator()
        X_cex_hl = self.run(self.cex_hl)
        X_d.append(X_cex_hl)
        x0_loop = X_cex_hl[-1, -self.num_z_loop:]
        # utils.visualize_results_down(X_cex_hl, 'cex_hl')
        Ad_holduploop = self.cal2(X_cex_hl[:, -1], self.u[0], X_cex_hl[:, 447])
        print("Adsorbed mAb in  Holdup Loop is " + str(Ad_holduploop) + ' mg')

        # Holdup loop + AEX - simulator -----------------------------------------------
        self.x0 = np.concatenate((x0_loop, np.zeros(self.num_z_aex)))
        self.u = np.array([F/16.7*22.2, cF])
        self.num_sim = int(6000 * 3)
        self.init_simulator()
        X_hl_aex = self.run(self.hl_aex)
        X_d.append(X_hl_aex)
        # utils.visualize_results_down(X_hl_aex, 'hl_aex')
        Ad_aex = self.cal3(X_hl_aex[:, -1], self.u[0])
        print("Recovered mAb in Aex is " + str(Ad_aex) + ' mg')

    def cal(self, c_btcurve, F, cF):
        c_bt = np.zeros(c_btcurve.shape[0])
        k = np.arange(1, c_bt.shape[0] - 1)
        c_bt[k] = c_btcurve[k] * self.delta_t
        # Top Boundary
        c_bt[0] = c_btcurve[0] * self.delta_t / 2
        # Bottom boundary
        c_bt[-1] = c_btcurve[-1] * self.delta_t / 2
        # Adding the areas
        Total_area_below_btcurve = np.sum(c_bt)
        adsorbed_mAb = F * cF * self.num_sim * self.delta_t - F * np.sum(c_bt)

        return adsorbed_mAb

    def cal2(self, c_btcurve, F, cF):
        # calculating the area for outlet
        c_bt = np.zeros(c_btcurve.shape[0])
        k = np.arange(1, c_bt.shape[0] - 1)
        # Top Boundary
        c_bt[0] = c_btcurve[0] * self.delta_t / 2
        # Bottom boundary
        c_bt[-1] = c_btcurve[-1] * self.delta_t / 2
        # Adding the areas
        Total_area_below_btcurve = np.sum(c_bt)

        # calculating the area for inlet
        c_bt2 = np.zeros(cF.shape[0])
        k2 = np.arange(1, c_bt2.shape[0] - 1)
        c_bt2[k2] = cF[k2] * self.delta_t
        # Top Boundary
        c_bt2[0] = cF[0] * self.delta_t / 2
        # Bottom boundary
        c_bt2[-1] = cF[-1] * self.delta_t / 2
        # Adding the areas
        Total_area_below_cF = np.sum(c_bt2)

        adsorbed_mAb = F * np.sum(c_bt2) - F * np.sum(c_bt)

        return adsorbed_mAb

    def cal3(self, c_btcurve, F):
        c_bt = np.zeros(c_btcurve.shape[0])
        k = np.arange(1, c_bt.shape[0] - 1)
        c_bt[k] = c_btcurve[k] * self.delta_t
        # Top Boundary
        c_bt[0] = c_btcurve[0] * self.delta_t / 2
        # Bottom boundary
        c_bt[-1] = c_btcurve[-1] * self.delta_t / 2
        # Adding the areas
        Total_area_below_btcurve = np.sum(c_bt)
        recovered_mAb = F * np.sum(c_bt)

        return recovered_mAb


xscale = np.array([3e10, 5e10, 5e3, 50, 200, 50, 150, 3000, 2e10, 3e10,5000, 100,
                   250, 100, 200, 3000, 37.0])
uscale = np.array([3000, 3000, 3000, 3000, 2000, 2000, 50])

class UtilsHelper:
    def save_results(self, t, X, U, res_dir='results'):
        np.save(os.path.join(res_dir, 't'), t)
        np.save(os.path.join(res_dir, 'X'), X)
        np.save(os.path.join(res_dir, 'U'), U)

    def save_ss(self, xss, uss, res_dir='data'):
        np.save(os.path.join(res_dir, 'xss'), xss)
        np.save(os.path.join(res_dir, 'uss'), uss)

    def load_ss(self, res_dir='data'):
        xss = np.load(os.path.join(res_dir, 'xss.npy'))
        uss = np.load(os.path.join(res_dir, 'uss.npy'))
        return (xss, uss)

    def process_results(self, X, U):
        X = np.array(X)
        U = np.array(U)
        return X, U

    def compare_controllers(self, t, Xm, Um, Xe, Ue, Xs, Us):

        # create figure (fig), and array of axes (ax)
        plt.close("all")

        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, Xm[:, 0])
        plt.plot(t, Xe[:, 0])
        plt.plot(t, Xs[:, 0], linestyle="--")
        plt.ylabel('Xv_1 (cell/L)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 8])
        plt.plot(t, Xe[:, 8])
        plt.plot(t, Xs[:, 8], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('Xv_2 (cell/L)')
        plt.tight_layout()
        plt.show()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(t, Xm[:, 1])
        plt.plot(t, Xe[:, 1])
        plt.plot(t, Xs[:, 1], linestyle="--")
        plt.ylabel('Xt_1 (cell/L)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 9])
        plt.plot(t, Xe[:, 9])
        plt.plot(t, Xs[:, 9], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('Xt_2 (cell/L)')
        plt.tight_layout()
        plt.show()

        plt.figure(3)
        plt.subplot(211)
        plt.plot(t, Xm[:, 2])
        plt.plot(t, Xe[:, 2])
        plt.plot(t, Xs[:, 2], linestyle="--")
        plt.ylabel('GLC_1 (mM)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 10])
        plt.plot(t, Xe[:, 10])
        plt.plot(t, Xs[:, 10], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('GLC_2 (mM)')
        plt.tight_layout()
        plt.show()

        plt.figure(4)
        plt.subplot(211)
        plt.plot(t, Xm[:, 3])
        plt.plot(t, Xe[:, 3])
        plt.plot(t, Xs[:, 3], linestyle="--")
        plt.ylabel('GLN_1 (mM)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 11])
        plt.plot(t, Xe[:, 11])
        plt.plot(t, Xs[:, 11], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('GLN_2 (mM)')
        plt.tight_layout()
        plt.show()

        plt.figure(5)
        plt.subplot(211)
        plt.plot(t, Xm[:, 4])
        plt.plot(t, Xe[:, 4])
        plt.plot(t, Xs[:, 4], linestyle="--")
        plt.ylabel('LAC_1 (mM)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 12])
        plt.plot(t, Xe[:, 12])
        plt.plot(t, Xs[:, 12], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('LAC_2 (mM)')
        plt.tight_layout()
        plt.show()

        plt.figure(6)
        plt.subplot(211)
        plt.plot(t, Xm[:, 5])
        plt.plot(t, Xe[:, 5])
        plt.plot(t, Xs[:, 5], linestyle="--")
        plt.xlabel('')
        plt.ylabel('AMM_1 (mM)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 13])
        plt.plot(t, Xe[:, 13])
        plt.plot(t, Xs[:, 13], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('AMM_2 (mM)')
        plt.tight_layout()
        plt.show()

        plt.figure(7)
        plt.subplot(211)
        plt.plot(t, Xm[:, 6])
        plt.plot(t, Xe[:, 6])
        plt.plot(t, Xs[:, 6], linestyle="--")
        plt.xlabel('')
        plt.ylabel('mAb_1 (mg/L)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 14])
        plt.plot(t, Xe[:, 14])
        plt.plot(t, Xs[:, 14], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('mAb_2 (mg/L)')
        plt.tight_layout()
        plt.show()

        plt.figure(8)
        plt.subplot(211)
        plt.plot(t, Xm[:, 7])
        plt.plot(t, Xe[:, 7])
        plt.plot(t, Xs[:, 7], linestyle="--")
        plt.xlabel('')
        plt.ylabel('V_1 (L)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(t, Xm[:, 15])
        plt.plot(t, Xe[:, 15])
        plt.plot(t, Xs[:, 15], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('V_2 (L)')
        plt.tight_layout()
        plt.show()

        plt.figure(9)
        plt.subplot(211)
        plt.plot(t, Xm[:, 16])
        plt.plot(t, Xe[:, 16])
        plt.plot(t, Xs[:, 16], linestyle="--")
        plt.ylabel('T ($^\circ$C)')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        t = t[:-1]
        plt.subplot(212)
        plt.step(t, Um[:, 6], where='post')
        plt.step(t, Ue[:, 6], where='post')
        plt.plot(t, Us[:, 6], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('Tc ($^\circ$C)')
        plt.tight_layout()
        plt.show()

        plt.figure(10)
        plt.subplot(211)
        plt.step(t, Um[:, 0], where='post')
        plt.step(t, Ue[:, 0], where='post')
        plt.plot(t, Us[:, 0], linestyle="--")
        plt.ylabel('F_in')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.step(t, Um[:, 1], where='post')
        plt.step(t, Ue[:, 1], where='post')
        plt.plot(t, Us[:, 1], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('F_1')
        plt.tight_layout()
        plt.show()

        plt.figure(11)
        plt.subplot(211)
        plt.step(t, Um[:, 2], where='post')
        plt.step(t, Ue[:, 2], where='post')
        plt.plot(t, Us[:, 2], linestyle="--")
        plt.ylabel('F_r')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.step(t, Um[:, 3], where='post')
        plt.step(t, Ue[:, 3], where='post')
        plt.plot(t, Us[:, 3], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('F_2')
        plt.tight_layout()
        plt.show()

        plt.figure(12)
        plt.subplot(211)
        plt.step(t, Um[:, 4], where='post')
        plt.step(t, Ue[:, 4], where='post')
        plt.plot(t, Us[:, 4], linestyle="--")
        plt.ylabel('GLC_in')
        plt.legend(['MPC', 'EMPC', 'SS'])
        plt.tight_layout()

        plt.subplot(212)
        plt.step(t, Um[:, 5], where='post')
        plt.step(t, Ue[:, 5], where='post')
        plt.plot(t, Us[:, 5], linestyle="--")
        plt.xlabel('Time (h)')
        plt.ylabel('GLN_in')
        plt.tight_layout()
        plt.show()

    def load_to_elu(self, x_load):
        # For every 13 states (1 for c, 10 for cp, 2 for q)
        # pick c as 1st state in new state vector.
        # add 2 q up, then assign the summation as 2nd state in the new vector
        #* for now, initialize the 3rd state, cs, as 0.9
        num_x_layer_load = 13  # number of states per layer in load mode
        num_x_layer_elu = 3  # number of states per layer in elution mode
        num_z = int(x_load.size/num_x_layer_load)  # number of layers in load mode
        x_elu = np.zeros((num_z*3,))  # number of states in elution mode
        for i in range(num_z):
            c = x_load[i*num_x_layer_load+0]  # pick first state
            q = x_load[i*num_x_layer_load+11] + x_load[i*num_x_layer_load+12]  # add two q
            cs = 0.9  # assign 0.9 to cs
            x_elu[i*num_x_layer_elu+0] = c
            x_elu[i*num_x_layer_elu+1] = q
            x_elu[i*num_x_layer_elu+2] = cs
        return x_elu

    def load_to_elu_cex(self, x_load):  #sandra
        # For every 12 states (1 for c, 10 for cp, 1 for q) #sandra
        # pick c as 1st state in new state vector.
        # pick q as 2nd state in the new vector
        #* for now, initialize the 3rd state, cs, as 0.9
        num_x_layer_load = 12  # number of states per layer in load mode # sandra #changed to 12 for cex column because q2 is removed
        num_x_layer_elu = 3  # number of states per layer in elution mode
        num_z = int(x_load.size/num_x_layer_load)  # number of layers in load mode
        x_elu = np.zeros((num_z*3,))  # number of states in elution mode
        for i in range(num_z):
            c = x_load[i*num_x_layer_load+0]  # pick first state
            q = x_load[i*num_x_layer_load+11]
            cs = 0.9 # assign 0 to cs
            x_elu[i*num_x_layer_elu+0] = c
            x_elu[i*num_x_layer_elu+1] = q
            x_elu[i*num_x_layer_elu+2] = cs
        return x_elu

    def visualize_results_up(self, t, X, U):
        plt.close("all")

        # States
        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, X[:, 0])
        plt.ylabel('$X_{v, reactor}$ (cell/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 8])
        plt.xlabel('Time (h)')
        plt.ylabel('$X_{v, separator}$ (cell/L)')
        plt.show()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(t, X[:, 1])
        plt.ylabel('$X_{t, reactor}$ (cell/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 9])
        plt.xlabel('Time (h)')
        plt.ylabel('$X_{t, separator}$ (cell/L)')
        plt.show()

        plt.figure(3)
        plt.subplot(211)
        plt.plot(t, X[:, 2])
        plt.ylabel('$[GLC]_{reactor}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 10])
        plt.xlabel('Time (h)')
        plt.ylabel('$[GLC]_{separator}}$ (mM)')
        plt.show()

        plt.figure(4)
        plt.subplot(211)
        plt.plot(t, X[:, 3])
        plt.ylabel('$[GLN]_{reactor}}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 11])
        plt.xlabel('Time (h)')
        plt.ylabel('$[GLN]_{separator}$ (mM)')
        plt.show()

        plt.figure(5)
        plt.subplot(211)
        plt.plot(t, X[:, 4])
        plt.ylabel('$[LAC]_{reactor}}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 12])
        plt.xlabel('Time (h)')
        plt.ylabel('$[LAC]_{separator}$ (mM)')
        plt.show()

        plt.figure(6)
        plt.subplot(211)
        plt.plot(t, X[:, 5])
        plt.ylabel('$[AMM]_{reactor}}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 13])
        plt.xlabel('Time (h)')
        plt.ylabel('$[AMM]_{separator}$ (mM)')
        plt.show()

        plt.figure(7)
        plt.subplot(211)
        plt.plot(t, X[:, 6])
        plt.ylabel('$[mAb]_{reactor}}$ (mg/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 14])
        plt.xlabel('Time (h)')
        plt.ylabel('$[mAb]_{separator}$ (mg/L)')
        plt.savefig("results/upstream/up_state.pdf")
        plt.show()

        plt.figure(8)
        plt.subplot(211)
        plt.plot(t, X[:, 7])
        plt.ylabel('$V_{reactor}}$ (L)')
        plt.subplot(212)
        plt.plot(t, X[:, 15])
        plt.xlabel('Time (h)')
        plt.ylabel('$V_{separator}$ (L)')
        plt.show()

        plt.figure(9)
        plt.plot(t, X[:, 16])
        plt.xlabel('Time (h)')
        plt.ylabel('T (K)')

        # Inputs
        t = t[:-1]
        plt.figure(10)
        plt.subplot(211)
        plt.step(t, U[:, 0], where='post')
        plt.ylabel('$F_{in}$')
        plt.subplot(212)
        plt.step(t, U[:, 1], where='post')
        plt.xlabel('Time (h)')
        plt.ylabel('$F_1$')

        plt.figure(11)
        plt.subplot(211)
        plt.step(t, U[:, 2], where='post')
        plt.ylabel('$F_r$')
        plt.subplot(212)
        plt.step(t, U[:, 3], where='post')
        plt.xlabel('Time (h)')
        plt.ylabel('$F_2$')

        plt.figure(12)
        plt.subplot(211)
        plt.step(t, U[:, 4], where='post')
        plt.ylabel('$[GLC]_{in}$')
        plt.subplot(212)
        plt.step(t, U[:, 5], where='post')
        plt.xlabel('Time (h)')
        plt.ylabel('$GLN_{in}$')

        plt.figure(13)
        plt.step(t, U[:, 6], where='post')
        plt.xlabel('Time (h)')
        plt.ylabel('$T_c$ (K)')

    def visualize_results_buffer(self, t, X, U):
        plt.close("all")

        # States
        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, X[:, 0])
        plt.ylabel('$h$ (dm)')
        plt.subplot(212)
        plt.plot(t, X[:, 1])
        plt.xlabel('Time (h)')
        plt.ylabel('$[mAb]$ (mg/L)')
        plt.show()

        # Inputs
        t = t[:-1]
        plt.figure(2)
        plt.subplot(211)
        plt.step(t, U[:, 0], where='post')
        plt.ylabel('$F_{out}$ (L/h)')
        plt.subplot(212)
        plt.step(t, U[:, 1], where='post')
        plt.xlabel('Time (h)')
        plt.ylabel('$F_{in}$ (L/h)')
        plt.show()

        plt.figure(3)
        plt.step(t, U[:, 2], where='post')
        plt.xlabel('Time (h)')
        plt.ylabel('$[mAb]_{in}$ (mg/L)')
        plt.show()

    def visualize_results_down(self, X, key):
        if key == 'cap_load':
            # Figure 1 Capture column loading ------------------------------
            plt.figure()
            plt.subplot(231)
            plt.plot(X[:, 13 * 1 + 1])
            plt.plot(X[:, 13 * 25 + 1])
            plt.plot(X[:, 13 * 49 + 1])
            # plt.ylim((0.88, 0.91))
            plt.ylabel("cp [mg/ml] r=1")
            plt.xlabel("t [min]")
            plt.subplot(232)
            plt.plot(X[:, 13 * 1 + 5])
            plt.plot(X[:, 13 * 25 + 5])
            plt.plot(X[:, 13 * 49 + 5])
            # plt.ylim((0.88, 0.91))
            plt.ylabel("cp [mg/ml] r=5")
            plt.xlabel("t [min]")
            plt.subplot(233)
            plt.plot(X[:, 13 * 1 + 10])
            plt.plot(X[:, 13 * 25 + 10])
            plt.plot(X[:, 13 * 49 + 10])
            # plt.ylim((0.88, 0.91))
            plt.ylabel("cp [mg/ml] r=10")
            plt.xlabel("t [min]")
            plt.subplot(234)
            plt.plot(X[:, 13 * 1 + 0])
            plt.plot(X[:, 13 * 25 + 0])
            plt.plot(X[:, 13 * 49 + 0])
            # plt.ylim((0.88, 0.91))
            plt.ylabel("c [mg/ml]")
            plt.xlabel("t [min]")
            plt.subplot(235)
            plt.plot(X[:, 13 * 1 + 11])
            plt.plot(X[:, 13 * 25 + 11])
            plt.plot(X[:, 13 * 49 + 11])
            # plt.ylim((32, 35))
            plt.ylabel("q1 [mg/ml]")
            plt.xlabel("t [min]")
            plt.subplot(236)
            plt.plot(X[:, 13 * 1 + 12])
            plt.plot(X[:, 13 * 25 + 12])
            plt.plot(X[:, 13 * 49 + 12])
            # plt.ylim((71, 74))
            plt.ylabel("q2 [mg/ml]")
            plt.xlabel("t [min]")
            plt.legend(['level 1', 'level 25', 'level 49'])
            plt.tight_layout()
            plt.savefig("results/downstream/x1_capture_level1_25_49.png")
            plt.show()
        elif key == 'cap_vi':
            # Figure 2 Capture + VI -----------------------------------------
            plt.figure()
            plt.subplot(411)
            plt.plot(X[:, 3 * 1 + 0])
            plt.plot(X[:, 3 * 17 + 0])
            plt.plot(X[:, 3 * 34 + 0])
            plt.plot(X[:, 3 * 49 + 0])
            plt.xlabel("t [min]")
            plt.ylabel("c_cc [mg/ml]")
            plt.tight_layout()
            plt.subplot(412)
            plt.plot(X[:, 3 * 150 + 1])
            plt.plot(X[:, 3 * 150 + 17])
            plt.plot(X[:, 3 * 150 + 34])
            plt.plot(X[:, 3 * 150 + 49])
            plt.xlabel("t [min]")
            plt.ylabel("c_vi [mg/ml]")
            plt.tight_layout()
            plt.subplot(413)
            plt.plot(X[:, 3 * 1 + 1])
            plt.plot(X[:, 3 * 17 + 1])
            plt.plot(X[:, 3 * 34 + 1])
            plt.plot(X[:, 3 * 49 + 1])
            plt.xlabel("t [min]")
            plt.ylabel("q [mg/ml]")
            plt.tight_layout()
            plt.subplot(414)
            plt.plot(X[:, 3 * 1 + 2])
            plt.plot(X[:, 3 * 17 + 2])
            plt.plot(X[:, 3 * 34 + 2])
            plt.plot(X[:, 3 * 49 + 2])
            plt.xlabel("t [min]")
            plt.ylabel("cs [mg/ml]")
            plt.legend(['z=1', '17', '34', '49'])
            plt.tight_layout()
            plt.savefig("results/downstream/x2_elution_vi")
            plt.show()
        if key == 'vi_cex':
            # Figure 3 VI + CEX -----------------------------------------
            plt.figure()
            plt.plot(X[:, 1])
            plt.plot(X[:, 17])
            plt.plot(X[:, 34])
            plt.plot(X[:, 49])
            plt.xlabel("t [min]")
            plt.ylabel("c_vi [mg/ml]")
            plt.tight_layout()
            plt.savefig("results/downstream/x3_vi_cex_1")
            plt.show()
    
            plt.figure()
            plt.subplot(231)
            plt.plot(X[:, 150 + 13 * 1 + 1])
            plt.plot(X[:, 150 + 13 * 25 + 1])
            plt.plot(X[:, 150 + 13 * 49 + 1])
            # plt.ylim((0.88, 0.91))
            plt.ylabel("cp [mg/ml] r=1")
            plt.xlabel("t [min]")
            plt.subplot(232)
            plt.plot(X[:, 150 + 13 * 1 + 5])
            plt.plot(X[:, 150 + 13 * 25 + 5])
            plt.plot(X[:, 150 + 13 * 49 + 5])
            # plt.ylim((0.88, 0.91))
            plt.ylabel("cp [mg/ml] r=5")
            plt.xlabel("t [min]")
            plt.subplot(233)
            plt.plot(X[:, 150 + 13 * 1 + 10])
            plt.plot(X[:, 150 + 13 * 25 + 10])
            plt.plot(X[:, 150 + 13 * 49 + 10])
            # plt.ylim((0.88, 0.91))
            plt.ylabel("cp [mg/ml] r=10")
            plt.xlabel("t [min]")
            plt.subplot(234)
            plt.plot(X[:, 150 + 13 * 1 + 0])
            plt.plot(X[:, 150 + 13 * 25 + 0])
            plt.plot(X[:, 150 + 13 * 49 + 0])
            # plt.ylim((0.88, 0.91))
            plt.ylabel("c [mg/ml]")
            plt.xlabel("t [min]")
            plt.subplot(235)
            plt.plot(X[:, 150 + 13 * 1 + 11])
            plt.plot(X[:, 150 + 13 * 25 + 11])
            plt.plot(X[:, 150 + 13 * 49 + 11])
            # plt.ylim((32, 35))
            plt.ylabel("q1 [mg/ml]")
            plt.xlabel("t [min]")
            plt.subplot(236)
            plt.plot(X[:, 150 + 13 * 1 + 12])
            plt.plot(X[:, 150 + 13 * 25 + 12])
            plt.plot(X[:, 150 + 13 * 49 + 12])
            # plt.ylim((71, 74))
            plt.ylabel("q2 [mg/ml]")
            plt.xlabel("t [min]")
            plt.legend(['level 1', 'level 25', 'level 49'])
            plt.tight_layout()
            plt.savefig("results/downstream/x3_vi_cex_2")
            plt.show()
        elif key == 'cex_hl':
            # Figure 4 CEX + HUL -----------------------------------------
            plt.figure()
            plt.subplot(411)
            plt.plot(X[:, 3 * 1 + 0])
            plt.plot(X[:, 3 * 17 + 0])
            plt.plot(X[:, 3 * 34 + 0])
            plt.plot(X[:, 3 * 49 + 0])
            plt.xlabel("t [min]")
            plt.ylabel("c_cc [mg/ml]")
            plt.tight_layout()
            plt.subplot(412)
            plt.plot(X[:, 3 * 150 + 1])
            plt.plot(X[:, 3 * 150 + 17])
            plt.plot(X[:, 3 * 150 + 34])
            plt.plot(X[:, 3 * 150 + 49])
            plt.xlabel("t [min]")
            plt.ylabel("c_vi [mg/ml]")
            plt.tight_layout()
            plt.subplot(413)
            plt.plot(X[:, 3 * 1 + 1])
            plt.plot(X[:, 3 * 17 + 1])
            plt.plot(X[:, 3 * 34 + 1])
            plt.plot(X[:, 3 * 49 + 1])
            plt.xlabel("t [min]")
            plt.ylabel("q [mg/ml]")
            plt.tight_layout()
            plt.subplot(414)
            plt.plot(X[:, 3 * 1 + 2])
            plt.plot(X[:, 3 * 17 + 2])
            plt.plot(X[:, 3 * 34 + 2])
            plt.plot(X[:, 3 * 49 + 2])
            plt.xlabel("t [min]")
            plt.ylabel("cs [mg/ml]")
            plt.legend(['z=1', '17', '34', '49'])
            plt.tight_layout()
            plt.savefig("results/downstream/x4_cex_hul")
            plt.show()
        elif key == 'hl_aex':
            # Figure 5 HUL + AEX -----------------------------------------
            plt.figure()
            plt.subplot(211)
            plt.plot(X[:, 1])
            plt.plot(X[:, 17])
            plt.plot(X[:, 34])
            plt.plot(X[:, 49])
            plt.xlabel("t [min]")
            plt.ylabel("c_hl [mg/ml]")
            plt.tight_layout()
            plt.subplot(212)
            plt.plot(X[:, 150 + 1])
            plt.plot(X[:, 150 + 17])
            plt.plot(X[:, 150 + 34])
            plt.plot(X[:, 150 + 49])
            plt.xlabel("t [min]")
            plt.ylabel("c_aex [mg/ml]")
            plt.legend(['z=1', '17', '34', '49'])
            plt.tight_layout()
            plt.savefig("results/downstream/x5_hul_aex")
            plt.show()

    def visualize_results_up_buffer(self, t, X, U):
        plt.close("all")

        # States
        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, X[:, 0])
        plt.ylabel('$X_{v1}$ (cell/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 8])
        plt.xlabel('Time (h)')
        plt.ylabel('$X_{v2}$ (cell/L)')
        plt.savefig("state1.pdf")
        plt.show()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(t, X[:, 1])
        plt.ylabel('$X_{t1}$ (cell/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 9])
        plt.xlabel('Time (h)')
        plt.ylabel('$X_{t2}$ (cell/L)')
        plt.savefig("state2.pdf")
        plt.show()

        plt.figure(3)
        plt.subplot(211)
        plt.plot(t, X[:, 2])
        plt.ylabel('$[GLC]_1$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 10])
        plt.xlabel('Time (h)')
        plt.ylabel('$[GLC]_2$ (mM)')
        plt.savefig("state3.pdf")
        plt.show()

        plt.figure(4)
        plt.subplot(211)
        plt.plot(t, X[:, 3])
        plt.ylabel('$[GLN]_1$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 11])
        plt.xlabel('Time (h)')
        plt.ylabel('$[GLN]_2$ (mM)')
        plt.savefig("state4.pdf")
        plt.show()

        plt.figure(5)
        plt.subplot(211)
        plt.plot(t, X[:, 4])
        plt.ylabel('$[LAC]_1$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 12])
        plt.xlabel('Time (h)')
        plt.ylabel('$[LAC]_2$ (mM)')
        plt.savefig("state5.pdf")
        plt.show()

        plt.figure(6)
        plt.subplot(211)
        plt.plot(t, X[:, 5])
        plt.ylabel('$[AMM]_1$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 13])
        plt.xlabel('Time (h)')
        plt.ylabel('$[AMM]_2$ (mM)')
        plt.savefig("state6.pdf")
        plt.show()

        plt.figure(7)
        plt.subplot(211)
        plt.plot(t, X[:, 6])
        plt.ylabel('$[mAb]_1$ (mg/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 14])
        plt.xlabel('Time (h)')
        plt.ylabel('$[mAb]_2$ (mg/L)')
        plt.savefig("state7.pdf")
        plt.show()

        plt.figure(8)
        plt.subplot(211)
        plt.plot(t, X[:, 7])
        plt.ylabel('$V_1$ (L)')
        plt.subplot(212)
        plt.plot(t, X[:, 15])
        plt.xlabel('Time (h)')
        plt.ylabel('$V_2$ (L)')
        plt.savefig("state8.pdf")
        plt.show()

        plt.figure(9)
        plt.plot(t, X[:, 16])
        plt.xlabel('Time (h)')
        plt.savefig("state9.pdf")
        plt.ylabel('T (K)')

        plt.figure(10)
        plt.subplot(211)
        plt.plot(t, X[:, 17])
        plt.ylabel('$h$ (dm)')
        plt.subplot(212)
        plt.plot(t, X[:, 18])
        plt.plot(t, X[:, 14])
        plt.xlabel('Time (h)')
        plt.ylabel('$[mAb]$ (mg/L)')
        plt.legend(('Buffer tank', 'Inlet of buffter tank'))
        plt.savefig("state10.pdf")
        plt.show()

        # Inputs
        t = t[:-1]
        plt.figure(11)
        plt.subplot(211)
        plt.step(t, U[:, 0], where='post')
        plt.ylabel('$F_{in}$')
        plt.subplot(212)
        plt.step(t, U[:, 1], where='post')
        plt.xlabel('Time (h)')
        plt.savefig("input1.pdf")
        plt.ylabel('$F_1$')

        plt.figure(12)
        plt.subplot(211)
        plt.step(t, U[:, 2], where='post')
        plt.ylabel('$F_r$')
        plt.subplot(212)
        plt.step(t, U[:, 3], where='post')
        plt.xlabel('Time (h)')
        plt.savefig("input2.pdf")
        plt.ylabel('$F_2$')

        plt.figure(13)
        plt.subplot(211)
        plt.step(t, U[:, 4], where='post')
        plt.ylabel('$[GLC]_{in}$')
        plt.subplot(212)
        plt.step(t, U[:, 5], where='post')
        plt.xlabel('Time (h)')
        plt.savefig("input3.pdf")
        plt.ylabel('$GLN_{in}$')

        plt.figure(14)
        plt.step(t, U[:, 6], where='post')
        plt.xlabel('Time (h)')
        plt.savefig("input4.pdf")
        plt.ylabel('$T_c$ (K)')

        plt.figure(15)
        plt.subplot(211)
        plt.step(t, U[:, 7], where='post')
        plt.ylabel('$F_{out, buffer tank}$ (L/h)')
        plt.subplot(212)
        plt.step(t, U[:, 3], where='post')
        plt.xlabel('Time (h)')
        plt.ylabel('$F_{in, buffer tank}$ (L/h)')
        plt.savefig("input5.pdf")
        plt.show()

    def visualize_results_all(self, t, X, U):
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
        plt.savefig("results/integratedmodel/state1_upstream.png")
        plt.show()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(t, X[:, 1])
        plt.ylabel('$X_{t, reactor}$ (cell/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 9])
        plt.xlabel('Time (min)')
        plt.ylabel('$X_{t, separator}$ (cell/L)')
        plt.savefig("results/integratedmodel/state2_upstream.png")
        plt.show()

        plt.figure(3)
        plt.subplot(211)
        plt.plot(t, X[:, 2])
        plt.ylabel('$[GLC]_{reactor}}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 10])
        plt.xlabel('Time (min)')
        plt.ylabel('$[GLC]_{separator}}$ (mM)')
        plt.savefig("results/integratedmodel/state3_upstream.png")
        plt.show()

        plt.figure(4)
        plt.subplot(211)
        plt.plot(t, X[:, 3])
        plt.ylabel('$[GLN]_{reactor}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 11])
        plt.xlabel('Time (min)')
        plt.ylabel('$[GLN]_{separator}$ (mM)')
        plt.savefig("results/integratedmodel/state4_upstream.png")
        plt.show()

        plt.figure(5)
        plt.subplot(211)
        plt.plot(t, X[:, 4])
        plt.ylabel('$[LAC]_{reactor}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 12])
        plt.xlabel('Time (min)')
        plt.ylabel('$[LAC]_{separator}$ (mM)')
        plt.savefig("results/integratedmodel/state5_upstream.png")
        plt.show()

        plt.figure(6)
        plt.subplot(211)
        plt.plot(t, X[:, 5])
        plt.ylabel('$[AMM]_{reactor}$ (mM)')
        plt.subplot(212)
        plt.plot(t, X[:, 13])
        plt.xlabel('Time (min)')
        plt.ylabel('$[AMM]_{separator}$ (mM)')
        plt.savefig("results/integratedmodel/state6_upstream.png")
        plt.show()

        plt.figure(7)
        plt.subplot(211)
        plt.plot(t, X[:, 6])
        plt.ylabel('$[mAb]_{reactor}$ (mg/L)')
        plt.subplot(212)
        plt.plot(t, X[:, 14])
        plt.xlabel('Time (min)')
        plt.ylabel('$[mAb]_{separator}$ (mg/L)')
        plt.savefig("results/integratedmodel/state7_upstream.png")
        plt.show()

        plt.figure(8)
        plt.subplot(211)
        plt.plot(t, X[:, 7])
        plt.ylabel('$V_{reactor}$ (L)')
        plt.subplot(212)
        plt.plot(t, X[:, 15])
        plt.xlabel('Time (min)')
        plt.ylabel('$V_{separator}$ (L)')
        plt.savefig("results/integratedmodel/state8_upstream.png")
        plt.show()

        plt.figure(9)
        plt.plot(t, X[:, 16])
        plt.xlabel('Time (min)')
        plt.ylabel('T (K)')
        plt.savefig("results/integratedmodel/state9_upstream.png")

        plt.figure(10)
        plt.subplot(211)
        plt.plot(t, X[:, 17])
        plt.ylabel('$h_{buffer tank}$ (dm)')
        plt.subplot(212)
        plt.plot(t, X[:, 18])
        plt.xlabel('Time (min)')
        plt.ylabel('$[mAb]_{buffer tank}$ (mg/L)')
        plt.savefig("results/integratedmodel/state10_buffertank.png")
        plt.show()

        plt.figure(11)
        plt.subplot(231)
        plt.plot(t, X[:, 19+13 * 1 + 1])
        plt.plot(t, X[:, 19+13 * 75 + 1])
        plt.plot(t, X[:, 19+13 * 149 + 1])
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$c_p$ [mg/L] r=1")
        plt.xlabel("t [min]")
        plt.subplot(232)
        plt.plot(t, X[:, 19+13 * 1 + 5])
        plt.plot(t, X[:, 19+13 * 75 + 5])
        plt.plot(t, X[:, 19+13 * 149 + 5])
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$c_p$ [mg/L] r=5")
        plt.xlabel("t [min]")
        plt.subplot(233)
        plt.plot(t, X[:, 19+13 * 1 + 10])
        plt.plot(t, X[:, 19+13 * 75 + 10])
        plt.plot(t, X[:, 19+13 * 149 + 10])
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$c_p$ [mg/L] r=10")
        plt.xlabel("t [min]")
        plt.subplot(234)
        plt.plot(t, X[:, 19+13 * 1 + 0])
        plt.plot(t, X[:, 19+13 * 75 + 0])
        plt.plot(t, X[:, 19+13 * 149 + 0])
        # plt.ylim((0.88, 0.91))
        plt.ylabel("$[mAb]_{column}$ [mg/L]")
        plt.xlabel("t [min]")
        plt.subplot(235)
        plt.plot(t, X[:, 19+13 * 1 + 11])
        plt.plot(t, X[:, 19+13 * 75 + 11])
        plt.plot(t, X[:, 19+13 * 149 + 11])
        # plt.ylim((32, 35))
        plt.ylabel("$q_1$ [mg/L]")
        plt.xlabel("t [min]")
        plt.subplot(236)
        plt.plot(t, X[:, 19+13 * 1 + 12])
        plt.plot(t, X[:, 19+13 * 75 + 12])
        plt.plot(t, X[:, 19+13 * 149 + 12])
        # plt.ylim((71, 74))
        plt.ylabel("$q_2$ [mg/L]")
        plt.xlabel("t [min]")
        plt.legend(['level 1', 'level 25', 'level 49'])
        plt.tight_layout()
        plt.savefig("results/integratedmodel/state11_capturecolumn.png")
        plt.show()

        # Inputs
        t = t[:-1]
        plt.figure(12)
        plt.subplot(211)
        plt.step(t, U[:, 0], where='post')
        plt.ylabel('$F_{in}$ (L/min)')
        plt.subplot(212)
        plt.step(t, U[:, 1], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$F_1$ (L/min)')
        plt.savefig("results/integratedmodel/input1.png")
        plt.show()

        plt.figure(13)
        plt.subplot(211)
        plt.step(t, U[:, 2], where='post')
        plt.ylabel('$F_r$ (L/min)')
        plt.subplot(212)
        plt.step(t, U[:, 3], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$F_2$ (L/min)')
        plt.savefig("results/integratedmodel/input2.png")
        plt.show()

        plt.figure(14)
        plt.subplot(211)
        plt.step(t, U[:, 4], where='post')
        plt.ylabel('$[GLC]_{in}$ (mM)')
        plt.subplot(212)
        plt.step(t, U[:, 5], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$[GLN]_{in}$ (mM)')
        plt.savefig("results/integratedmodel/input3.png")
        plt.show()

        plt.figure(15)
        plt.step(t, U[:, 6], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$T_c$ (K)')
        plt.savefig("results/integratedmodel/input4.png")
        plt.show()

        plt.figure(16)
        plt.subplot(211)
        plt.step(t, U[:, 7], where='post')
        plt.ylabel('$F_{out, buffer tank}$ (L/min)')
        plt.subplot(212)
        plt.step(t, U[:, 3], where='post')
        plt.xlabel('Time (min)')
        plt.ylabel('$F_{in, buffer tank}$ (L/min)')
        plt.savefig("results/integratedmodel/input5.png")
        plt.show()

