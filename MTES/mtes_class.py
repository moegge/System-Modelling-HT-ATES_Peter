import numpy as np

class MTES:
    def __init__(self, T_tank, T_rock, step_size, L, lambda_rock, r_tank, r_th_max, m_tank, m_rock, cp_water, cp_rock, rho_rock):
        self.T_tank = T_tank
        self.T_rock = T_rock
        self.step_size = step_size
        self.L = L
        self.lambda_rock = lambda_rock 
        self.r_tank = r_tank
        self.r_th_max = r_th_max
        self.m_tank = m_tank
        self.m_rock = m_rock
        self.cp_water = cp_water
        self.cp_rock = cp_rock
        self.rho_rock = rho_rock 

    def do_step(self, T_in, m_dot):
        # 1. Apply inflow heating/cooling
        Q_in = m_dot * self.cp_water * (T_in - self.T_tank) * self.step_size
        delta_T_in = Q_in / (self.m_tank * self.cp_water)
        self.T_tank += delta_T_in
        # 2. Conduction with rock buffer
        Q_cond = (2 * np.pi * self.L * self.lambda_rock * (self.T_tank - self.T_rock) * self.step_size) / (np.log(self.r_th_max / self.r_tank))
        self.T_tank -= Q_cond / (self.m_tank * self.cp_water)
        self.T_rock += Q_cond / (self.m_rock * self.cp_rock)
        

    

    