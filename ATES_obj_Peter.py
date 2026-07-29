# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:21:58 2024

@author: 6100430
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import time 
import math

from line_profiler import profile
from bisect import bisect_left, bisect_right


def find_nearest(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl
    
def get_closests(df, col, val):
    lower_idx = bisect_left(df[col].values, val)
    higher_idx = bisect_right(df[col].values, val)
    if higher_idx == lower_idx:      #val is not in the list
        return lower_idx 
    else:                            #val is in the list
        return lower_idx
    

class ATES_obj:
    """
    ATES object based on data.
    Parameters
    ----------
    supplier : str
        The supplier of heat for the ATES system.
    thickness : float, optional
        Thickness of the aquifer in meters (default: 20).
    porosity : float, optional
        Porosity of the aquifer (default: 0.3).
    kh : float, optional
        Hydraulic conductivity of the aquifer in m/day (default: 10).
    ani : float, optional
        Anisotropy of the aquifer (default: 10).
    T_ground : float, optional
        Ground temperature in the aquifer in degrees Celsius (default: 10).
    density_fluid : float, optional
        Density of the heat transfer fluid in kg/m^3 (default: 997).
    heat_capacity_fluid : float, optional
        Heat capacity of the heat transfer fluid in J/(kg K) (default: 4186).
    costperkW : float, optional
        Capital expenditure per kilowatt (kW) capacity in euros (default: 900).
    fixed_opex : float, optional
        Fixed operational expenditure per kilowatt per year in euros (default: 180).
    var_opex : float, optional
        Variable operational expenditure per kilowatt-hour in euros (default: 12).

    Attributes
    ----------
    data : DataFrame
        Loaded data from the 'results_filtered' parquet file.
    DOE_data : DataFrame
        Processed Design of Experiment data containing unique combinations of input parameters, further defined in a paper to be published
    last_year_data : DataFrame
        Subset of data containing records from the last year of ATES operation, being the 8th year.

    Methods
    -------
    __init__
        Initializes the ATES_obj with specified parameters and loads required data.
    initialize(volume, T_in, len_timestep)
        Initializes the ATES system with specified volume, injected temperature, and time step.
    predict_reff(volume, T_in)
        Predicts the recovery efficiency based on a machine learning model.
    nearest_neighbour(T_in, reff, volume)
        Finds the nearest neighbors in the data for temperature prediction.
    predict_temp_out(T_in, reff, volume)
        Predicts the outlet temperature based on nearest neighbors.
    correct_for_volume(volume, temp_out, len_timestep)
        Corrects the outlet temperature for the specified volume and time step.
    calc_heat(T_cutoff, T_demand_out, storage_extraction, missing_energy, HP=None, len_timestep=3600, firstyear=False)
        Calculates the heat extracted from the aquifer a given time period.
    remove_data_point(thickness, porosity)
        Removes a data point from the dataset based on specified thickness and porosity. (Unused)
    re_add_data_point()
        Re-adds a previously removed data point back to the dataset. (Unused)

    Notes
    -----
    This class encapsulates the functionality of an ATES system, including data handling, temperature prediction, and heat calculation.
    """
    def __init__(self, supplier, thickness = 20, porosity = 0.3, kh = 10, 
                 ani = 10, T_ground = 10,density_fluid =997,
                 heat_capacity_fluid = 4186,costperm3 = 3400000/320, 
                 HP=None, max_V=150, 
                 depth = 400, N_wells=2, var_opex = 2/40, fixed_opex = 765.6,
                 lifetime = 25, HX_eta = 0.9,start_full_volume = True,timing=False,
                 elec_price = 0.2,pump_efficiency=0.5):
        # Save data
        self.supplier = supplier
        self.name = 'ATES'
        self.control = 'storage'
        self.type = 'supply'
        self.max_V = max_V #m^3/hour
        self.thickness = thickness
        self.por = porosity
        self.kh = kh
        self.ani=ani
        self.T_g = T_ground
        self.density = density_fluid #kg / m^3
        self.heat_cap = heat_capacity_fluid #J/(kg K)
        self.depth = depth
        self.capex = costperm3*max_V  #euro --> kw * (euro/kW)
        self.fix_opex = fixed_opex*max_V #euro/yr
        self.var_opex = var_opex #euro/kg, 
        self.lifetime = lifetime
        self.elec_price = elec_price #euro/kWh
        self.pump_efficiency=pump_efficiency
        #source:https://www.warmingup.info/documenten/feasibility-study-for-combined-geothermal-and-ht-ates-systems.pdf
        #Taking the fix_opex and var_opex number divided by delta_T
                
        self.start_full_volume = start_full_volume
        self.timing=timing #Whether to print timing to terminal
        self.Reff_set=False
        #Currently unused
        self.HX_eta = HX_eta

        if HP==None:
            self.HP=None
        elif HP.name=="Heat pump":
            self.HP = HP
        else:
            print("ATES connected Heat pump is not recognised, set to no Heat pump")
        

        if timing:
            start = time.time()
        # Get data from earlier research, saved in parquet file and manipulate it 
        self.data = pd.read_parquet('results_AXI_V2')

        if timing:
            print('Loading parquet data took {}s'.format(time.time() - start))


        self.last_year_data = self.data[self.data.Day>2554]


    def initialize(self,volume,T_in, len_timestep):
        """
        Initializes the ATES_obj instance with the given volume, injected temperature, and length of timestep.
        This should be called when adding the ATES system
        
        Parameters
        ----------
        volume : float
            Volume of the injected water in cubic meters.
        T_in : float
            Injected temperature in degrees Celsius.
        len_timestep : float
            Length of each timestep in seconds.
        
        This method involves the following steps:
        1. Predicts the recovery efficiency (reff) using the predict_reff method.
        2. Predicts the outlet temperature (temp_out) using the predict_temp_out method.
        3. Corrects the data for the given volume and temp_out using the correct_for_volume method.
        
        Parameters
        ----------
        volume : float
            Volume of the aquifer in cubic meters.
        T_in : float
            Injected temperature in degrees Celsius.
        len_timestep : float
            Length of each timestep in seconds.
        """
        # Step 1: Predict recovery efficiency using ML
        self.volume=volume
        if volume < 1:
            print("Volume smaller than 1 m^3 per year, set to 0")
            volume = 0
            return
        if self.timing:
            start = time.time()
            
        if self.Reff_set!=True:
                
            reff = self.predict_reff(volume, T_in)
            self.Reff= reff
        if self.timing:
            print('Predicting reff took {}s'.format(time.time() - start))
        
        if self.timing:
            start = time.time()
        # Step 2: Predict outlet temperature based on the Reff, T_in, V
        temp_out = self.predict_temp_out(T_in, self.Reff, volume)
        
        if self.timing:
            print('Nearest neighbour search took {}s'.format(time.time() - start))
        
        
        if self.timing:
            start = time.time()
            
        if self.start_full_volume == True:
            
            temp_out = self.Correct_half_volume(temp_out)

        # Step 3: Correct data for volume
        self.correct_for_volume(volume, T_in,temp_out, len_timestep)
        
        if self.timing:
            print('Manipulation took {}s'.format(time.time() - start))
    def predict_reff(self, volume, T_in):
        """
        Predicts recovery efficiency based on the given volume and injected temperature.
        
        Parameters
        ----------
        volume : float
            Volume of the injected water in cubic meters.
        T_in : float
            Injected temperature in degrees Celsius.
        
        Returns
        -------
        float
            Predicted recovery efficiency of the aquifer.
        
        Notes
        -----
        - The Ml model is generated based on the data found in self.data. It uses boostedregression from scikit.learn
        - The predicted recovery efficiency is obtained and stored in the instance variable Reff.
        """
        # Load the ML model
        model = joblib.load("Predict_REFF_boostedregression.pkl")        


        # Prepare inputs for prediction
        Reff = model.predict(pd.DataFrame({'Porosity':self.por,
                                           "Volume" :volume,
                                           "T_injected_hot" :T_in,
                                           "T_ground":self.T_g,
                                           "thickness aquifer" :self.thickness,
                                           'Hydraulic conductivity aquifer':self.kh,
                                           'anisotropy':self.ani},index=[0]))
        # Get a float out, not a list 
        Reff=Reff[0]
        
        if volume<50000:
            pass
            #print("Volume injected into aquifer very low, consider increasing")
        ## Old model, used interpolation between DoE
        # if T_in>80:
        #     T_in= 80
        #     print("T_in set to 80 degrees for predicting recovery efficiency. If T_in <100 this is fine, most likely")
        # elif T_in < 25:
        #     T_in = 25
        #     print("T_in set to 25 degrees for predicting recovery efficiency. Check if this is intended")
        # try:
        #     Reff = float(interpn(self.points, self.values, np.array([self.por,volume,T_in,self.T_g,self.thickness,self.kh,self.ani])))
        # except:
        #     ValueError("Either Volume is too high or low to be realistic or T_in is lower than 25 degrees")
        #     Reff = 0
        
        # Store the Reff and return it
        #print(Reff,volume)
        return Reff
    
    def nearest_neighbour(self,T_in,reff,volume):
        """
        Finds the nearest neighbors in the dataset based on input parameters.
    
        Parameters
        ----------
        T_in : float
            Injected temperature in degrees Celsius.
        reff : float
            Recovery efficiency.
        volume : float
            Volume of the aquifer in cubic meters.
    
        Returns
        -------
        Tuple
            A tuple containing the indices of the nearest neighbors and their total distances.
    
        Notes
        -----
        - Calculates relative distances for temperature, recovery efficiency, ground temperature, and volume.
        - Computes the total distance as the Euclidean norm of the relative distances.
        - Finds the indices of the nearest neighbors and their total distances.
        """
        #Calculate relative distances
        relative_distance_1 = abs(((self.data["T_injected_hot"])-(T_in))/90)#(T_in))
        relative_distance_2 = abs((self.data["Efficiency_well_lastyear"]-reff)/0.9)#/reff)
        relative_distance_3 = abs((self.data["T_ground"]-self.T_g)/30)#self.T_g)
        relative_distance_4 = 0#abs((self.data["Volume"]-volume)/volume)
        relative_distance_5 = 0#((self.data["anisotropy"]-self.ani)/self.ani)**2+((self.data['Porosity']-self.por)/self.por)**2+((self.data["thickness aquifer"]-self.thickness)/self.thickness)**2+((self.data['Hydraulic conductivity aquifer']-self.kh)/self.kh)**2


        # Compute the total distance
        total_distance = np.sqrt(relative_distance_1+relative_distance_2+relative_distance_3+relative_distance_4+relative_distance_5)

        # Find the indices of the nearest neighbors
        lowest = total_distance[total_distance==total_distance.min()].index

        return lowest,total_distance
    def predict_temp_out(self,T_in,reff,volume):
        """
        Predicts the outlet temperature based on the nearest neighbors.
        
        Parameters
        ----------
        T_in : float
            Injected temperature in degrees Celsius.
        reff : float
            Recovery efficiency.
        volume : float
            Volume of the aquifer in cubic meters.
        
        Returns
        -------
        pandas.Series
            Predicted outlet temperature for all of the 8th years.
        
        Notes
        -----
        - Finds the nearest neighbors and their total distances using the nearest_neighbour method.
        - Retrieves the outlet temperature of the nearest neighbors.
        - Normalizes the temperature values.
        - Stores the predicted outlet temperature in the instance variable temp_out.
        - Returns the predicted outlet temperature.
        """
        # Find the nearest neightbours and their total distance
        lowest,total_distance = self.nearest_neighbour(T_in,reff,volume)
        
        # Retrieve the outlet temperature of the nearest neighbors
        temp_out = self.data.loc[lowest]["Outlet_T_hotwell"]

        # Correct the temperature if nearest neighbour temperature is not exactly the same
        temp_out = temp_out/(temp_out.iloc[-1]/T_in)

        #Store and return the outlet temperature
        self.temp_out=temp_out
        return temp_out
    def correct_for_volume(self,volume, T_in,temp_out,len_timestep):
        """
        Corrects the temperature output based on the provided volume.
        This is based on the data calculation, which used periods of 1 week.
        
        Parameters
        ----------
        volume : float
            Volume in cubic meters per year.
        temp_out : pd.Series
            Temperature output.
        len_timestep : int
            Length of each time step in seconds.
        
        Notes
        -----
        - Computes flow based on provided volume.
        - Adjusts the temperature output accordingly.
        """
        #Volume in m^3 per year

        perlen=7
        PerPerYear= int(round(365/perlen, 0))

        # Calculate flow
        flow = self.calculate_flow(volume, PerPerYear)  
        
        # Set up time index
        self.elongation_constant=1
        index = pd.Series(np.linspace(0,416*24*7*self.elongation_constant,417),dtype = int)
        #### 5 is removed
        #Calculate flow

        flow = np.cumsum(flow.clip(min=0))

        # Calculate temperature out based on the volume
        temp_out.reset_index(drop=True,inplace=True)

        temp_out = pd.concat([temp_out,pd.Series(flow,name="flow")],axis=1)
        temp_out = temp_out.set_index(index)
        # self.total_heat_extracted_vs_T_ground_kWh_first_8_years=np.zeros(8)
           # for i in range(8):
        #     self.total_heat_extracted_vs_T_ground_kWh_first_8_years[i] = sum(np.diff(temp_out.loc[(i)*8736:(i+1)*8736,"flow"],prepend=0)*(temp_out.loc[(i)*8736:(i+1)*8736,"Outlet_T_hotwell"]-self.T_g)*1000*4186/3600000)
        # # Interpolate between missing values
        temp_out = temp_out.reindex(range(int(temp_out.index.min()),int(temp_out.index.max())))
        temp_out = temp_out.interpolate()

        # Interpolation and taking the nearest neighbour messes with the Reff
        # Therefore correct for the Reff again. Reff of ML is quite accurate, so stick to it.
        after_inter = sum((temp_out.iloc[-(52*24*7*self.elongation_constant):,0]-self.T_g)*np.diff(temp_out.iloc[-(52*24*7*self.elongation_constant):,1],prepend=min(temp_out.iloc[-(52*24*7*self.elongation_constant):,1])))/((T_in-self.T_g)*volume)
        factor = 5
        factor_save=5
        while factor < 0.99 or factor >1.01 :
            factor = self.Reff/after_inter
            if abs(1/factor_save-1)<abs(factor-1):
                factor = ((factor)-1)*0.5+1
            temp_out.loc[:,"Outlet_T_hotwell"]=(temp_out.loc[:,"Outlet_T_hotwell"]-self.T_g)*(((factor-1)*1)+1)+self.T_g
            #temp_out.loc[:,"Outlet_T_hotwell"]=(temp_out.loc[:,"Outlet_T_hotwell"]-T_in)/(((factor-1))+1)+T_in
            after_inter = sum((temp_out.iloc[-(52*24*7*self.elongation_constant):,0]-self.T_g)*np.diff(temp_out.iloc[-(52*24*7*self.elongation_constant):,1],prepend=min(temp_out.iloc[-(52*24*7*self.elongation_constant):,1])))/((T_in-self.T_g)*volume)
            factor_save=factor
        

        #Save it
        self.output_t = temp_out.copy()
        self.output_t_firstyear = self.output_t.head(8760*self.elongation_constant).copy()
        self.output_t_lastyear = self.output_t.tail(8760*self.elongation_constant).copy()
                
        # Correct flow, for earlier flows
        self.output_t_lastyear.loc[:,"flow"] = self.output_t_lastyear["flow"] - min(self.output_t_lastyear["flow"]) 
        difference = np.diff(self.output_t_lastyear["flow"],prepend= 0)
        # T_ave = sum((self.output_t_lastyear.loc[:,"Outlet_T_hotwell"])*difference)/sum(difference)
        self.output_t_lastyear = self.output_t_lastyear[difference>0]

    def calculate_flow(self, volume, PerPerYear):
        """
        Calculates the flow based on the provided volume.

        Parameters
        ----------
        volume : float
            Volume in cubic meters per year.
        weeks_per_year : int
            Number of weeks in a year.

        Returns
        -------
        np.ndarray
            Array representing the calculated flow.
        """
        sum_sine = 0
        periods_per_half_year = int(PerPerYear / 2)
        flow = np.zeros(417)

        # Calculate sum of sine values
        for i in range(periods_per_half_year):
            sine = np.sin(np.pi * i / periods_per_half_year)
            sum_sine += sine

        # Calculate flow
        for j in range(len(flow)):
            flow[j] = round(np.cos(np.pi * j / periods_per_half_year) / sum_sine * (-1) * volume, 0)

        return flow
    
    def func_fit(self,x, a, b,c): # polytrope equation from zunzun.com
        return a+b/(2**(x/c))
    
    def Correct_half_volume(self, temp_out):
        """
        Corrects the temperature output for half-volume cycles.
        
        Parameters
        ----------
        temp_out : pandas.Series
            Temperature output data.
      
        Returns
        -------
        pandas.Series
            Corrected temperature output data.
     
        Notes
        -----
        The function identifies half-volume cycles in the temperature output data and corrects them by fitting a curve
        and adjusting the values accordingly.      
        """
        
        # Skip the first year of the temperature profile
        keep = temp_out[52:417]
        add = temp_out[417-52:417]
        temp_out = pd.concat([keep,add])
        temp_out.reset_index(drop=True,inplace=True)
        return temp_out
        
    def set_reff(self, Reff):
        if Reff > 1:
            ValueError("Reff higher than 100%, don't do that")
        self.Reff = Reff
        self.Reff_set = True
    def _energy_split(self, flow_m3, T_mean, T_inj, T_demand_out, hp_running):
        """
        Split the heat delivered in one timestep into its physical components.

        flow_m3   : volume extracted this timestep [m3]
        T_mean    : mean hot-well extraction temperature over the timestep [C]
        T_inj     : cold-well injection temperature this timestep [C]
                    (= self.T_return if HP off, = self.T_floor if HP on)
        Returns (Q_dir, Q_evap, P_el, Q_tot, COP) in kWh.
        """
        C_A = flow_m3 * self.density * self.heat_cap / 3.6e6      # [kWh/K]

        # (a) Direct HX: only the part of the ATES water ABOVE the DHN return
        Q_dir = C_A * max(0.0, T_mean - self.T_return)

        if not hp_running or self.HP is None:
            return Q_dir, 0.0, 0.0, Q_dir, np.nan

        # (b) HP evaporator: from min(T_mean, T_return) down to the cold-well floor
        T_evap_in = min(T_mean, self.T_return)
        Q_evap = C_A * max(0.0, T_evap_in - T_inj)
        if Q_evap <= 0.0:
            return Q_dir, 0.0, 0.0, Q_dir, np.nan

        # (c) Compressor work.  Q_cond = Q_evap + W  and  COP = Q_cond / W
        # P: ToDo Check if implementation of HP with temperatures and everythin is correct
        # Sensitivity vs David's inlet convention + pinch: revisit later.
        T_source = 0.5 * (T_evap_in + T_inj)          # mean source T over the evaporator
        COP = self.HP.Calculate_COP(T_demand_out, T_source) #P: Check if this is necessary (guard on COP < 1), implement decision if C is feasible later
        #if not np.isfinite(COP) or COP <= 1.05:
            #return Q_dir, 0.0, 0.0, Q_dir, np.nan  # HP not viable this step
        P_el = Q_evap / (COP - 1.0)

        return Q_dir, Q_evap, P_el, Q_dir + Q_evap + P_el, COP

    def calc_heat(self, T_cutoff, T_demand_out, storage_extraction, missing_energy,
                  hp_on=None, hp_override_below_cutoff=True,
                  HP=None, len_timestep=3600, firstyear=False, control=None):
        """
        Calculates the energy output based on temperature constraints and missing energy.

        Parameters
        ----------
        T_cutoff : float
            DHN return temperature. Below this the ATES water cannot heat the network
            passively, so it is both the HX floor and the mode-A injection temperature.
        T_demand_out : float
            DHN supply temperature (the HP condenser sink).
        storage_extraction : array
            Mask defining when the ATES is allowed to run.
        missing_energy : pd.Series
            Heat still to be covered, per timestep [kWh].
        hp_on : array of bool, optional
            Per-timestep HP dispatch intent from main2. None -> HP never runs (mode A).
        hp_override_below_cutoff : bool, optional
            If True, the ATES forces the HP on whenever the extraction temperature has
            dropped below T_cutoff (mode D), because the HX can deliver nothing there.
        len_timestep : int, optional
            Length of each timestep in seconds.
        firstyear : bool, optional
            Flag indicating if it's the first year.

        Returns
        -------
        np.ndarray
            Total heat delivered to the DHN per timestep [kWh] (= ATES + HP).
            The split is stored on the object: output_dir, output_evap, output_HP,
            P_el, COP, T_extract, T_inject, mode.
            P: How does this differ from the old version? Does the old version only contain one column, while the new one contains multiple ones and the sum of each row is the total heat delivered per timestep?
        """
        self.len_timestep = len_timestep
        max_flow_generated = self.max_V * len_timestep / 3600
        missing_energy = missing_energy * storage_extraction

        # --- Result arrays -----------------------------------------------------
        n = len(missing_energy)
        flow = 0
        #P: What is all this done for?
        output              = np.zeros(n)   # total heat to the DHN [kWh]
        self.flow_extracted = np.zeros(n)
        self.output_dir     = np.zeros(n)   # ATES via HX
        self.output_evap    = np.zeros(n)   # ATES via HP source side
        self.output_HP      = np.zeros(n)   # HP condenser output = evap + P_el
        self.P_el           = np.zeros(n)   # compressor electricity
        self.COP            = np.full(n, np.nan)
        self.T_extract      = np.zeros(n)   # realized mean extraction T
        self.T_inject       = np.full(n, float(T_cutoff))   # realized cold-well injection T
        self.mode           = np.full(n, 'off', dtype=object)

        # --- 8-year extractable-heat accounting (unchanged, feeds LCOH) --------
        # NOTE: measured against the un-lowered T_cutoff. Conservative when the HP runs. (Check P)
        self.total_heat_extracted_vs_T_ground_kWh_first_8_years = np.zeros(8)
        difference = (self.output_t.loc[:, "Outlet_T_hotwell"] - T_cutoff)
        difference[difference < 0] = 0
        # Calculate the heat output of the first 8 years of the hot well based on the difference with the output temperature of the heat network
        for i in range(8):
            self.total_heat_extracted_vs_T_ground_kWh_first_8_years[i] = \
                sum(np.diff(self.output_t.loc[(i)*8736*self.elongation_constant:(i+1)*8736*self.elongation_constant-1, "flow"],
                            prepend=min(self.output_t.loc[(i)*8736*self.elongation_constant:(i+1)*8736*self.elongation_constant-1, "flow"]))
                    * (difference[(i)*8736*self.elongation_constant:(i+1)*8736*self.elongation_constant]) * 1000*4186/3600000)

        # --- Starting hot-well temperature ------------------------------------- (P: What is done here? Old comment:# Find the starting temperature of the ATES is on removing any unnecessary numbers from the flow.
        """try:
            T_start = max(self.output_t_lastyear["Outlet_T_hotwell"] * (self.output_t_lastyear["flow"] > 0))
        except Exception:
            T_start = T_cutoff #P: What is this?"""
        # Starting hot-well temperature: warmest point of the last-year curve with positive flow.
        masked_T = self.output_t_lastyear["Outlet_T_hotwell"] * (self.output_t_lastyear["flow"] > 0)
        if len(masked_T) and masked_T.max() > 0:
            T_start = masked_T.max()

        else:
            print("Warning: no positive-flow extraction curve; T_start set to T_cutoff")
            T_start = T_cutoff

        if min(self.output_t_lastyear["Outlet_T_hotwell"]) > T_cutoff: #P: is this msg neccessary?
            if self.HP is not None:
                print("Heat pump not necessary for boosting ATES temperature: used for "
                      "direct heating of return temperature, please consider the feasibility of this")

        # --- Temperature levels (set once, never mutated) -----------------------
        self.T_return = T_cutoff                                    # HX floor
        if self.HP is not None:
            self.T_floor = max(T_cutoff - self.HP.delta_T_coldside, self.T_g)
            if T_cutoff - self.HP.delta_T_coldside < self.T_g:
                print(f"Warning: HP floor below ground temperature; clipped to T_g = {self.T_g}")
        else:
            self.T_floor = T_cutoff                                 # no HP -> nothing below the return

        # --- HP dispatch intent from main2 --------------------------------------
        if hp_on is None or self.HP is None:
            hp_on = np.zeros(n, dtype=bool)
        else:
            if len(hp_on) != n:
                raise ValueError("hp_on must be one value per timestep")
            hp_on = np.asarray(hp_on).astype(bool)

        # Backward compatibility: old main2 reads storage_obj.HP.COP
        if self.HP is not None:
            self.HP.COP = self.COP

        # --- Iterate, starting halfway through the year (after summer) ----------
        index = missing_energy[missing_energy > 0]
        index1 = index.where(index.index > np.mean(missing_energy.index)).dropna()
        index2 = index.where(index.index <= np.mean(missing_energy.index)).dropna()
        index = pd.concat([index1, index2])

        if control == "Peak shaving":
            usefull_volume = self.output_t_lastyear[self.output_t_lastyear.Outlet_T_hotwell < T_cutoff-1].flow.iloc[0]
            peak_shaving_control = usefull_volume / sum(np.clip(missing_energy, a_min=0, a_max=None)) \
                                   * np.clip(missing_energy, a_min=0, a_max=None)
            rest = 0
            for i, ele in enumerate(peak_shaving_control):
                if i == len(peak_shaving_control)-1:
                    break
                elif peak_shaving_control[i] > max_flow_generated:
                    rest = peak_shaving_control[i] - max_flow_generated
                    peak_shaving_control[i] = max_flow_generated
                    peak_shaving_control[i+1] = peak_shaving_control[i+1] + rest
                    rest = 0
            if rest != 0:
                peak_shaving_control[0] = peak_shaving_control[0] + rest

        max_flow  = self.output_t_lastyear["flow"].iloc[-1]
        flow_vals = self.output_t_lastyear["flow"].values
        T_vals    = self.output_t_lastyear["Outlet_T_hotwell"].values
        n_curve   = len(T_vals)

        for t in index.index:

            # ---- 1. Pick the trial flow and the resulting end-of-step temperature ---- #P: Peak shaving still buggy
            if control == "Peak shaving":
                max_flow_generated = peak_shaving_control[t]
            else:
                if max_flow_generated + flow > max_flow:
                    max_flow_generated = max(0.0, max_flow - flow)

            idx = min(bisect_left(flow_vals, max_flow_generated + flow), n_curve - 1)
            T_after = T_vals[idx]

            # ---- 2. Decide the HP state for this hour --------------------------------
            T_mean = 0.5 * (T_after + T_start)
            hp_running = bool(hp_on[t])

            # Mode-D override: below the DHN return the HX delivers nothing, so the HP
            # is the only way to get heat out. The ATES overrules main2.
            if (self.HP is not None and hp_override_below_cutoff
                    and T_mean < self.T_return):
                hp_running = True

            T_inj = self.T_floor if hp_running else self.T_return

            # ---- 3. Heat for the trial flow ------------------------------------------- #P: What trial flow?
            Q_dir, Q_evap, P_el, energy_generated, COP = self._energy_split(
                max_flow_generated, T_mean, T_inj, T_demand_out, hp_running)

            # ---- 4. Well depleted for this mode? -------------------------------------- #P: What is this?
            if energy_generated <= 0.0:
                max_flow_generated = self.max_V * len_timestep / 3600
                continue                      # no flow, do NOT advance T_start

            # ---- 5. Don't over-deliver: shrink the flow to match missing_energy -------- #P: Why should energy generated be too big in the first place? What does this do
            if energy_generated > missing_energy[t]:
                for _ in range(20):
                    factor = missing_energy[t] / energy_generated
                    if 0.995 <= factor <= 1.0:
                        break
                    max_flow_generated *= factor
                    idx = min(bisect_left(flow_vals, max_flow_generated + flow), n_curve - 1)
                    T_after = T_vals[idx]
                    T_mean = 0.5 * (T_after + T_start)
                    Q_dir, Q_evap, P_el, energy_generated, COP = self._energy_split(
                        max_flow_generated, T_mean, T_inj, T_demand_out, hp_running)
                    if energy_generated <= 0.0:
                        break
                if energy_generated <= 0.0:
                    max_flow_generated = self.max_V * len_timestep / 3600
                    continue

            # ---- 6. Store --------------------------------------------------------------
            hp_active = (Q_evap > 0.0)

            self.flow_extracted[t] = max_flow_generated
            self.output_dir[t]     = Q_dir
            self.output_evap[t]    = Q_evap
            self.output_HP[t]      = Q_evap + P_el
            self.P_el[t]           = P_el
            self.COP[t]            = COP
            self.T_extract[t]      = T_mean
            self.T_inject[t]       = T_inj if hp_active else self.T_return
            self.mode[t] = ('A' if not hp_active
                            else ('B' if T_mean >= self.T_return else 'D'))
            output[t] = energy_generated

            # ---- 7. Advance ------------------------------------------------------------
            T_start = T_after
            flow   += max_flow_generated
            max_flow_generated = self.max_V * len_timestep / 3600

        self.flow_extracted = np.clip(self.flow_extracted, a_min=0, a_max=None)
        return output

    def Thiem_equation(self):
        rw=0.3
        flow = self.flow_extracted+self.flow_injected
        rh = np.sqrt((self.volume)/(self.por*np.pi*self.thickness))*2
        dh = (flow/(self.len_timestep/24/3600))/(2*np.pi*self.kh*self.thickness)*np.log(rh/rw)
        #Calculate the head difference using radial flow thiem equation [m]
        #h1-h2 = Q/(2*pi*kh*thickness)*ln(r2/r1)
        return dh
        
    def init_cold_well(self, T_in, volume):
        reff = self.predict_reff(volume, T_in)
        T_ave = (T_in-self.T_g)*reff+self.T_g
        self.cold_well_reff = reff
        self.cold_well_T_ave = T_ave

    
    def calc_opex(self, kWh_generated):
        try:
            #opex = sum(self.flow_extracted)*self.var_opex
            opex = sum(abs(self.Thiem_equation())*10*2*abs(self.flow_extracted+self.flow_injected)/self.len_timestep/self.pump_efficiency*self.elec_price)*(self.len_timestep/3600)
            #Calculate the power required from the pumps based on the thiem equation (See Daniilidis et al. (2022) for equation)
            #This is times two to represent the cold well as well (approximation).
            opex = opex+self.fix_opex
        except:
            opex = 0

        return opex
    def calc_emissions(self,result):
        #TO DO add the emissions from the electricity.
        try:
            sum_CO2 = 0
            for i in self.supplier:
                sum_CO2 = i.CO2_kg+sum_CO2
            return_value =  sum(result["ATES corrected"])*(sum_CO2/self.Reff/len(self.supplier))
            
        except:
            return_value = 0
        return return_value
    def remove_data_point(self,thickness,porosity):
        location = self.data[self.data["Porosity"]==porosity]
        self.store_data = location
        self.data.drop(index=location.index,inplace=True)
        return location
    
    def re_add_data_point(self):
        self.data = pd.concat([self.data,self.store_data])
        self.data.sort_index(inplace=True)





if __name__ == "__main__":

    #Parameters
    thickness_aquifer = 50 #[m] Thickness of the aquifer (assuming homogenous and constant thickness)
    porosity = 0.2 #[-] porosity aquifer
    horizontal_conductivity = 5  #[m day^-1] Horizontal hydraulic conductivity
    anisotropy =1 #[-] Horizontal hydraulic conductivity/vertical hydraulic conductivity
    ground_temperature = 25 #[degrees C] Undisturbed ground temperature
    supplier = 0
    time_sum=0
    start_total=time.time()
    for i in range(10):
        start = time.time()
        ATES = ATES_obj(supplier, max_V = 100,thickness=thickness_aquifer, porosity=porosity,kh=horizontal_conductivity,
                        ani=anisotropy,T_ground=ground_temperature)

        Volume =403680 #m^3/year, volume injected as well as extracted (assuming mass balance needs to be preserved)
        Temp_in = 90 #[degrees C] Temperature of the water going in the aquifer
        ATES.initialize(Volume, Temp_in, 3600) #Generates values for T_out
        print(" Time taken", time.time()-start)
    print("Time taken per thing",(time.time()-start_total)/50)
    plt.plot(ATES.output_t.loc[:,"flow"],ATES.output_t.loc[:,"Outlet_T_hotwell"])
    plt.xlabel("Volume (m^3) extracted")
    plt.ylabel("Temperature out of ATES")
    print("Predicted recovery efficiency = ",ATES.Reff)
    plt.figure(dpi=800)
    plt.plot(np.linspace(0,8,417),ATES.temp_out,label="Well temperature")
    plt.xlabel("Time (years)")
    plt.ylabel("Well temperature (Celsius)")
    ylim=plt.ylim()
    plt.ylim(ylim)
    new_tick_locations = ((np.array([.5, 1, 1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])-0.25))
    plt.vlines(new_tick_locations,ymin=0,ymax=90,color="grey",alpha=0.5,label="Operation mode switch")
    plt.legend()


    # This is for multiple cycles. Each cycle containing the amount of volume.



if __name__ == "__main__":

    # --- Aquifer / system parameters ---
    thickness_aquifer       = 50    # [m]
    porosity                = 0.2   # [-]
    horizontal_conductivity = 5     # [m/day]
    anisotropy              = 1     # [-]
    ground_temperature      = 25    # [deg C]
    supplier                = 0

    Volume       = 403680   # [m^3/yr] injected = extracted
    len_timestep = 3600     # [s]

    ATES = ATES_obj(supplier, max_V=100, thickness=thickness_aquifer,
                    porosity=porosity, kh=horizontal_conductivity,
                    ani=anisotropy, T_ground=ground_temperature)

    injection_temps = list(range(50, 19, -5))   # [50,45,40,35,30,25,20]
    print("Sweep injection temps:", injection_temps, flush=True)

    T_in_arr, T_ave_arr = [], []
    print(f"\n{'T_inj [C]':>10} | {'avg T_extract [C]':>17}", flush=True)
    print("-" * 31, flush=True)

    for T_in in injection_temps:
        ATES.init_cold_well(T_in, Volume)
        T_in_arr.append(T_in)
        T_ave_arr.append(ATES.cold_well_T_ave)
        print(f"{T_in:>10.1f} | {ATES.cold_well_T_ave:>17.2f}", flush=True)

    # --- Visual ---
    plt.figure(dpi=150)
    plt.plot(T_in_arr, T_ave_arr, "o-", label="avg cold-well extraction T")
    plt.axhline(ground_temperature, color="grey", ls="--", alpha=0.6,
                label="ground temperature")
    plt.plot(T_in_arr, T_in_arr, color="black", ls=":", alpha=0.4,
             label="injection = extraction (1:1)")
    plt.xlabel("Cold-well injection temperature [C]")
    plt.ylabel("Average cold-well extraction temperature [C]")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()