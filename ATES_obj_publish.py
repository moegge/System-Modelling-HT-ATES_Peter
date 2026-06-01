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
    def calc_heat(self,T_cutoff,T_demand_out, storage_extraction, missing_energy,
                  HP=None,len_timestep=3600,firstyear=False,control = None):
        """
        Calculates the energy output based on temperature constraints and missing energy.
       
        Parameters
        ----------
        T_cutoff : float
            Cutoff temperature from the grid.
        T_demand_out : float
            Desired outlet temperature of the grid input.
        storage_extraction : float
            Storage extraction input that defines when the ATES system can be on, 
            assuming you do not want to switch it on/off multiple times.
        missing_energy : pd.Series
            Series containing missing energy values.
        HP : Any, optional
            Heat pump information. 
        len_timestep : int, optional
            Length of each time step in seconds.
        firstyear : bool, optional
            Flag indicating if it's the first year.
       
        Returns
        -------
        np.ndarray
            Array representing the calculated energy output.
        """
        self.len_timestep = len_timestep
        # Get the max flow and the amount of energy to be filled by ATES
        max_flow_generated = self.max_V*len_timestep/3600
        
        missing_energy = missing_energy*storage_extraction
        
        # Initialize flow and output as 0
        flow = 0
        self.flow_extracted = np.zeros(len(missing_energy))
        output = np.zeros(len(missing_energy))
        #Heat_generated_HP= np.zeros(len(missing_energy))
        
#        self.total_heat_extracted_vs_T_ground_kWh_first_8_years = sum(np.diff(self.output_t.loc[:,"flow"],prepend=0)*(self.output_t["Outlet_T_hotwell"]-T_cutoff)*1000*4186/3600000)
        self.total_heat_extracted_vs_T_ground_kWh_first_8_years=np.zeros(8)
        difference = (self.output_t.loc[:,"Outlet_T_hotwell"]-T_cutoff)
        difference[difference<0]=0
        #Calculate the heat output of the first 8 years of the hot well based on the difference with the output temperature of the heat network
        for i in range(8):
            self.total_heat_extracted_vs_T_ground_kWh_first_8_years[i] = \
                sum(np.diff(self.output_t.loc[(i)*8736*self.elongation_constant:(i+1)*8736*self.elongation_constant-1,"flow"],\
                            prepend=min(self.output_t.loc[(i)*8736*self.elongation_constant:(i+1)*8736*self.elongation_constant-1,"flow"]))\
                    *(difference[(i)*8736*self.elongation_constant:(i+1)*8736*self.elongation_constant])*1000*4186/3600000)
                    
        # Find the starting temperature of the ATES is on removing any unnecessary numbers from the flow.
        try:
            T_start = max(self.output_t_lastyear["Outlet_T_hotwell"]*(self.output_t_lastyear["flow"]>0))
        except:
            x = 5
       
        if min(self.output_t_lastyear["Outlet_T_hotwell"])>T_cutoff:
            if self.HP != None:
                print("Heat pump not necessary for boosting ATES temperature: used for direct heating of return temperature, please consider the feasibility of this")
            
        #Initialize the heat pump                
        if self.HP!=None:
            self.HP.COP = np.zeros(len(missing_energy))
            T_cutoff=T_cutoff-self.HP.delta_T_coldside
        
        # Iterate over the missing energy, start at halfway through year (after summer) done by changing index
        
        index = missing_energy[missing_energy>0]
        index1 = index.where(index.index>np.mean(missing_energy.index)).dropna()
        index2 = index.where(index.index<=np.mean(missing_energy.index)).dropna()
        index = pd.concat([index1,index2])
        count = 0
        if control == "Peak shaving":
            usefull_volume = self.output_t_lastyear[self.output_t_lastyear.Outlet_T_hotwell<T_cutoff-1].flow.iloc[0]
            #peak_shaving_control = self.volume/sum(np.clip(missing_energy,a_min=0,a_max=None))*np.clip(missing_energy,a_min=0,a_max=None)
            peak_shaving_control = usefull_volume/sum(np.clip(missing_energy,a_min=0,a_max=None))*np.clip(missing_energy,a_min=0,a_max=None)
            rest = 0
            for i,ele in enumerate(peak_shaving_control):
                if i == len(peak_shaving_control)-1:
                    break
                elif peak_shaving_control[i] > max_flow_generated:
                    rest = peak_shaving_control[i]-max_flow_generated
                    peak_shaving_control[i]=max_flow_generated
                    peak_shaving_control[i+1]=peak_shaving_control[i+1]+rest
                    rest = 0
            if rest != 0:
                peak_shaving_control[0]=peak_shaving_control[0]+rest
        
        # max_diff_flow = np.nanmax(self.output_t_lastyear["flow"].diff())
        max_flow = (self.output_t_lastyear["flow"].iloc[-1])
        # min_index = min(self.output_t_lastyear.index)
        for t in index.index:
            if control == "Peak shaving":
                max_flow_generated=peak_shaving_control[t]
                target_flow = max_flow_generated+flow
                T_after = self.output_t_lastyear.loc[abs((self.output_t_lastyear['flow']-(target_flow))).idxmin()]
                T_after = T_after["Outlet_T_hotwell"]
                
                # Calculate the energy generated
                energy_generated = max_flow_generated*((T_after+T_start)/2-T_cutoff)*self.density*self.heat_cap/3600000 #kWh
                
                #Check which mode the HP is on. We want the smallest temperature difference in the heat pump
                if self.HP != None:
                    if (T_after+T_start)/2<T_cutoff+self.HP.delta_T_coldside:
                        Tsupply = T_demand_out
                        Tsource = (T_after+T_start)/2
                    elif (T_after+T_start)/2>=T_cutoff+self.HP.delta_T_coldside:
                        Tsupply = T_demand_out
                        Tsource = T_cutoff+self.HP.delta_T_coldside
                    self.HP.COP[t] = self.HP.Calculate_COP(Tsupply, Tsource)
                    energy_generated = max_flow_generated*((T_after+T_start)/2-T_cutoff)*self.density*self.heat_cap/3600000 
                    + self.HP.delta_T_coldside*max_flow_generated/(self.HP.COP[t]-1)*self.density*self.heat_cap/3600000 #kWh
                
                #Check if energy is negative and prevent that
                if energy_generated<0:
                    energy_generated=0
                    max_flow_generated = 0
                
                #Check that we are not fully using the heat pump to boost temperature to unusable temperatuer
                if self.HP != None:

                    if T_after < T_cutoff+0.5*self.HP.delta_T_coldside:
                        energy_generated = 0 
                        max_flow_generated = 0
                   
                # Check if enough (and not too much) energy is generated
                if energy_generated <= missing_energy[t]:
                    output[t]= energy_generated
                
                # if too much energy is generated adjust the flow out
                elif energy_generated > missing_energy[t]:
                    factor = missing_energy[t]/energy_generated
                    
                    # Keep iterating, to find the needed flow
                    while energy_generated > missing_energy[t]*1.005 or factor >1.05:
                        max_flow_generated = max_flow_generated *factor
                        T_after = self.output_t_lastyear.loc[abs((self.output_t_lastyear['flow']-(max_flow_generated+flow))).idxmin()]
                        T_after = T_after["Outlet_T_hotwell"]
                        if self.HP != None:
                            energy_generated = max_flow_generated*((T_after+T_start)/2-T_cutoff)*self.density*self.heat_cap/3600000 
                            + self.HP.delta_T_coldside*max_flow_generated/(self.HP.COP[t]-1)*self.density*self.heat_cap/3600000 #kWh
                        else:
                            energy_generated = max_flow_generated*((T_after+T_start)/2-T_cutoff)*self.density*self.heat_cap/3600000 
                        factor = missing_energy[t]/energy_generated

                    #Save the output        
                self.flow_extracted[t]=max_flow_generated
                output[t] = energy_generated
                
                #Set everything for next iteration                   
                T_start = T_after
                flow = flow + max_flow_generated
                max_flow_generated = self.max_V*len_timestep/3600
                
                
            else:
                # Adjust the maximum flow rate if it exceeds the available volume ATES 
                if max_flow_generated + flow > max_flow:
                    max_flow_generated = self.output_t_lastyear["flow"].iloc[-1] - flow
                    if max_flow_generated<0:
                        max_flow_generated=0
                
                
                # Get the temperature after the specified maximum flow this is an approximation
                closest_ind = bisect_left(self.output_t_lastyear["flow"].values, max_flow_generated+flow)
                #closest_ind = get_closests(self.output_t_lastyear, "flow", max_flow_generated+flow)
                
                T_after = self.output_t_lastyear["Outlet_T_hotwell"].iat[closest_ind]
                #T_after = self.output_t_lastyear.iloc[(self.output_t_lastyear['flow']-(max_flow_generated+flow)).abs().argsort()[:1]]
                # T_after = T_after["Outlet_T_hotwell"].iloc[0]
                #exceed_flow = -1
                target_flow = max_flow_generated+flow
                count_diff = closest_ind-count
                count = closest_ind


                
                # Calculate the energy generated
                energy_generated = max_flow_generated*((T_after+T_start)/2-T_cutoff)*self.density*self.heat_cap/3600000 #kWh
                
                #Check which mode the HP is on. We want the smallest temperature difference in the heat pump
                if self.HP != None:
                    if (T_after+T_start)/2<T_cutoff+self.HP.delta_T_coldside:
                        Tsupply = T_demand_out
                        Tsource = (T_after+T_start)/2
                    elif (T_after+T_start)/2>=T_cutoff+self.HP.delta_T_coldside:
                        Tsupply = T_demand_out
                        Tsource = T_cutoff+self.HP.delta_T_coldside
                    self.HP.COP[t] = self.HP.Calculate_COP(Tsupply, Tsource)
                    energy_generated = max_flow_generated*((T_after+T_start)/2-T_cutoff)*self.density*self.heat_cap/3600000 
                    + self.HP.delta_T_coldside*max_flow_generated/(self.HP.COP[t]-1)*self.density*self.heat_cap/3600000 #kWh
                
                #Check if energy is negative and prevent that
                if energy_generated<0:
                    energy_generated=0
                    max_flow_generated = 0
                
                #Check that we are not fully using the heat pump to boost temperature to unusable temperatuer
                if self.HP != None:

                    if T_after < T_cutoff+0.5*self.HP.delta_T_coldside:
                        energy_generated = 0 
                        max_flow_generated = 0
                    # Check if enough (and not too much) energy is generated
                if energy_generated <= missing_energy[t]:
                    output[t]= energy_generated
                
                # if too much energy is generated adjust the flow out
                elif energy_generated > missing_energy[t]:
                    factor = missing_energy[t]/energy_generated
                    
                    # Keep iterating, to find the needed flow
                    while energy_generated > missing_energy[t]*1.005:
                        if factor>1:
                            x = 5
                            break
                        max_flow_generated = max_flow_generated *factor
                        new_loc =count - int(np.round(count_diff*(1-factor)))
                        if new_loc<0:
                            x=0
                        try:
                            T_after = self.output_t_lastyear["Outlet_T_hotwell"].iat[closest_ind]
                        except:
                            x = 5
                        count = new_loc
                        count_diff = count_diff-int(np.round(count_diff*(1-factor)))

                        if self.HP != None:
                            energy_generated = max_flow_generated*((T_after+T_start)/2-T_cutoff)*self.density*self.heat_cap/3600000
                            + self.HP.delta_T_coldside*max_flow_generated/(self.HP.COP[t]-1)*self.density*self.heat_cap/3600000 #kWh #P: Why is the COP inserted here?  delta_T_coldside · flow / (COP−1) · ρcₚ = W (electrical work); in main file the thermal energy is used
                        else:
                            energy_generated = max_flow_generated*((T_after+T_start)/2-T_cutoff)*self.density*self.heat_cap/3600000 
                        factor = missing_energy[t]/energy_generated
                    #Save the output
                self.flow_extracted[t]=max_flow_generated
                output[t] = energy_generated
                
                #Set everything for next iteration                   
                T_start = T_after
                flow = flow + max_flow_generated
                max_flow_generated = self.max_V*len_timestep/3600

                    

        self.flow_extracted=np.clip(self.flow_extracted, a_min=0,a_max=None)   
        # Return output
        #output = np.clip(output, a_min = 0, a_max=None)
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