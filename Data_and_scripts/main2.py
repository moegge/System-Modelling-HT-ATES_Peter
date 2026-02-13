# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:26:20 2023

@author: 6100430
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import interpn
import joblib
import time
import pytest
from ATES_obj import ATES_obj
import math


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")
    
pd.options.mode.chained_assignment = None  # default='warn'

class flow:
    """
    A class that represents the flow of a substance.
    
    Parameters
    ----------
    mass : float
        The mass of the substance in the flow.
    temp : float
        The temperature of the substance in the flow.
    
    Attributes
    ----------
    mass : float
        The mass flow of the substance in the flow [kg/hour].
    temp : float
        The temperature of the substance in the flow [°C].
    heat_capacity : float, optional
        the specific heat capacity of the stream [J/(kg K)] default is 4186 (water)
    density : float, optional
        the density of the stream [kg/m^3]. default is 997 (water)
    dynamic_viscosity : float, optional
        the dynamic viscosity of the stream [Pa*s]. Default is 0.001 (water)    
    """
        
    def __init__(self,mass, temp,heat_capacity = 4186,density = 997,dynamic_viscosity=0.001):
        self.mass=mass
        self.temp = temp
        self.heat_capacity = heat_capacity
        self.density = density
        self.dynamic_viscosity = dynamic_viscosity

class Biomass_boiler:
    def __init__(self, power = 1000, eff = 0.90):  #costperkW = 2500, gas_price = 0.1, opexascapex = 0.05):
        self.name = 'Biomass boiler'
        self.control = 'controlled'
        self.type = 'supply'
        self.rated_power = power #kW
        self.gas_content = 14.97 #kWh/kg
        self.costperkW = 1#costperkW #euro/kW
        self.capex = self.costperkW*self.rated_power
        self.price = 1#gas_price #euro/kWh
        self.opexascapex = 1#opexascapex
        ### Costs still need to be determined
        self.lifetime = 15
        self.eff = eff

    def calc_output(self,needed_output):
        mass_burned = needed_output/self.eff/self.gas_content #kg
        output = np.clip(needed_output,a_min=0,a_max=None)
        return output              
    def calc_opex(self,kWh_generated):
        kWh_in = kWh_generated /self.eff
        var_opex = kWh_in*self.price
        fix_opex = self.capex * self.opexascapex
        opex = var_opex+fix_opex
        return opex

class Solar_collector:
    def __init__(self, peak_power=1000, output_array=None, weather_data=None, T_out = 60,
                 efficiency = 0.75, heat_capacity_fluid = 4186, density_fluid = 997,
                 cost_per_kW = 340, fixed_opex = 4.1, var_opex=0.0019,heat_per_kWp = 600):
        self.control = 'uncontrolled'
        self.type = 'supply'
        self.name = "Solar boiler"
        self.T_out = T_out
        self.heat_cap = heat_capacity_fluid #J/(kg K)
        self.density = density_fluid #kg / m^3
        ##Economics from SDE++ subsidy dutch government
        self.capex = peak_power * cost_per_kW #euro/kwP * kWp
        self.fixed_opex = fixed_opex*peak_power # euro/kWP/year * kWp
        self.var_opex = var_opex #euro/kWh
        self.lifetime = 20
        self.peak_power = peak_power
        self.CO2_kg = 0 #gCO2/kWh or kgCO2/MWh

        heat_generated = peak_power * heat_per_kWp #Peak power in kW
        if output_array is not None:
            sum_output = sum(output_array)
            self.output = heat_generated * (output_array/sum_output)

            # df_weather = pd.read_excel(r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Weather_data.xlsx',header=28)
            # df_weather.drop(df_weather.index[8784:17572],inplace=True)
            # df_weather.drop(df_weather.index[1416:1440],inplace=True)
            # df_weather.to_parquet("weather_data_adapted")
        else:
            df_weather = pd.read_parquet('/eejit/home/6100430/Python_run/Optimization/weather_small_Amsterdam')
            output = np.array(df_weather["Solar irradiance"])#.astype(float)*efficiency)
            sum_output = sum(output)
            self.output = heat_generated * (output/sum_output)
    def calc_output(self,demand):
        return self.output
    def adjust_for_timesetting(self, len_timestep = 3600):
        if len_timestep == 3600:
                pass
        else:
            factor = 3600/len_timestep 
            if factor > 1:
                self.output = np.repeat(self.output, factor)/factor
            if factor < 1:
                self.output = self.output.reshape(int(len(self.output)*factor),-1).sum(1)
    def calc_opex(self,kWh_generated):
        opex = self.fixed_opex + kWh_generated*self.var_opex
        return opex
    def calc_flow(self,T_in):
        Joules = self.output*3600000     
        delta_t = self.T_out- T_in
        volume = Joules/delta_t/self.heat_cap/self.density
        return volume, self.T_out
    def calc_emissions(self,result):
        return 0
              
class demand_class:
    """
    A class representing hourly demand data for a location.
    
    Attributes
    ----------
    timestep : int, optional
        The length of each timestep in seconds. Default is 3600.
    T_in : float, optional
        The temperature of the flow coming [°C]. Default is 40.
    T_out : float, optional
        The temperature of the return flow [°C]. Default is 35.
    
    Notes:
    -----
    This is pre-initialized with a load from Harbin, but any txt document with
    hourly data will work.     
    """
    def __init__(self,T_in=40,T_out=35,demand_array = None, heat_capacity_fluid = 4186, density_fluid = 997, example_demand="Den Haag"):
        #Assumes hourly data
        #timestep is in seconds
        self.T_in = T_in
        self.T_out= T_out
        self.len_timestep=3600
        self.type = 'demand'
        self.heat_cap = heat_capacity_fluid #J/(kg K)
        self.density = density_fluid #kg / m^3
        if type(demand_array) == type(None):
            if example_demand == "Harbin":
                path = r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Harbinload.txt'
                text_file = open(path, "r")
                lines = text_file.readlines()
                for i in range(round(len(lines))):
                    lines[i]=float(lines[round(i)])
                self.data= np.array(lines)/3600 #kWh needed for that hour.
                saved = self.data[1:4344]
                self.data=self.data[4344:]
                self.data = np.append(self.data, saved)
            elif example_demand == "Utrecht":
                #Does not work yet
                path = r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Warmtevraag 2017_Utrecht_edit.xlsx'
                excel_file = pd.read_excel(path,"Warmtevraag")

                excel_file = excel_file * 1000
                excel_file.interpolate(inplace=True)
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Den Haag":
                path = r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Warmtevraag 2016_DenHaag.xlsx'
                excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "TU Delft":
                path = '/eejit/home/6100430/Python_run/Optimization/Warmtevraag_Delft_parquet'
                excel_file = pd.read_parquet(path)
                #excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file.drop(["Demand Total","Demand OWD"],inplace=True,axis=1)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Delft City":
                path = '/eejit/home/6100430/Python_run/Optimization/Warmtevraag_Delft_parquet'
                excel_file = pd.read_parquet(path)
                #excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file.drop(["Demand Total","Demand TUD"],inplace=True,axis=1)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]        
            elif example_demand == "Delft Total":
                path = '/eejit/home/6100430/Python_run/Optimization/Warmtevraag_Delft_parquet'
                excel_file = pd.read_parquet(path)
                #excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file.drop(["Demand OWD","Demand TUD"],inplace=True,axis=1)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]  
            elif example_demand == "Amsterdam":
                path = '/eejit/home/6100430/Python_run/Optimization/Warmtevraag_Amsterdam_50GWh_parquet'
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]  
            elif example_demand == "Tianjin":
                path = '/eejit/home/6100430/Python_run/Optimization/Warmtevraag_Tianjin_50GWh_parquet'
                #excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]  
            elif example_demand == "Athens":
                path = '/eejit/home/6100430/Python_run/Optimization/Warmtevraag_Athens_50GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Bagdad":
                path = '/eejit/home/6100430/Python_run/Optimization/Warmtevraag_Bagdad_50GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Berlin":
                path = '/eejit/home/6100430/Python_run/Optimization/Warmtevraag_Berlin_50GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Chongqing":
                path = '/eejit/home/6100430/Python_run/Optimization/Warmtevraag_Chongqin_50GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand =="Lhasa":
                path = '/eejit/home/6100430/Python_run/Optimization/Parquet_demand/Warmtevraag_Lhasa_50_GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand =="Edinburgh":
                path = '/eejit/home/6100430/Python_run/Optimization/Parquet_demand/Warmtevraag_Edinburgh_50_GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand =="Osaka":
                path = '/eejit/home/6100430/Python_run/Optimization/Parquet_demand/Warmtevraag_Osaka_50_GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand =="Tokyo":
                path = '/eejit/home/6100430/Python_run/Optimization/Parquet_demand/Warmtevraag_Tokyo_50_GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]             
            elif example_demand =="Los Angeles":
                path = '/eejit/home/6100430/Python_run/Optimization/Parquet_demand/Warmtevraag_Los Angeles_50_GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
                
            else:
                print("No valid demand_array given, options are 'Utrecht', 'Den Haag','Harbin', 'TU Delft, 'Delft City','Delft Total', taking Den Haag")
                path = r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Warmtevraag 2016_DenHaag.xlsx'
                excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
                
                
                

        else:
            if len(demand_array) != 8760 :
                raise ValueError("Please provide demand data based on hourly values for a year. Length demand_array should be 8760")
            self.data = demand_array #Please provide this in kWh for every hour.
        #self.hourly_data = self.data
    def plot(self):
        #plt.figure(dpi = 2000)
        plt.plot(np.linspace(0,8760,len(self.data)),self.data/1000/8760*len(self.data))
        plt.xlabel('Time (hours)')
        plt.ylabel("Demand (MW)")
        plt.xlim([0,8760])
        #tikzplotlib.save("demand_data.tex")
    def plot_different_timesetting(self):
        plt.figure()
        plt.plot(self.data)
        plt.xlabel('Timestep number')
        plt.ylabel('Demand each timestep (kWh)')
        plt.xlim([0,len(self.data)])
        plt.show()
    def adjust_for_timesetting(self, len_timestep = 3600):
        self.len_timestep=len_timestep
        if len_timestep == 3600:
                pass
        else:
            factor = 3600/len_timestep 
            if factor > 1:
                self.data = np.repeat(self.data, factor)/factor
            if factor < 1:
                new = [self.data[i*int(len(self.data)/(len(self.data)*factor)):(i+1)*int(len(self.data)/(len(self.data)*factor))] for i in range(0, int(len(self.data)*factor), 1)]
                new = np.sum(new,axis=1)
                # missing = sum(self.data)-sum(new)
                # new = np.append(new,missing)
                self.data=new
                #self.data = self.data.reshape(int(len(self.data)*factor),-1).sum(1)
    def calc_flow(self):
        joules_needed = self.data *3600000 #convert to Joules
        delta_t = self.T_in - self.T_out
        mass_water = joules_needed/delta_t/self.heat_cap
        volume = mass_water/self.density
        return volume , self.T_out

class heat_pump_ATES:
    def __init__(self, power_th = None,delta_T_coldside=None, costperkW = 200,
                 fixed_opex=60, elec_price = 0.2, lifetime = 15):
        self.control = 'controlled'
        self.type = 'supply'
        self.name = "Heat pump"
        if power_th == None and delta_T_coldside ==None:
            ValueError("give either power or delta_T")
        if power_th !=None and delta_T_coldside!=None:
            print("both power and delta_T_coldside given, using delta_T")
            power=None
            
        self.power_th = power_th #kWth
        self.delta_T_coldside=delta_T_coldside
        self.delta_T_coldside = delta_T_coldside
        self.message_printed=False
        self.capex = costperkW #euro/kWth  
        #https://energy.nl/wp-content/uploads/industrial-high-temperature-heat-pump-2-7.pdf
        self.fixed_opex = fixed_opex #euro/kWth/yr
        self.elec_price = elec_price #electricity price euro/kWh. Electricity price relatively high
        self.lifetime = lifetime
    def init(self,ATES):
        if ATES.name !='ATES':
            ValueError("wrong storage type connected.")
        if self.delta_T_coldside == None:
            self.delta_T_coldside = self.power_th/ATES.max_V/4186/1000*3600000
    def Calculate_COP(self,Tsupply, Tsource):
       if Tsource>Tsupply:
           print("HP configuration incorrect. Temperature source should be lower than heat sink temperature")
       fc = 0.35 + 0.6/200 * (Tsupply - Tsource)
       COP = fc * (Tsupply + 273)/(Tsupply -Tsource)
       #https://energy.nl/wp-content/uploads/industrial-high-temperature-heat-pump-2-7.pdf
       if COP>5:
           COP=5
           if self.message_printed != True:                        
               #print("COP higher than 5, seems unlikely, check your inputs. Set to 5")
               self.message_printed=True
       return COP 
    def calc_output(self,needed_output):
        return 0
    def calc_emissions(self,result):
        return 0
    def calc_opex(self,kWh_generated):
        try:
                
            self.capex=self.capex*self.rated_power
            fixed_opex = self.fixed_opex*self.rated_power
            var_opex = self.elec_input*self.elec_price
            var_opex[np.isnan(var_opex)] = 0
    
            opex = sum(var_opex)+fixed_opex
        except:
            self.capex = 0
            opex = 0
        return opex
        
     
class geothermal:
    def __init__(self, flow_rate = None,power = None,costperkW = 1909, fixed_opex = 69,
                 var_opex = 0.0072, T_out = 90, depth = 2000,N_wells=2,lifetime = 30,
                 heat_capacity_fluid = 4186, density_fluid = 997, CO2_kg=27):
        self.name = 'Geothermal well'
        self.control = 'stable'
        self.type = 'supply'
        if flow_rate!=None:
            if power!= None:
                #print("both flow_rate and power given, only one is needed, taking flow rate")
                self.flow_rate=flow_rate #m^3/hour
                self.power = None
            else:
                self.flow_rate=flow_rate #m^3/hour
                self.power= None
        else:
            if power == None:
                ValueError("Both power and flow_rate are not given, please provide one of these values")
            else:
                self.power = power
                self.flow_rate=None
        #self.rated_power = power #hourly kW output
        
        #self.capex = (375000 +1150*depth+0.3*depth**2)*N_wells #https://www.thermogis.nl/en/economic-model
        #self.capex = self.power*self.costperkW #euro --> kw * (euro/kW)
        self.costperkW = costperkW
        self.fixed_opex = fixed_opex #euro/kW/year
        self.var_opex = var_opex #euro/kWh
        self.T_out = T_out
        self.heat_cap = heat_capacity_fluid #J/(kg K)
        self.density= density_fluid #kg / m^3
        #### economics from MSc_Thesis_Report_ToonvdGriendt
        self.CO2_kg = CO2_kg
        self.lifetime = lifetime
    def calc_output(self,len_timestep,demand):
        if self.power == None:
            self.power = self.flow_rate*(self.T_out-demand.T_out)*10**-7*4186*1000*2.77777
            if self.power <=0:
                pass
                #print("geothermal well does not produce any heat compared to district heating network, check inputs")
        self.capex = self.power*self.costperkW #euro --> kw * (euro/kW)
        output = self.power*len_timestep/3600 #kWh
        self.output = output
        return output
    def calc_opex(self, produced_kWh):
        fixed_opex = self.fixed_opex*self.power
        var_opex = produced_kWh*self.var_opex
        opex = var_opex+fixed_opex
        return opex
    def calc_flow(self,T_in):
        Joules = self.output*3600000     
        delta_t = self.T_out- T_in
        volume = Joules/delta_t/self.heat_cap/self.density #m^3
        return volume, self.T_out
    def calc_emissions(self,result):
        generated = sum(result["Geothermal well corrected"])
        try:
            to_storage = sum(result["Geothermal well percentage to storage"]*result["Geothermal well production"])
        except:
            to_storage=0
        if not np.isnan(to_storage):
            generated = generated-to_storage
        else:
            generated = generated = generated
        return generated*self.CO2_kg
        #return sum(result["Geothermal well corrected"])*self.CO2_kg #g_CO2
        
class gas_boiler:
    def __init__(self, power = 1000, eff = 0.93,costperkW = 100, gas_price = 0.1, 
                 opexascapex = 0.02,lifetime = 15,CO2_kg = 200):
        self.name = 'Gas boiler'
        self.control = 'controlled'
        self.type = 'supply'
        self.rated_power = power #kW Source economics: MSc_Thesis_Report_ToonvdGriendt
        self.gas_content = 14.97 #kWh/kg
        self.costperkW = costperkW #euro/kW
        self.capex = self.costperkW*self.rated_power
        self.price = gas_price #euro/kWh
        self.opexascapex = opexascapex
        self.lifetime = lifetime
        self.eff = eff
        self.CO2_kg = CO2_kg #kg CO2/MWh or gCO2/kWh
        
    def calc_output(self,needed_output):
            mass_burned = needed_output/self.eff/self.gas_content #kg
            self.gas_burned = mass_burned
            output = np.clip(needed_output,a_min=0,a_max=None)
#            print(sum(np.clip(mass_burned,a_min=0,a_max=None)))
            return output              
    def calc_opex(self,kWh_generated):
        kWh_in = kWh_generated /self.eff
        var_opex = kWh_in*self.price
        fix_opex = self.capex * self.opexascapex
        opex = var_opex+fix_opex
        return opex
    def calc_emissions(self,result):
        return sum(result["Gas boiler corrected"])*self.CO2_kg #Grams
def system(demand, supply, len_timestep = 3600, time_horizon=8760,control = None):
    """
    Simulates the interaction between demand and supply components in an district heating system.
    
    Parameters
    ----------
    demand : Demand_obj
        An object representing the energy demand component of the system.
    supply : list of Supply_obj
        A list of objects representing the energy supply components of the system.
    len_timestep : int, optional
        Length of each simulation time step in seconds (default is 3600).
    time_horizon : int, optional
        Total time horizon for the simulation in hours (default is 8760).
        Errors if it is not 8760. Not yet implemented
    
    Returns
    -------
    tuple
        A tuple containing two pandas DataFrames:
        1. `result`: A DataFrame with columns representing the demand, total production,
           and production of each supply component at each time step.
        2. `df_flow`: A DataFrame with columns representing the time, demand volume, and
           flow information for each supply component.
    
    Raises
    ------
    ValueError
        If the type of demand or supply is incorrect.
        If there are multiple storage components provided.
    """

    # Check if the demand type is correct
    if demand.type != 'demand':
        raise ValueError("Demand type is wrong")

    # Adjust the demand for the simulation time settings    
    demand.adjust_for_timesetting(len_timestep=len_timestep) 
    
    #Initialize storage check
    check_stor = 0
    Storage = False

    # Check the type of each supply component and raise errors if wrong
    for i in supply:
        if i.type != 'supply':
            raise ValueError("Supply type is wrong") 
        if i.control == 'uncontrolled':
            i.adjust_for_timesetting(len_timestep=len_timestep)
        if i.control == 'storage':
            # Check if there is only one storage component and copy it. Currenlty it can only work with one component
            if check_stor == 1:
                raise ValueError("Too many storage components supplied, can only work with one")
            Storage = True
            storage_obj = i
            check_stor = 1
            
    # Generate a time series and DataFrame for the simulation to store data
    time_series = np.linspace(0+time_horizon/len(demand.data),time_horizon,len(demand.data))
    result = pd.DataFrame({'Demand':demand.data,'Time (hours)':time_series, 'Total production':0.0 })
    for i in supply:
    
        # Simulate the production for each supply component if it is an uncontrollable source or stable source. Otherwise initialize to 0
        if i.control == 'uncontrolled':
            result[i.name + " production"]=i.output
            result.loc[:,"Total production"] = result.loc[:,'Total production']+ result.loc[:,i.name + " production"]
        elif i.control == 'stable':
            result[i.name + " production"]=i.calc_output(len_timestep,demand)
            result.loc[:,"Total production"] = result.loc[:,'Total production']+ result.loc[:,i.name + " production"]
        else:
            result[i.name + " production"] = 0



    #%% Simulate the flow of energy, only if storage is implemented
    # Initialize a DataFrame to save the flows in the system  

    df_flow = pd.DataFrame({'Time (hours)':time_series})

    if Storage:
        # Initialize the flow of demand
        df_flow["Demand volume"], T = demand.calc_flow()
        
        # Initialize storage extraction and injection, so the ATES will not 
        # Eradically charge and discharge
        # yhat = savgol_filter(demand.data , int(2000/len_timestep*3600), 2)
        # storage_extraction = yhat>result["Total production"]
        # storage_injection = yhat<result["Total production"]
        # Currently this is unused, it is a legacy aspect
        storage_extraction = np.ones(len(result["Total production"]))
        storage_injection = np.ones(len(result["Total production"]))
        
        # Initialize the flows of the uncontrollable sources
        for i in supply:
            if i.control == 'stable' or i.control == 'uncontrolled':
                df_flow[i.name + " Volume out"],df_flow[i.name + " T out"]= i.calc_flow(demand.T_out)
                #Correct the volume for the required input temperature of the DH
                df_flow[i.name + " Corrected volume"] = df_flow[i.name + " Volume out"] * (df_flow[i.name + " T out"]-demand.T_out)/(demand.T_in-demand.T_out)
        
        # Save demand flow in variable to be manipulated
        flow_not_covered = df_flow["Demand volume"]
        
        # Check if there is a uncontrollable/stable sources not connected to storage and adjust uncovered flow to demand.
        for i in supply:
            if i.control == "controlled" or i.control == 'storage':
                continue
            for j in storage_obj.supplier: 
                if i.name == j.name:
                    break_loop = 1
            if break_loop == 1:
                break_loop=0
                continue
            flow_not_covered = flow_not_covered - df_flow[i.name + " Corrected volume"]
        
        # Uncovered flow cannot be negative, clip to 0
        flow_not_covered = np.clip(flow_not_covered,a_min = 0, a_max=None)
        
        # Initialize
        flow_av_demand = 0
        
        # Calculate the flow to demand of the sources connected to storage
        for i in storage_obj.supplier: 
            flow_av_demand = flow_av_demand + df_flow[i.name+ " Corrected volume"]
              
        # Calculate the percentage of flow going to demand             
        percentage_used = flow_not_covered / flow_av_demand
        percentage_used = np.clip(percentage_used,0,1)
        percentage_used[np.isnan(percentage_used)] = 0

        # Initialize for later
        df_flow["Total flow to storage"] = 0
        T_total = 0
        
        # For each supply connected to storage, check how much volume can go to the storage
        for i in storage_obj.supplier: 
            #Check if there is a HP and initalize it to see what the role of it is.
            if storage_obj.HP != None:
                storage_obj.HP.init(storage_obj)
                #Calculate the reduction of volume that can go the to ATES due to a larger temperature difference generated by the HP.
                #This is based on the assumption that a lower temperature is injected into the cold side and this is then needs to be heated up further, reducing total volume
                Factor_due_HP = (i.T_out-(demand.T_out-storage_obj.HP.delta_T_coldside))/(i.T_out-demand.T_out)
            else:
                Factor_due_HP=1
                
            
            # Check if we are injecting, calculate the flows to the storage
            result[i.name + " percentage to storage"] = (1-percentage_used)*storage_injection
            df_flow[i.name + " flow to storage"] = (1-percentage_used)*df_flow[i.name+ " Volume out"]*storage_injection/Factor_due_HP
            T_total =  T_total+sum(df_flow[i.name + " flow to storage"]) * i.T_out
            df_flow["Total flow to storage"] =  df_flow["Total flow to storage"] + df_flow[i.name + " flow to storage"]
    
        # Calculate the temperature to the storage
        if sum(df_flow["Total flow to storage"])>0:
             T_av = T_total/sum(df_flow["Total flow to storage"])
        else:
            T_av=0
       
        volume  = sum(df_flow["Total flow to storage"])
        
        #ATES system has internal HX which reduces inlet temperatures
        T_ineff_due_HX = 0#(1-storage_obj.HX_eta)* (T_av-demand.T_out)
        
        loss_percentage = 1
        #Calculate losses in cold well, which needs to be compensated for. 
        for j in storage_obj.supplier:
            if j.name != 'Geothermal well':
                k=0
                #iterative process to find the losses in the cold well
                while k<3:
                    #Set up flow
                    dummy_flow = df_flow["Total flow to storage"]*(loss_percentage)
                    #Calculate percentage of flow going to storage initially
                    #Check if everything can be injected into storage, if not reduce the inflow.
                    percentage=(storage_obj.max_V*len_timestep/3600)/dummy_flow
                    percentage[percentage==np.inf]=0
                    percentage= np.nan_to_num(percentage, nan=0)
                    percentage = np.clip(percentage,a_min=None,a_max=1)
                    dummy_flow=dummy_flow*percentage
                    volume  = sum(dummy_flow)
                    Heat_to_storage = sum(dummy_flow)*(T_av-demand.T_out)
                    
                    #Initialize cold well and calculate Reff of that well
                    if storage_obj.HP != None:
                        storage_obj.init_cold_well(demand.T_out-storage_obj.HP.delta_T_coldside,volume)
                    else:
                        storage_obj.init_cold_well(demand.T_out+T_ineff_due_HX,volume)
                        
                    #Calculate extra volume required due to heat losses
                    volume = Heat_to_storage/(T_av-storage_obj.cold_well_T_ave)
                    k = k+1
                    if sum(dummy_flow) ==0:
                        loss_percentage = 1
                    else:
                        loss_percentage = 1-(volume/sum(dummy_flow))
            #If we are working with a geothermal well, the returrn temperature should be as low as possible
            else:
                percentage=(storage_obj.max_V*len_timestep/3600)/df_flow["Total flow to storage"]
                percentage[percentage==np.inf]=0
                percentage = np.clip(percentage,a_min=None,a_max=1)

                            
        #Save total flow to storage
        df_flow["Total flow to storage"]=df_flow["Total flow to storage"]*percentage
        storage_obj.flow_injected = np.array( df_flow["Total flow to storage"])

        for i in storage_obj.supplier: 
            result[i.name + " percentage to storage"] = result[i.name + " percentage to storage"]*percentage
            df_flow[i.name + " flow to storage"] = df_flow[i.name + " flow to storage"]*percentage
       
        # Run the storage simulation
        if sum(df_flow["Total flow to storage"])>1:
            # Initialize storage
            storage_obj.initialize(sum(df_flow["Total flow to storage"]),T_av-T_ineff_due_HX, len_timestep)

            # Calculate energy that can be covered by the storage and what the output of storage is
            missing_energy = result['Demand']-result['Total production']
            output_storage = storage_obj.calc_heat(demand.T_out,demand.T_in,storage_extraction,missing_energy,len_timestep=len_timestep,control = control)    

            # Calculate the contribution of the heat pump and the required power of the HP
            if storage_obj.HP != None:
                output_HP = storage_obj.HP.delta_T_coldside*storage_obj.flow_extracted*1000*4186/3600000
                result["Heat pump production"]=output_HP
                storage_obj.HP.rated_power=max(output_HP)/(len_timestep/3600) #kWh
                storage_obj.HP.COP[storage_obj.HP.COP == 0] = np.nan 
                storage_obj.HP.elec_input = output_HP/storage_obj.HP.COP
            else:
                output_HP = 0
                result["Heat pump production"]=output_HP
            
            # Save the calculations.
            result[storage_obj.name+" production"] = output_storage-output_HP
            result['Total production'] = result['Total production']+ result[storage_obj.name+" production"]+result["Heat pump production"]
        
        # If not flow to storage, it is deemed useless    
        else:
            #print("Storage is neglected, no volume available for storage")
            result[storage_obj.name+" production"] = 0
    # Calculate the missing energy after the storage
    missing_energy = result['Demand']-result['Total production']
    
    # Fill the missing energy with the controllable sources
    for i in supply:
        if i.control == 'controlled':
            if i.name !="Heat pump":
                output = i.calc_output(missing_energy)
                result[i.name + " production"] = output
                i.rated_power=max(result[i.name + " production"])/(len_timestep/3600)
                i.capex=i.rated_power*i.costperkW
            else:
                pass

    ## correct for not everything used, initialize here
    i_list = []
    save_value = 0
    to_storage = 0

    # Store everything corrected after the simulation
    for i in supply:
        i_list.append(i)
        value = 0
        for j in range(len(i_list)):
            value += result[i_list[j].name + " production"] 
        if Storage:
            
            for j in storage_obj.supplier:
                if i.name == j.name:
                    to_storage = to_storage + result[i.name + " percentage to storage"]*result[i.name+ " production"]

        value[value > result["Demand"]+to_storage]=result["Demand"]+to_storage
        result[i.name + " corrected"]= value-save_value
        save_value = value#/(len_timestep/3600)
        

    return result, df_flow

def LCOE_calc_Yang(result,supply,df_eco,disc_rate=0.05,lifetime_system = 60,capex_network=0,opex_network_perc = 0,lifetime_network=60):
    """
    Calculates the Levelized Cost of Energy (LCOE) for each supply component based on economic parameters.
    
    Parameters
    ----------
    result : pd.DataFrame
        DataFrame containing the simulation results, including corrected production values for each supply component.
    supply : list
        List of supply components in the energy system.
    df_eco : pd.DataFrame
        DataFrame containing economic parameters such as opex, capex, etc.
    disc_rate: float
        The discount rate to be used.
    
    Returns
    -------
    pd.DataFrame
        DataFrame df_eco updated with the calculated LCOE values for each supply component.
    
    Notes
    -----
    The LCOE is calculated using the formula:
    LCOE = (Sum of present value of costs) / (Sum of present value of generated energy)
    
    The calculation considers the discount rate, lifetime, and economic parameters for each supply component.
    
    Lifetime is assumed to be in years.
    
    The result DataFrame should contain columns representing the corrected production for each supply component.
    """  
    # Initialize LCOE column in the economic DataFrame
    df_eco["LCOE_System"]=np.nan

    # Loop over supply technologies
    generated = 0
    sum_cost=0
    for i in supply:
        # If name is ATES, do special calculations
        if i.name == "ATES":
            
            # Initialize para to 0

            real_extracted = sum(result["ATES corrected"])
            if real_extracted == 0:
                continue
            max_extracted = i.total_heat_extracted_vs_T_ground_kWh_first_8_years[-1]
            percentage = real_extracted/max_extracted
            opex = df_eco.at[i.name,"opex"]

            # Calculate costs and generated energy for each year in the lifetime
            for j in range(lifetime_system):

                if j%i.lifetime == 0:
                    # Initial year includes both capex and opex
                    opex_fy = (opex + df_eco["capex"].loc[i.name])/ (1 + disc_rate) ** j
                    sum_cost = sum_cost + opex_fy
                    generated = generated + i.total_heat_extracted_vs_T_ground_kWh_first_8_years[j%i.lifetime]*percentage / (1 + disc_rate) ** j
                elif j%i.lifetime < 8:
                    # Accumulate generated energy for the first 8 years
                    generated = generated + i.total_heat_extracted_vs_T_ground_kWh_first_8_years[j%i.lifetime]*percentage / (1 + disc_rate) ** j
                    sum_cost = sum_cost+opex/ (1 + disc_rate) ** j
                else:
                    # Calculate costs and generated energy for subsequent years
                    sum_cost = sum_cost + opex / (1 + disc_rate) ** j
                    generated = generated + i.total_heat_extracted_vs_T_ground_kWh_first_8_years[-1]*percentage / (1 + disc_rate) ** j

                    # Calculate LCOE for the supply component
                    LCOE = sum_cost / generated
                    # Update LCOE in the economic DataFrame



        # Calculate LCOE of each technology        
        else:                
            try: 
                Total_stored = sum(result[i.name+ " percentage to storage"]*result[i.name + " production"])
            except:
                Total_stored=0
            # Calculate costs and generated energy for each year in the lifetime
            Total_gen =  sum(result[i.name + " corrected"])
            opex = df_eco["opex"].loc[i.name]

            for j in range(lifetime_system):
                year = j
                if year%i.lifetime == 0:
                    # Initial year includes both capex and opex
                    opex_fy = (opex + df_eco["capex"].loc[i.name])/ (1 + disc_rate) ** j
                    sum_cost = sum_cost + opex_fy
                    # Sum corrected production for the first year
                    generated = generated +Total_gen/ (1 + disc_rate) ** j
                    try:
                        to_storage = Total_stored/ (1 + disc_rate) ** j
                        if not np.isnan(to_storage):
                            generated = generated-to_storage
                    except:
                        pass
                else:
                    # Calculate costs and generated energy for subsequent years
                    sum_cost = sum_cost + opex / (1 + disc_rate) ** j
                    generated = generated +Total_gen / (1 + disc_rate) ** j
                    try:
                        to_storage = (Total_stored)/ (1 + disc_rate) ** j
                        if not np.isnan(to_storage):
                            generated = generated-to_storage
                    except:
                        pass
                    # Calculate LCOE for the supply component
                    if sum_cost != 0 and generated !=0:
                        LCOE = sum_cost / generated
                    else:
                        LCOE = np.nan
                    # Update LCOE in the economic DataFrame
                    
    #Add system costs here
     
    for year in range(lifetime_system):
        if year%lifetime_network==0:
            opex_fy = opex_network_perc*capex_network+capex_network
            sum_cost = sum_cost + opex_fy
        else:
            sum_cost = sum_cost+opex_network_perc*capex_network
    LCOE = sum_cost / generated

    #sum_cost= sum_cost+piping_cost*piping_length
    df_eco["LCOE_System"]= LCOE
    LCOH_system = sum_cost/generated
    return LCOH_system

def LCOE_calc(result, supply, df_eco,disc_rate=0.05):
    """
    Calculates the Levelized Cost of Energy (LCOE) for each supply component based on economic parameters.
    
    Parameters
    ----------
    result : pd.DataFrame
        DataFrame containing the simulation results, including corrected production values for each supply component.
    supply : list
        List of supply components in the energy system.
    df_eco : pd.DataFrame
        DataFrame containing economic parameters such as opex, capex, etc.
    disc_rate: float
        The discount rate to be used.
    
    Returns
    -------
    pd.DataFrame
        DataFrame df_eco updated with the calculated LCOE values for each supply component.
    
    Notes
    -----
    The LCOE is calculated using the formula:
    LCOE = (Sum of present value of costs) / (Sum of present value of generated energy)
    
    The calculation considers the discount rate, lifetime, and economic parameters for each supply component.
    
    Lifetime is assumed to be in years.
    
    The result DataFrame should contain columns representing the corrected production for each supply component.
    """  
    # Initialize LCOE column in the economic DataFrame
    df_eco["LCOE"]=np.nan
    add_opex_ATES=0
    # Loop over supply technologies
    for i in supply:
        # If name is ATES, do special calculations
        if i.name == "ATES":
            
            # Initialize para to 0
            generated = 0
            sum_cost=0
            real_extracted = sum(result["ATES corrected"])
            if real_extracted == 0:
                continue
            max_extracted = i.total_heat_extracted_vs_T_ground_kWh_first_8_years[-1]
            percentage = real_extracted/max_extracted
            opex = df_eco.at[i.name,"opex"]

            # Calculate costs and generated energy for each year in the lifetime
            for j in range(i.lifetime):

                if j == 0:
                    # Initial year includes both capex and opex
                    sum_cost = sum_cost + opex +add_opex_ATES+ df_eco["capex"].loc[i.name]
                    generated = generated + i.total_heat_extracted_vs_T_ground_kWh_first_8_years[j]*percentage / (1 + disc_rate) ** j
                elif j < 8:
                    # Accumulate generated energy for the first 8 years
                    generated = generated + i.total_heat_extracted_vs_T_ground_kWh_first_8_years[j]*percentage / (1 + disc_rate) ** j
                    sum_cost = sum_cost+(opex+add_opex_ATES)/ (1 + disc_rate) ** j
                else:
                    # Calculate costs and generated energy for subsequent years
                    sum_cost = sum_cost + (opex+add_opex_ATES) / (1 + disc_rate) ** j
                    generated = generated + i.total_heat_extracted_vs_T_ground_kWh_first_8_years[-1]*percentage / (1 + disc_rate) ** j

                    # Calculate LCOE for the supply component
            LCOE = sum_cost / generated
            # Update LCOE in the economic DataFrame
            df_eco.at[i.name,"LCOE"] = LCOE
            df_eco.at[i.name,"generated discounted"] = generated

        # Calculate LCOE of each technology        
        else:                
            try: 
                Total_stored = sum(result[i.name+ " percentage to storage"]*result[i.name + " production"])
                percentage_stored = Total_stored/sum(result[i.name + " production"])

                if np.isnan(Total_stored):
                    Total_stored=0
                    percentage_stored=0

            except:
                Total_stored=0                
                percentage_stored=0
            sum_cost = 0
            generated = 0

            # Calculate costs and generated energy for each year in the lifetime
            Tot_prod = sum(result[i.name + " corrected"])-Total_stored
            if Total_stored!=0:
                opex = df_eco.at[i.name,"opex"]- i.var_opex*Total_stored
                add_opex_ATES = add_opex_ATES+ i.var_opex*Total_stored
            else:
                opex = df_eco.at[i.name,"opex"]
            for j in range(i.lifetime):
                year = j
                if year == 0:
                    # Initial year includes both capex and opex
                    sum_cost = sum_cost + opex + df_eco["capex"].loc[i.name]
                    # Sum corrected production for the first year
                    generated = Tot_prod
                else:
                    # Calculate costs and generated energy for subsequent years
                    
                    sum_cost = sum_cost + opex / (1 + disc_rate) ** j
                    generated = generated + Tot_prod / (1 + disc_rate) ** j

            # Calculate LCOE for the supply component
            if sum_cost != 0 and generated !=0:
                LCOE = sum_cost / generated
            else:
                LCOE = np.nan
                    # Update LCOE in the economic DataFrame
            df_eco.at[i.name,"LCOE"] = LCOE
            df_eco.at[i.name,"generated discounted"] = generated
    return df_eco

def economic_analysis(results_system, supply,disc_rate = 0.05,incorporate_CO2=False,CO2_price = 70,opex_ATES_fixed=False):
    """
    Performs economic analysis to calculate operational and capital expenses, as well as the Levelized Cost of Energy (LCOE)
    for each supply component in the energy system.

    Parameters
    ----------
    results_system : pd.DataFrame
        DataFrame containing the simulation results, including corrected production values for each supply component.
    supply : list
        List of supply components in the energy system.

    Returns
    -------
    pd.DataFrame
        DataFrame containing economic analysis results, including name, capex, opex, and LCOE for each supply component.

    Notes
    -----
    The function creates an economic DataFrame with columns for supply component name, capital expenses (capex),
    operational expenses (opex), and LCOE. The LCOE is calculated using the LCOE_calc function.
    """
    index = []
    for i in supply:
        index.append(i.name)

    # Initialize economic DataFrame
    df_eco = pd.DataFrame(index=index, columns=["name", "capex", "opex", "generated discounted"])

    # Populate economic DataFrame with supply component information
    if incorporate_CO2:
        CO2_df = CO2_emissions_calc(results_system, supply, CO2_price=CO2_price)
    for count, i in enumerate(supply):
        if i.name == "ATES":
            if opex_ATES_fixed:
                try:
                    df_eco.loc[i.name,"opex"]=i.fix_opex+(i.volume+sum(i.flow_extracted))/2/1000000*1389*1000*i.elec_price
                except:
                    df_eco.loc[i.name,"opex"] = 0
            
            else:
                df_eco.loc[i.name,"opex"]= i.calc_opex(sum(results_system[i.name + " corrected"]))
        else:
            #df_eco["opex"].iloc[count] = i.calc_opex(sum(results_system[i.name + " corrected"]))
            df_eco.loc[i.name,"opex"]= i.calc_opex(sum(results_system[i.name + " corrected"]))
        df_eco.at[i.name,"name"] = i.name
        df_eco.at[i.name,"capex"] = i.capex
        if incorporate_CO2:
            df_eco.at[i.name,"opex"] =df_eco.at[i.name, "opex"] + CO2_df.at[i.name,'Cost_CO2']

    # Calculate LCOE for each supply component
    df_eco = LCOE_calc(results_system, supply, df_eco,disc_rate=disc_rate)

    return df_eco


def CO2_emissions_calc(result,supply,CO2_price = 70):
    #Price CO2 in euro/ton
    
    index = []
    for i in supply:
        index.append(i.name)
    df_CO2 = pd.DataFrame(index=index, columns=["name", "CO2_emission [kg]"])
    
    for i in supply:
        df_CO2.at[i.name,"name"] = i.name
        df_CO2.at[i.name,"CO2_emission [kg]"] = i.calc_emissions(result)/1000
    df_CO2["Cost_CO2"] = df_CO2["CO2_emission [kg]"]/1000*CO2_price
    return df_CO2

def system_plot(result, supply, demand, len_timestep = 3600,setting = "everything"):
    fig,ax = plt.subplots()
    new_supply = []
    back_up_supply = []
    Storage = False
    for i in supply:
        if i.control == 'storage':
            Storage = True
            storage_obj = i

        
    for i in supply:
        if i.control == "stable":
            if Storage:
                    
                for j in storage_obj.supplier:
                    if i.name == j.name:
                        back_up_supply.append(i)
            else:
                new_supply.append(i)
                    
    for i in supply:
        if i.control == "uncontrolled":
            if Storage:
                for j in storage_obj.supplier:
                    if i.name == j.name:
                        back_up_supply.append(i)
                    else:
                        new_supply.append(i)
    for i in back_up_supply:
        new_supply.append(i)        
    for i in supply:
        if i.control == "storage":
            new_supply.append(i)
    for i in supply:
        include_it = True
        for j in new_supply:
            if i.name == j.name:
                include_it = False
        if include_it:
            new_supply.append(i)
    supply = new_supply
    #plt.plot(result["Time (hours)"],result["Demand"],label = 'demand')
    #ax.fill_between(result["Time (hours)"],result["Demand"],0)
    #plt.plot(result["Time (hours)"],result["Total production"], label = "Total production")

  
    if setting == "everything":
        i_list = []
        save_value = np.zeros(len(result))
        for i in supply:
            i_list.append(i)
            value = 0
            for j in range(len(i_list)):
                value += result[i_list[j].name + " production"] /(len_timestep/3600)
            #plt.plot(result["Time (hours)"],value,label = i.name,visible=False)
            ax.fill_between(result["Time (hours)"],value, save_value,label=i.name)
            save_value = value
        plt.plot(result["Time (hours)"],result["Demand"]/(len_timestep/3600),label = 'demand', color = 'k',linewidth = 0.5)
        plt.legend()
        plt.xlim([0,max(result["Time (hours)"])])
        plt.xlabel("Time (hours)")
        plt.ylabel("Energy (kW)")
    elif setting == "ordered":
        result = result.sort_values(by=["Demand"],ascending=False)
        i_list = []
        save_value = np.zeros(len(result))
        for i in supply:
            if np.isnan(result[i.name + " corrected"][0] ):
                continue
            i_list.append(i)
            value = 0

            for j in range(len(i_list)):
                value += result[i_list[j].name + " corrected"] /(len_timestep/3600)
            #plt.plot(result["Time (hours)"],value,label = i.name)
            ax.fill_between(np.linspace(0,len(result),len(result)),value, save_value, label=i.name)
            save_value = value
        if Storage:
            for i in supply:
                for j in storage_obj.supplier: 
                    if i.name == j.name:                                
                        if storage_obj.HP != None:
                            Factor_due_HP = (j.T_out-(demand.T_out-storage_obj.HP.delta_T_coldside))/(j.T_out-demand.T_out)
                        else:
                            Factor_due_HP = 1
            to_storage = 0
            for i in supply:
                for j in storage_obj.supplier:
                    if j.name == i.name:
                        to_storage = to_storage + result[i.name + " percentage to storage"]*result[i.name+ " production"]/Factor_due_HP
                        unused = result[i.name + " production"] -to_storage*Factor_due_HP-result["Demand"]

            unused = np.clip(unused, a_min=0,a_max=None) 
            unused = unused *(unused > 0.001)
            unused = result["Demand"]+unused
            HP = to_storage*Factor_due_HP+unused
            to_storage = to_storage+unused
            ax.fill_between(np.linspace(0,len(result),len(result)),HP/(len_timestep/3600),result['Demand']/(len_timestep/3600), label="HP to storage")
            ax.fill_between(np.linspace(0,len(result),len(result)),to_storage/(len_timestep/3600),result['Demand']/(len_timestep/3600),label="To storage")
            ax.fill_between(np.linspace(0,len(result),len(result)),unused/(len_timestep/3600),result['Demand']/(len_timestep/3600), label="Unused")
        plt.plot(np.linspace(0,len(result),len(result)),result["Demand"]/(len_timestep/3600),label = 'demand',color = 'k',linewidth = 0.5)
        plt.legend()
        plt.ylim([0,max(result["Demand"])/(len_timestep/3600)*1.1])
        plt.xlim([0,len(result)])
        plt.xlabel("Time (hours)")
        plt.ylabel("Energy (kW)")
        
    elif setting == "demand_met":
        i_list = []
        save_value = np.zeros(len(result))
        for i in supply:
            if np.isnan(result[i.name + " corrected"][0] ):
                continue
            i_list.append(i)
            value = 0

            for j in range(len(i_list)):
                value += result[i_list[j].name + " corrected"] /(len_timestep/3600)
            #plt.plot(result["Time (hours)"],value,label = i.name)
            ax.fill_between(result["Time (hours)"],value, save_value, label=i.name)
            save_value = value
        if Storage:
            for i in supply:
                for j in storage_obj.supplier: 
                    if i.name == j.name:                                
                        if storage_obj.HP != None:
                            Factor_due_HP = (j.T_out-(demand.T_out-storage_obj.HP.delta_T_coldside))/(j.T_out-demand.T_out)
                        else:
                            Factor_due_HP = 1
            to_storage = 0
            for i in supply:
                for j in storage_obj.supplier:
                    if j.name == i.name:
                        to_storage = to_storage + result[i.name + " percentage to storage"]*result[i.name+ " production"]/Factor_due_HP
                        unused = result[i.name + " production"] -to_storage*Factor_due_HP-result["Demand"]

            unused = np.clip(unused, a_min=0,a_max=None) 
            unused = unused *(unused > 0.001)
            unused = result["Demand"]+unused
            HP = to_storage*Factor_due_HP+unused
            to_storage = to_storage+unused
            if (HP.sum())-to_storage.sum() !=0:
                ax.fill_between(result["Time (hours)"],HP/(len_timestep/3600),result['Demand']/(len_timestep/3600), label="HP to storage")
            if (to_storage.sum()-unused.sum())!=0:
                ax.fill_between(result["Time (hours)"],to_storage/(len_timestep/3600),result['Demand']/(len_timestep/3600),label="To storage")
            if (unused.sum())-(to_storage.sum())!=0:
                ax.fill_between(result["Time (hours)"],unused/(len_timestep/3600),result['Demand']/(len_timestep/3600), label="Unused")
 
                        
                        
        plt.plot(result["Time (hours)"],result["Demand"]/(len_timestep/3600),label = 'demand',color = 'k',linewidth = 0.5)
        plt.legend()
        plt.xlim([0,max(result["Time (hours)"])])
        plt.ylim([0,max(result["Demand"])/(len_timestep/3600)*1.1])
        plt.xlabel("Time (hours)")
        plt.ylabel("Energy (kW)")
    elif setting == "Geothermal plot":
          i_list = []
          save_value = np.zeros(len(result))
          for i in supply:
              if i.name == "Geothermal well":
                  
                i_list.append(i)
                value = 0
                for j in range(len(i_list)):
                    value += max(result[i_list[j].name + " corrected"]) /(len_timestep/3600)
                #plt.plot(result["Time (hours)"],value,label = i.name)
                ax.fill_between(result["Time (hours)"],value, save_value, label=i.name)
                save_value = value
          if Storage:
              for i in supply:
                  for j in storage_obj.supplier: 
                      if i.name == j.name:                                
                          if storage_obj.HP != None:
                              Factor_due_HP = (j.T_out-(demand.T_out-storage_obj.HP.delta_T_coldside))/(j.T_out-demand.T_out)
                          else:
                              Factor_due_HP = 1
              to_storage = 0
              for i in supply:
                  for j in storage_obj.supplier:
                      if j.name == i.name:
                          to_storage = to_storage + result[i.name + " percentage to storage"]*result[i.name+ " production"]/Factor_due_HP

              ax.fill_between(result["Time (hours)"],save_value-(to_storage/(len_timestep/3600)),save_value,label="To storage")
   
                          
                          
          plt.plot(result["Time (hours)"],result["Demand"]/(len_timestep/3600),label = 'demand',color = 'k',linewidth = 0.5)
          plt.legend()
          plt.xlim([0,max(result["Time (hours)"])])
          plt.ylim([0,max(result["Demand"])/(len_timestep/3600)*1.1])
          plt.xlabel("Time (hours)")
          plt.ylabel("Energy (kWh)")
    else:
        raise ValueError("Wrong setting chosen. Choose between demand_met and everything")

if __name__ == "__main__":
    well_cap_V_ATES=80
    N_wells_per_hotwell = 3
    well_cost = 155000
    facilities_cost = 12500000
    maintenance_yearly = 100000
    well_maintenance_yearly = 17000
    elec_use_ATES = 0.4625#kW/m^3 in one hour both charging and discharging no depth dependency
    Inefficiency_factor_elec = 0.5
    Elec_cost = 0.1 #euro/kWh
    CO2_emissions_grid = 400 #g/kWh
    
    identity = 0
    total_result = pd.DataFrame()
    
    
    timestep = 3600*24
    demand = demand_class(T_in =90,T_out = 55,example_demand="Delft Total")
    demand.data = demand.data*1
    gas = gas_boiler()
    
    for i in range(1):
        for j in range(1):
            i = 9
            j = 9              
    
    
            geo = geothermal(power=5000+i*1000,T_out=80)#,flow_rate = 320)
            heat_pump = heat_pump_ATES(delta_T_coldside=0)
            #ATES = ATES_obj([geo],max_V = 50+j*50,HP=heat_pump,thickness=40,kh=5,ani=4,T_ground=15,N_wells=12)
            solar = Solar_collector(power = 120)
            supply = [geo,gas,solar]
    
            result,df_flow = system(demand,supply,len_timestep=timestep)
    
            system_plot(result, supply, demand, len_timestep=timestep, setting ='ordered')
            system_plot(result, supply, demand, len_timestep=timestep, setting ='demand_met')
            system_plot(result, supply, demand, len_timestep=timestep, setting ='everything')
            plt.figure(dpi=800)
            plt.plot(demand.data,label = "Demand")
            plt.plot(solar.output,label = "Solar thermal")
            plt.hlines(geo.power*20,xmin=0,xmax=365,label="Geothermal heat",color="green")
            plt.legend()
            plt.yticks([])
            plt.xlabel("Day of the year")
            plt.ylabel("Heat generated")
            
            #plt.show()
            result.insert(loc = 0,column="id",value = identity)
            total_result = pd.concat([total_result,result])
            identity = identity+1
    
    total_demand = sum(demand.data)
    RES_share_list = []
    boiler_list = []
    #GENERALIZE THIS PART
    for i in range(max(total_result["id"])+1):
        RES_lib = ["Geothermal well corrected","ATES corrected","Heat pump corrected", "Solar boiler corrected"]
        RES_share = 0
        for j in total_result.columns:
            if any(RES in j for RES in RES_lib):
                RES_share = RES_share+sum(total_result[total_result["id"]==i][j])
                try:
                    RES_share = RES_share - sum(total_result[total_result["id"]==i][j.replace('corrected',"percentage to storage")]*total_result[total_result["id"]==i][j.replace(" corrected"," production")])
                    pass
                except:
                    pass
        RES_share = RES_share/total_demand
        RES_share_list.append(RES_share)
    
           
    plt.plot(RES_share_list,'.')
    df_eco = economic_analysis(result, supply)
    sum_cost = 0
    
    df_CO2 = CO2_emissions_calc(result, supply)
    for i in supply:
        if np.isnan(df_eco.loc[i.name]["LCOE"]):
            pass
        else:
            sum_cost = sum_cost + df_eco.loc[i.name]["LCOE"]*sum(result[str(i.name)+" corrected"])
    for i in range(len(df_eco)):
        print("LCOH of ", df_eco.iloc[i,0]," = ", round(df_eco.iloc[i].loc["LCOE"],3),"euro/kWh")
               
               
               
               
               
    
    
