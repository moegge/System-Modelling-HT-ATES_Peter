# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:26:20 2023

@author: 6100430
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import pytest
from ATES_obj_Peter import ATES_obj
from line_profiler import profile

pytestmark = pytest.mark.filterwarnings("error::FutureWarning")
    
pd.options.mode.chained_assignment = None  # default='warn'


# class Biomass_boiler:
#     #This object is not finalized and does not work properly
#     def __init__(self, power = 1000, eff = 0.90):  #costperkW = 2500, gas_price = 0.1, opexascapex = 0.05):
#         self.name = 'Biomass boiler'
#         self.control = 'controlled'
#         self.type = 'supply'
#         self.rated_power = power #kW
#         self.gas_content = 14.97 #kWh/kg
#         self.costperkW = 1#costperkW #euro/kW
#         self.capex = self.costperkW*self.rated_power
#         self.price = 1#gas_price #euro/kWh
#         self.opexascapex = 1#opexascapex
#         ### Costs still need to be determined
#         self.lifetime = 15
#         self.eff = eff

#     def calc_output(self,needed_output):
#         output = np.clip(needed_output,a_min=0,a_max=None)
#         return output              
#     def calc_opex(self,kWh_generated):
#         kWh_in = kWh_generated /self.eff
#         var_opex = kWh_in*self.price
#         fix_opex = self.capex * self.opexascapex
#         opex = var_opex+fix_opex
#         return opex

class Solar_collector:
    """
    A class representing a solar thermal collector for District heating.

    Parameters
    ----------
    peak_power : float, optional
        The peak power output of the solar collector in kW (default is 1000 kW).
    output_array : numpy.ndarray, optional
        Precomputed solar output distribution (default is None, will use weather data).
    weather_data : pandas.DataFrame, optional
        Weather data containing solar irradiance (default is None).
    T_out : float, optional
        The output temperature of the heated fluid in °C (default is 60°C).
    efficiency : float, optional
        Efficiency of the solar collector (default is 0.75).
    heat_capacity_fluid : float, optional
        Specific heat capacity of the heat transfer fluid in J/(kg·K) (default is 4186 J/kg·K).
    density_fluid : float, optional
        Density of the heat transfer fluid in kg/m³ (default is 997 kg/m³).
    cost_per_kW : float, optional
        Capital cost per kW installed (default is 340 €/kW).
    fixed_opex : float, optional
        Fixed operational expenses per year per kWp (default is 4.1 €/kWp/year).
    var_opex : float, optional
        Variable operational expenses per kWh generated (default is 0.0019 €/kWh).
    heat_per_kWp : float, optional
        Annual heat production per kWp in kWh (default is 600 kWh/kWp).
    lifetime : int, optional
        Expected lifetime of the solar collector in years (default is 20 years).

    Attributes
    ----------
    output : numpy.ndarray
        Time series of the solar collector's thermal output.
    """
    def __init__(self, peak_power=1000, output_array=None, weather_data=None, T_out = 60,
                 efficiency = 0.75, heat_capacity_fluid = 4186, density_fluid = 997,
                 cost_per_kW = 340, fixed_opex = 4.1, var_opex=0.0019,heat_per_kWp = 600,
                 lifetime = 20):
        self.control = 'uncontrolled'   # Defines how the system is managed
        self.type = 'supply'            # Specifies that this is a supply system
        self.name = "Solar boiler"      # Name of the supply system
        self.T_out = T_out              # Output temperature of heated fluid (°C)
        self.heat_cap = heat_capacity_fluid  # Heat capacity of the transfer fluid (J/kg·K)
        self.density = density_fluid    # Density of the transfer fluid (kg/m³), default is water
        
        ##Economics from SDE++ subsidy dutch government
        self.capex = peak_power * cost_per_kW   # capex in euro 
        self.fixed_opex = fixed_opex*peak_power # fixed opex, in euro/kWP/year * kWp
        self.var_opex = var_opex                #variable opex in euro/kWh
        self.lifetime = lifetime                #Lifetime in years
        self.peak_power = peak_power  # Installed peak power (kW)
        
        # Compute total heat generation based on installed capacity
        heat_generated = peak_power * heat_per_kWp  # Total kWh generated per year
        
        # Determine the output profile
        if output_array is not None:
            sum_output = sum(output_array)
            self.output = heat_generated * (output_array/sum_output)
        
        else:
            # Load weather data if no predefined output array is given
            df_weather = pd.read_parquet('weather_small_Amsterdam')
            output = np.array(df_weather["Solar irradiance"])
            sum_output = sum(output)
            self.output = heat_generated * (output/sum_output)
        
    def calc_output(self,demand):
        return self.output
    
    def adjust_for_timesetting(self, len_timestep = 3600):
        if len_timestep == 3600:
                pass # No adjustment needed for 1-hour timesteps
        else:
            factor = 3600/len_timestep # Compute scaling factor
            if factor > 1:
                self.output = np.repeat(self.output, factor)/factor
            if factor < 1:
                self.output = self.output.reshape(int(len(self.output)*factor),-1).sum(1)
                
    def calc_opex(self,kWh_generated):
        opex = self.fixed_opex + kWh_generated*self.var_opex
        return opex
    
    def calc_flow(self,T_in):
        #Calculate flow from the solar collector
        Joules = self.output*3600000     
        delta_t = self.T_out- T_in
        volume = Joules/delta_t/self.heat_cap/self.density
        return volume, self.T_out
    
    def calc_emissions(self,result):
        #calculate emissions. Set to 0
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
        if demand_array == None:
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
                path = r'C:\Users\0527831\PycharmProjects\System-Modelling-HT-ATES_Peter\Data_and_scripts\Demand_data\Warmtevraag_Delft_parquet'
                excel_file = pd.read_parquet(path)
                #excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file.drop(["Demand Total","Demand OWD"],inplace=True,axis=1)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Delft City":
                path = r'C:\Users\0527831\PycharmProjects\System-Modelling-HT-ATES_Peter\Data_and_scripts\Demand_data\Warmtevraag_Delft_parquet'
                excel_file = pd.read_parquet(path)
                #excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file.drop(["Demand Total","Demand TUD"],inplace=True,axis=1)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]        
            elif example_demand == "Delft Total":
                path = r'C:\Users\0527831\PycharmProjects\System-Modelling-HT-ATES_Peter\Data_and_scripts\Demand_data\Warmtevraag_Delft_parquet'
                excel_file = pd.read_parquet(path)
                #excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file.drop(["Demand OWD","Demand TUD"],inplace=True,axis=1)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]  
            elif example_demand == "Amsterdam":
                path = r'C:\Users\0527831\PycharmProjects\System-Modelling-HT-ATES_Peter\Data_and_scripts\Demand_data\Warmtevraag_Amsterdam_50_GWh_parquet'
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]  
            elif example_demand == "Tianjin":
                path = r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Warmtevraag_Tianjin_50GWh_parquet'
                #excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]  
            elif example_demand == "Athens":
                path = r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Warmtevraag_Athens_50GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Bagdad":
                path = r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Warmtevraag_Bagdad_50GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Berlin":
                path = r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Warmtevraag_Berlin_50GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Chongqin":
                path = r'C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Warmtevraag_Chongqin_50GWh_parquet'
                # excel_file = pd.read_excel(path,"Warmtevraag")
                excel_file = pd.read_parquet(path)
                excel_file = excel_file * 1000
                self.data=np.transpose(np.array(excel_file))[0,:]
            elif example_demand == "Oslo":
                pass
            #Add different types of demand here.
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
class supplier_template:
    """
    This is a template for how to make an supply object to be used in this script
    It should have the following function definitions within the object.
    """
    def __init__(self, power = 5, capex =100,opex=5):
        self.name = 'template' #System name
        self.control = 'stable'       #Control system type
        self.type = 'supply'          #Heat supply t ype
        self.capex = capex
        self.power = power
        self.CO2_kg = 200 #kgCO2/MWh or gCO2/kWh
    def calc_output(self,len_timestep,demand):
        return self.power/len_timestep #in kWh
    
    def calc_opex(self, produced_kWh):
        #Calculate opex in euros/year
        return self.opex * produced_kWh 
    
    def calc_flow(self,T_in):
        #Calculate flow object, necessary for STES
        volume = T_in *5 #Nonsensical equation
        return volume, self.T_out
    
    def calc_emissions(self,result):
        #Calculate emissions with some error handling
        generated = sum(result["supplier corrected"])
        return generated*self.CO2_kg #gram CO2


class heat_pump_ATES:
    """
    Heat pump on the ATES discharge side.
    FIXED inputs:  power_el [kW_el] (compressor rating), delta_T_coldside [K]
                   (cools ATES water this far below the DHN return -> fixed cold-well
                   injection temperature), and hence a CONSTANT COP.
    VARIABLE output: heat supplied by the HP, Q_HP = Q_evap + P_el, limited each
                   step by the compressor power AND the source water available.

    Required by the ATES engine: name == "Heat pump", power_el, delta_T_coldside,
    Calculate_COP(Tsink, Tsource), and a COP array the ATES writes into.
    """
    def __init__(self, power_el, delta_T_coldside,
                 costperkW=1200, fixed_opex=50, elec_price=0.15,
                 lifetime=15, COP_max=5.0, CO2_kg_el=200,
                 M_supplier=0.02, tau_transport=0.0198, P_contract_kW=None,
                 VR_per_month=36.75, c_contract_per_kW_month=2.0228,
                 c_max_per_kW_month=3.0966, APV_per_year=1505.00):
        self.control = 'controlled'
        self.type    = 'supply'
        self.name    = "Heat pump"

        self.power_el         = power_el          # [kW_el] fixed compressor rating
        self.delta_T_coldside = delta_T_coldside  # [K] fixed cooling below the DHN return

        self.capex      = costperkW    # euro/kW
        self.fixed_opex = fixed_opex   # euro/kW/yr
        self.elec_price = elec_price   # euro/kWh, flat fallback
        self.CO2_kg_el = CO2_kg_el  # gCO2/kWh grid intensity (flat; not hour-resolved)
        self.lifetime   = lifetime

        # Hourly spot price [EUR/kWh], attached by system() when dynamic dispatch
        # is used. None -> flat elec_price. Stedin MS 2026 tariffs for Eq. 2.
        self.elec_spot_series = None
        self.elec_cost_breakdown = None
        self.M_supplier              = M_supplier
        self.tau_transport           = tau_transport
        self.P_contract_kW           = power_el if P_contract_kW is None else P_contract_kW
        self.VR_per_month            = VR_per_month
        self.c_contract_per_kW_month = c_contract_per_kW_month
        self.c_max_per_kW_month      = c_max_per_kW_month
        self.APV_per_year            = APV_per_year

        self.COP_max     = COP_max
        self.COP         = None        # array, written by the ATES
        self.rated_power = None
        self.elec_input  = None

    def init(self, ATES):
        if ATES.name != 'ATES':
            raise ValueError("Heat pump connected to wrong storage type")

    def Calculate_COP(self, Tsink, Tsource):
        lift = Tsink - Tsource
        if lift <= 0:
            return self.COP_max
        fc  = 0.35 + 0.6/200 * lift
        return min(fc * (Tsink + 273.0) / lift, self.COP_max)

    def calc_output(self, needed_output):  return 0
    def calc_emissions(self, result):
        # Grid CO2 from compressor electricity [g] = P_el [kWh] * CO2_kg_el [gCO2/kWh]
        try:
            return float(np.nansum(self.elec_input)) * self.CO2_kg_el
        except Exception:
            return 0

    def elec_cost(self, len_timestep=3600):
        """
        Annual compressor electricity cost [EUR]. Eq. 2 (hourly spot + degressive
        tax + transport + capacity charges) when a spot series is attached;
        flat elec_price otherwise. Stores the component breakdown.
        """
        if self.elec_input is None:
            return 0.0
        if self.elec_spot_series is None:
            self.elec_cost_breakdown = None
            return float(np.nansum(np.nan_to_num(self.elec_input) * self.elec_price))
        total, breakdown = elec_cost_annual(
            self.elec_input, self.elec_spot_series, len_timestep=len_timestep,
            M=self.M_supplier, tau=self.tau_transport,
            P_contract_kW=self.P_contract_kW,
            VR_per_month=self.VR_per_month,
            c_contract_per_kW_month=self.c_contract_per_kW_month,
            c_max_per_kW_month=self.c_max_per_kW_month,
            APV_per_year=self.APV_per_year)
        self.elec_cost_breakdown = breakdown
        return total

    def calc_opex(self, kWh_generated):
        try:
            fixed_opex = self.fixed_opex * self.rated_power
            opex = self.elec_cost() + fixed_opex
        except Exception:
            opex = 0
        return opex
        
     
class geothermal:
    """
    A class representing a geothermal heat supply system.

    Parameters
    ----------
    flow_rate : float, optional
        The flow rate of the geothermal fluid in m³/hour (default is None).
    power : float, optional
        The thermal power output in kW (default is None).
    costperkW : float, optional
        Capital cost per kW installed (default is 1909 €/kW).
    fixed_opex : float, optional
        Fixed operational expenses per year per kW (default is 69 €/kW/year).
    var_opex : float, optional
        Variable operational expenses per kWh generated (default is 0.0072 €/kWh).
    T_out : float, optional
        The output temperature of the geothermal fluid in °C (default is 90°C).
    depth : int, optional
        Depth of the geothermal well in meters (default is 2000m).
    N_wells : int, optional
        Number of wells in the geothermal system (default is 2).
    lifetime : int, optional
        Expected lifetime of the geothermal plant in years (default is 30 years).
    heat_capacity_fluid : float, optional
        Specific heat capacity of the fluid in J/(kg·K) (default is 4186 J/kg·K).
    density_fluid : float, optional
        Density of the heat transfer fluid in kg/m³ (default is 997 kg/m³).
    CO2_kg : float, optional
        CO₂ emissions per MWh of generated heat (default is 27 kg CO₂/MWh).

    Attributes
    ----------
    output : float
        The thermal energy output in kWh.
    """
    def __init__(self, flow_rate = None,power = None,costperkW = 1909, fixed_opex = 69,
                 var_opex = 0.0072, T_out = 90, depth = 2000,N_wells=2,lifetime = 30,
                 heat_capacity_fluid = 4186, density_fluid = 997, CO2_kg=27):
        self.name = 'Geothermal well' #System name
        self.control = 'stable'       #Control system type
        self.type = 'supply'          #Heat supply t ype
        
        # Handling input parameters for either flow rate or power
        if flow_rate!=None:
            if power!= None:
                print("both flow_rate and power given, only one is needed, taking flow rate")
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
        
        #Economics from SDE++ subsidy.
        self.costperkW = costperkW #Capital cost, calculation will be done later.
        self.fixed_opex = fixed_opex #euro/kW/year
        self.var_opex = var_opex #euro/kWh
        self.T_out = T_out  # Output temperature of geothermal fluid in °C
        self.heat_cap = heat_capacity_fluid  # Specific heat capacity of the fluid (J/kg·K)
        self.density = density_fluid  # Density of the fluid (kg/m³)
        self.CO2_kg = CO2_kg  # CO₂ emissions per MWh of generated heat
        self.lifetime = lifetime  # Expected lifetime in years


    def calc_output(self,len_timestep,demand):
        if self.power == None:
            # Calculate power output based on flow rate and temperature difference
            self.power = self.flow_rate*(self.T_out-demand.T_out)*10**-7*4186*1000*2.77777
            if self.power <=0:
                pass
        
        # Capital expenditure (CAPEX) calculation
        self.capex = self.power*self.costperkW #euro --> kw * (euro/kW)

        # Calculate energy output (kWh)
        output = self.power*len_timestep/3600 #kWh
        self.output = output
        return output
    
    def calc_opex(self, produced_kWh):
        #Calculate opex
        fixed_opex = self.fixed_opex*self.power
        var_opex = produced_kWh*self.var_opex
        opex = var_opex+fixed_opex
        return opex

    def calc_flow(self,T_in):
        #Calculate flow of geothermal well
        Joules = self.output*3600000     
        delta_t = self.T_out- T_in
        volume = Joules/delta_t/self.heat_cap/self.density #m^3
        return volume, self.T_out
    
    def calc_emissions(self,result):
        #Calculate emissions with some error handling
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

# Energy tax 2026, electricity, excl. VAT (upper kWh limit, EUR/kWh).
# Zone 5 zakelijk = 0.00310; the compressor never reaches it on its own.
ENERGY_TAX_BRACKETS_2026 = [
    (2_900,       0.09161),
    (10_000,      0.09161),
    (50_000,      0.06671),
    (10_000_000,  0.03735),
    (np.inf,      0.00310),
]


def load_spot_price(csv_path, n_timesteps, year=2025, len_timestep=3600,
                    col_time="Datetime (Local)", col_price="Price (EUR/MWhe)"):
    """
    Day-ahead spot price [EUR/kWh], resampled to len_timestep and length-checked
    against n_timesteps. Shared by the dispatch signal and the cost calculation
    so both are driven by exactly the same series.
    """
    df = pd.read_csv(csv_path)
    df[col_time] = pd.to_datetime(df[col_time])
    df = df[df[col_time].dt.year == year].copy()
    df = df.dropna(subset=[col_price]).sort_values(col_time).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No price data for year {year} in {csv_path}")

    spot = df[col_price].values / 1000.0          # EUR/MWh -> EUR/kWh

    # Match the demand resampling in demand_class.adjust_for_timesetting
    factor = 3600 / len_timestep
    if factor > 1:
        if factor != int(factor):
            raise ValueError("len_timestep must divide 3600 evenly")
        spot = np.repeat(spot, int(factor))
    elif factor < 1:
        raise ValueError("len_timestep > 3600 not supported")

    if len(spot) != n_timesteps:
        raise ValueError(f"price series length {len(spot)} != n_timesteps "
                         f"{n_timesteps}. Price file gave {len(df)} hours for "
                         f"{year}; check for DST duplicates or gaps.")
    return spot


def elec_cost_annual(P_el, spot, len_timestep=3600,
                     M=0.02, tau=0.0198, P_contract_kW=None,
                     VR_per_month=36.75, c_contract_per_kW_month=2.0228,
                     c_max_per_kW_month=3.0966, APV_per_year=1505.00,
                     brackets=ENERGY_TAX_BRACKETS_2026):
    """
    Eq. 2 - full annual compressor electricity cost [EUR] from the REALISED
    per-timestep consumption. Post-simulation: P_el only exists after calc_heat.

    Grid terms are Stedin grootverbruik 2026, category MS (151-1500 kW).
    P_peak,m is taken as the monthly max of the timestep-average power, so it
    understates a true 15-min peak -> conservative on the kW-max charge.

    Returns (total_eur, breakdown_dict).
    """
    P_el = np.nan_to_num(np.asarray(P_el, dtype=float))
    spot = np.nan_to_num(np.asarray(spot, dtype=float))
    if len(P_el) != len(spot):
        raise ValueError(f"P_el ({len(P_el)}) and spot ({len(spot)}) length mismatch")

    E_total = float(P_el.sum())
    if E_total <= 0:
        return 0.0, {}

    # Degressive energy tax on the annual total
    tax, lower = 0.0, 0.0
    for upper, rate in brackets:
        if E_total <= lower:
            break
        tax += (min(E_total, upper) - lower) * rate
        lower = upper

    # Monthly peak power [kW]
    power_kW = P_el * (3600.0 / len_timestep)
    hours = np.arange(len(P_el)) * (len_timestep / 3600.0)
    month = (pd.Timestamp("2025-01-01") + pd.to_timedelta(hours, unit="h")).month
    peaks = np.array([power_kW[month == m].max() if (month == m).any() else 0.0
                      for m in range(1, 13)])

    if P_contract_kW is None:
        P_contract_kW = float(power_kW.max())

    b = {
        "commodity+markup":  float(((spot + M) * P_el).sum()),
        "energy tax":        tax,
        "transport per kWh": tau * E_total,
        "vastrecht":         12 * VR_per_month,
        "kW contract":       12 * c_contract_per_kW_month * P_contract_kW,
        "kW max":            float((c_max_per_kW_month * peaks).sum()),
        "APV":               APV_per_year,
    }
    return float(sum(b.values())), b


def build_hp_dispatch(n_timesteps,
                      csv_path=r"C:\Users\0527831\PycharmProjects\System-Modelling-HT-ATES_Peter\Data_and_scripts\Electricity Data\Netherlands.csv",
                      year=2025, len_timestep=3600,
                      M=0.02, eb_marg_decision=0.03735, tau=0.0198,
                      threshold_eur_mwh=60.0, basis="c_marg",
                      col_time="Datetime (Local)", col_price="Price (EUR/MWhe)",
                      verbose=True, return_spot=False):
    """
    Per-timestep HP dispatch intent from the day-ahead spot price.

    c_marg,i = P_spot,i + M + eb_marg + tau   [EUR/kWh]
    HP_ON,i  = basis,i < threshold

    Depends only on the price series - no HP consumption, no system state - so it
    can be built once in preprocessing and reused across scenarios and sweeps.
    eb_marg is pinned (top zakelijk bracket); see the standalone pricing script.

    Parameters
    ----------
    csv_path : str
        Hourly day-ahead price file [EUR/MWh].
    n_timesteps : int
        Length the returned vector must have (= len(demand.data) after
        adjust_for_timesetting, i.e. 8760 * 3600/len_timestep).
    basis : {"c_marg", "spot"}
        Threshold on the all-in marginal cost, or on the raw wholesale price.

    Returns
    -------
    np.ndarray of bool, length n_timesteps
    """
    if basis not in ("c_marg", "spot"):
        raise ValueError("basis must be 'c_marg' or 'spot'")

    spot = load_spot_price(csv_path, n_timesteps, year=year,
                           len_timestep=len_timestep,
                           col_time=col_time, col_price=col_price)

    # eb_marg_decision is the DECISION rate only: 0.03735 is the bracket the
    # compressor's annual consumption (~0.5-2 GWh) always lands in. The full
    # degressive stack is applied afterwards in elec_cost_annual.
    adder = M + eb_marg_decision + tau
    c_marg = spot + adder

    basis_mwh = (c_marg if basis == "c_marg" else spot) * 1000.0
    hp_on = basis_mwh < threshold_eur_mwh

    if verbose:
        n_on = int(hp_on.sum())
        print(f"  HP dispatch: {n_on:,}/{len(hp_on):,} steps ON "
              f"({n_on/len(hp_on)*100:.1f}%) | threshold "
              f"{threshold_eur_mwh:.1f} EUR/MWh on {basis} | adder "
              f"{adder*1000:.2f} EUR/MWh", flush=True)

    return (hp_on, spot) if return_spot else hp_on

def system(demand, supply, len_timestep = 3600, time_horizon=8760, control = None,
           hp_on=None, hp_dynamic_dispatch=False, hp_threshold_eur_mwh=60.0,
           hp_elec_spot_series=None):
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

    # P: why not changed for stable supply?
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
        #P: is this a fair method of correcting the flow rate?
        # Save demand flow in variable to be manipulated
        flow_not_covered = df_flow["Demand volume"]
        
        # Check if there is a uncontrollable/stable sources not connected to storage and adjust uncovered flow to demand.
        for i in supply:
            if i.control == "controlled" or i.control == 'storage':
                continue
            break_loop=0
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
        # Fixed cold-well temperature (modelling assumption: it does not change).
        # Set equal to the level the discharge HP cools to (= T_floor in calc_heat),
        # so charging and discharging use the SAME cold-well temperature. Bounded
        # below by the ground temperature. No HP -> cold well stays at the return.
        if storage_obj.HP is not None:
            T_cold = max(demand.T_out - storage_obj.HP.delta_T_coldside, storage_obj.T_g)
        else:
            T_cold = demand.T_out
        
        # For each supply connected to storage, check how much volume can go to the storage
        for i in storage_obj.supplier: 
            #Check if there is a HP and initalize it to see what the role of it is.
            if storage_obj.HP != None:
                storage_obj.HP.init(storage_obj)
                #Calculate the reduction of volume that can go the to ATES due to a larger temperature difference generated by the HP.
                # Charging works from the FIXED cold-well temperature up to the supply
                # temperature; larger delta -> less water needed for the same heat.
                Factor_due_HP = (i.T_out - T_cold) / (i.T_out - demand.T_out) #? Adapt later to actual losses! Also adapt plotting at comment here: Changeplothere
            else:
                Factor_due_HP=1
                
            
            # Check if we are injecting, calculate the flows to the storage
            result[i.name + " percentage to storage"] = (1-percentage_used)*storage_injection
            df_flow[i.name + " flow to storage"] = (1-percentage_used)*df_flow[i.name+ " Volume out"]*storage_injection/Factor_due_HP
            T_total =  T_total+sum(df_flow[i.name + " flow to storage"]) * i.T_out #Check so the flow to the storage is at the temperature of the supply out?
            df_flow["Total flow to storage"] =  df_flow["Total flow to storage"] + df_flow[i.name + " flow to storage"]
    
        # Calculate the temperature to the storage
        if sum(df_flow["Total flow to storage"])>0:
             T_av = T_total/sum(df_flow["Total flow to storage"]) #P: Adapt once ATES charging by HP is implemented
        else:
            T_av=0
       
        volume  = sum(df_flow["Total flow to storage"])
        
        #ATES system has internal HX which reduces inlet temperatures
        T_ineff_due_HX = 0#(1-storage_obj.HX_eta)* (T_av-demand.T_out) #P: check source
        
        loss_percentage = 1
        #Calculate losses in cold well, which needs to be compensated for. 
        for j in storage_obj.supplier:
            if j.name != 'Geothermal well': #P: Why is this only done for non-geo sources? Is the cold return temp considered as solar entry T?
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
                    storage_obj.init_cold_well(T_cold, volume) #P: HX Inefficiency could be implemented here
                        
                    #Calculate extra volume required due to heat losses
                    volume = Heat_to_storage/(T_av-storage_obj.cold_well_T_ave)
                    k = k+1
                    if sum(dummy_flow) ==0:
                        loss_percentage = 1
                    else:
                        loss_percentage = 1-(volume/sum(dummy_flow))
                        
            #If we are working with a geothermal well, the returrn temperature should be as low as possible
            #why not include the loop here as well?
            else:
                percentage=(storage_obj.max_V*len_timestep/3600)/df_flow["Total flow to storage"]
                percentage[percentage==np.inf]=0
                percentage = np.clip(percentage,a_min=None,a_max=1)

                            
        #Save total flow to storage
        df_flow["Total flow to storage"]=df_flow["Total flow to storage"]*percentage #P: is this flawed if it is chosen to use multiple suppliers? percentage used from last supplier (loop above)
        storage_obj.flow_injected = np.array(df_flow["Total flow to storage"])

        for i in storage_obj.supplier: 
            result[i.name + " percentage to storage"] = result[i.name + " percentage to storage"]*percentage
            df_flow[i.name + " flow to storage"] = df_flow[i.name + " flow to storage"]*percentage
       
        # Run the storage simulation
        if sum(df_flow["Total flow to storage"])>1:
            # Initialize storage
            storage_obj.initialize(sum(df_flow["Total flow to storage"]),T_av-T_ineff_due_HX, len_timestep)

            # Energy still to be covered, and the ATES(+HP) output for it.
            missing_energy = result['Demand'] - result['Total production']

            # Per-timestep HP dispatch vector, passed straight to the ATES model.
            # Built here from the spot price if none was supplied, so the caller
            # doesn't have to. demand.adjust_for_timesetting has already run, so
            # len(demand.data) is the correct timestep count.
            if storage_obj.HP is not None and hp_on is None and hp_dynamic_dispatch:
                hp_on, hp_elec_spot_series = build_hp_dispatch(
                    n_timesteps=len(demand.data), len_timestep=len_timestep,
                    threshold_eur_mwh=hp_threshold_eur_mwh, return_spot=True)

            # Carry the hourly spot price to the economics. Eq. 2 is evaluated
            # after the run, from the realised P_el. None -> flat elec_price.
            if storage_obj.HP is not None:
                storage_obj.HP.elec_spot_series = hp_elec_spot_series

            if storage_obj.HP is not None and hp_on is not None:
                hp_vec = np.asarray(hp_on, dtype=bool)
                if len(hp_vec) != len(missing_energy):
                    raise ValueError(f"hp_on must have one value per timestep "
                                     f"({len(missing_energy)}), got {len(hp_vec)}")
            else:
                # No HP, or no dispatch intent -> pass None. calc_heat keeps the HP off,
                # and the mode-D override is itself guarded by 'self.HP is not None',
                # so with no HP nothing can turn it on.
                hp_vec = None

            output_storage = storage_obj.calc_heat(
                demand.T_out, demand.T_in, storage_extraction, missing_energy,
                hp_on=hp_vec, len_timestep=len_timestep, control=control)

            # HP contribution is computed inside calc_heat and stored on the object.
            if storage_obj.HP is not None:
                output_HP = storage_obj.output_HP  # condenser heat = Q_evap + P_el [kWh]
                result["Heat pump production"] = output_HP
                storage_obj.HP.rated_power = storage_obj.HP.power_el  # fixed compressor rating [kW]
                storage_obj.HP.elec_input = storage_obj.P_el  # compressor electricity [kWh]
                storage_obj.HP.COP = storage_obj.COP  # per-timestep COP (NaN when off)
            else:
                output_HP = 0
                result["Heat pump production"] = output_HP

            # ATES production now represents the WHOLE subsystem (direct HX + HP condenser).
            # Keep the direct-only part (Q_dir) available for plotting / inspection.
            result[storage_obj.name + " production"]  = output_storage            # ATES + HP
            result[storage_obj.name + " direct only"] = output_storage - output_HP  # Q_dir
            result['Total production'] = (result['Total production']
                                          + result[storage_obj.name + " production"])
            # HP is already inside "ATES production" -> do NOT add "Heat pump production" again.

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
            # Utilisation of the well's annual capacity, on an HX-only basis.
            #   numerator   = "ATES corrected" = Q_dir + Q_evap + P_el, demand-clipped
            #                 (the WHOLE subsystem, HP condenser heat included)
            #   denominator = year-8 extractable heat above T_cutoff over the full
            #                 volume curve. NOTE the array name says T_ground but
            #                 calc_heat computes it against T_cutoff.
            # Can exceed 1.0 in GGAH: the HP draws from the band below T_cutoff,
            # which the denominator does not count. Diagnostic only - the LCOE
            # level is unaffected, since max_extracted cancels in mature years
            # (j % lifetime >= 8) and the array only carries the ramp shape.
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
            max_extracted = i.total_heat_extracted_vs_T_ground_kWh_first_8_years[-1] #P: check what exactly this does; is this affected by the implementation of the HP?
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

def economic_analysis(results_system, supply,disc_rate = 0.05,incorporate_CO2=False,CO2_price = 70,opex_ATES_fixed=False,len_timestep=3600):
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
        hp_capex = 0
        if i.name == "ATES":
            if opex_ATES_fixed:
                try:
                    df_eco.loc[i.name,"opex"]=i.fix_opex+(i.volume+sum(i.flow_extracted))/2/1000000*1389*1000*i.elec_price
                except:
                    df_eco.loc[i.name,"opex"] = 0
            else:
                df_eco.loc[i.name,"opex"]= i.calc_opex(sum(results_system[i.name + " corrected"]))

            # --- Discharge-side HP economics (HP is bolted onto the ATES) ---
            hp = getattr(i, "HP", None)
            if hp is not None:
                elec = getattr(hp, "elec_input", None)
                elec_kWh = float(np.nansum(elec)) if elec is not None else 0.0  # compressor kWh/yr
                hp_rating = getattr(hp, "rated_power", None) or hp.power_el  # kW
                # Eq. 2 when an hourly spot series is attached, flat otherwise.
                df_eco.loc[i.name, "opex"] = (df_eco.loc[i.name, "opex"]
                                              + hp.elec_cost(len_timestep=len_timestep)
                                              + hp.fixed_opex * hp_rating)  # HP fixed opex
                hp_capex = 0.0  # hp.capex holds euro/kW  -> total euro
                # The HP (hp.lifetime) is amortised inside the ATES row over i.lifetime.
                # Buy a unit whenever the previous one expires, discounted to year 0, and
                # pay only for the years actually used -> the last unit (and the first, if
                # the HP outlives the ATES) is bought as a fraction. CAPEX ONLY: the HP
                # opex above is euro/kW/yr on hp_rating and euro/kWh on electricity,
                # neither derived from hp_capex.
                for k in range(0, i.lifetime, hp.lifetime):
                    frac = min(hp.lifetime, i.lifetime - k) / hp.lifetime
                    hp_capex += (hp_rating * hp.capex * frac) / (1 + disc_rate) ** k

        else:
            #df_eco["opex"].iloc[count] = i.calc_opex(sum(results_system[i.name + " corrected"]))
            df_eco.loc[i.name,"opex"]= i.calc_opex(sum(results_system[i.name + " corrected"]))
        df_eco.at[i.name,"name"] = i.name
        df_eco.at[i.name,"capex"] = i.capex + hp_capex
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

    # Fixed cold-well temperature, identical to system(): keeps the plotted storage
    # bands consistent with the Factor_due_HP the simulation actually used.
    # P: Change this once the loop regarding the cold well losses is improved! Changeplothere
    if Storage and storage_obj.HP is not None:
        T_cold = max(demand.T_out - storage_obj.HP.delta_T_coldside, storage_obj.T_g)
    else:
        T_cold = demand.T_out
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
        hp = result["Heat pump production"] / (len_timestep / 3600) if "Heat pump production" in result else np.zeros(
            len(result))  # HP: discharge heat-pump band [kW]
        hp_offset = np.zeros(len(result))  # HP: added to later bands so they stack on top of the HP
        for i in supply:
            i_list.append(i)
            value = 0
            for j in range(len(i_list)):
                nm = i_list[j].name
                # ATES production now includes the HP; use the direct-only part for the band
                # so the HP is drawn once, as its own band on top.
                col = nm + " direct only" if (nm + " direct only") in result else nm + " production"
                value += result[col] / (len_timestep / 3600)
            value = value + hp_offset  # HP
            # plt.plot(result["Time (hours)"],value,label = i.name,visible=False)
            ax.fill_between(result["Time (hours)"], value, save_value, label=i.name)
            save_value = value
            if i.control == "storage" and np.nansum(hp) > 1e-9:  # HP: draw the discharge heat pump on top of the ATES
                ax.fill_between(result["Time (hours)"], value + hp, value, label="Heat pump")
                save_value = value + hp
                hp_offset = hp
        plt.plot(result["Time (hours)"], result["Demand"] / (len_timestep / 3600), label='demand', color='k',
                 linewidth=0.5)
        plt.legend()
        plt.xlim([0, max(result["Time (hours)"])])
        plt.xlabel("Time (hours)")
        plt.ylabel("Energy (kW)")
    elif setting == "ordered":
        result = result.sort_values(by=["Demand"], ascending=False)
        i_list = []
        save_value = np.zeros(len(result))
        hp = result["Heat pump production"] / (len_timestep / 3600) if "Heat pump production" in result else np.zeros(
            len(result))  # HP: discharge heat-pump band [kW]
        hp_offset = np.zeros(len(result))  # HP: added to later bands so they stack on top of the HP
        for i in supply:
            if np.isnan(result[i.name + " corrected"][0]):
                continue
            i_list.append(i)
            value = 0

            for j in range(len(i_list)):
                nm = i_list[j].name
                value += result[nm + " corrected"] / (len_timestep / 3600)
                # ATES corrected now includes HP heat; strip it so the HP band on top
                # isn't double counted (mirrors the "everything" setting).
                if i_list[j].control == "storage" and "Heat pump production" in result:
                    value -= result["Heat pump production"] / (len_timestep / 3600)
            value = value + hp_offset  # HP
            # plt.plot(result["Time (hours)"],value,label = i.name)
            ax.fill_between(np.linspace(0, len(result), len(result)), value, save_value, label=i.name)
            save_value = value
            if i.control == "storage" and np.nansum(hp) > 1e-9:  # HP: draw the discharge heat pump on top of the ATES
                ax.fill_between(np.linspace(0, len(result), len(result)), value + hp, value, label="Heat pump")
                save_value = value + hp
                hp_offset = hp
        if Storage:
            Factor_due_HP = 1 #P: Revisit to update cold well loss iteration runs, implement that it is not always lowered to T_floor
            to_storage = 0
            for i in supply:
                for j in storage_obj.supplier:
                    if j.name == i.name:
                        to_storage = to_storage + result[i.name + " percentage to storage"] * result[
                            i.name + " production"] / Factor_due_HP
                        unused = result[i.name + " production"] - to_storage * Factor_due_HP - result["Demand"]

            unused = np.clip(unused, a_min=0, a_max=None)
            unused = unused * (unused > 0.001)
            unused = result["Demand"] + unused
            to_storage = to_storage + unused
            ax.fill_between(np.linspace(0, len(result), len(result)), to_storage / (len_timestep / 3600),
                            result['Demand'] / (len_timestep / 3600), label="To storage")
            ax.fill_between(np.linspace(0, len(result), len(result)), unused / (len_timestep / 3600),
                            result['Demand'] / (len_timestep / 3600), label="Unused")
        plt.plot(np.linspace(0, len(result), len(result)), result["Demand"] / (len_timestep / 3600), label='demand',
                 color='k', linewidth=0.5)
        plt.legend()
        plt.ylim([0,max(result["Demand"])/(len_timestep/3600)*1.1])
        plt.xlim([0,len(result)])
        plt.xlabel("Time (hours)")
        plt.ylabel("Energy (kW)")
        
    elif setting == "demand_met":
        i_list = []
        save_value = np.zeros(len(result))
        hp = result["Heat pump production"] / (len_timestep / 3600) if "Heat pump production" in result else np.zeros(
            len(result))  # HP: discharge heat-pump band [kW]
        hp_offset = np.zeros(len(result))  # HP: added to later bands so they stack on top of the HP
        for i in supply:
            if np.isnan(result[i.name + " corrected"][0] ):
                continue
            i_list.append(i)
            value = 0

            for j in range(len(i_list)):
                nm = i_list[j].name
                value += result[nm + " corrected"] / (len_timestep / 3600)
                # ATES corrected now includes HP heat; strip it so the HP band on top
                # isn't double counted (mirrors the "everything" setting).
                if i_list[j].control == "storage" and "Heat pump production" in result:
                    value -= result["Heat pump production"] / (len_timestep / 3600)
            value = value + hp_offset  # HP
            # plt.plot(result["Time (hours)"],value,label = i.name)
            ax.fill_between(result["Time (hours)"], value, save_value, label=i.name)
            save_value = value
            if i.control == "storage" and np.nansum(hp) > 1e-9:  # HP: draw the discharge heat pump on top of the ATES
                ax.fill_between(result["Time (hours)"], value + hp, value, label="Heat pump")
                save_value = value + hp
                hp_offset = hp
        if Storage:
            for i in supply:
                for j in storage_obj.supplier: 
                    if i.name == j.name:                                
                        if storage_obj.HP != None:
                            Factor_due_HP = 1 #Changeplothere
                        else:
                            Factor_due_HP = 1
            to_storage = 0
            for i in supply:
                for j in storage_obj.supplier:
                    if j.name == i.name:
                        to_storage = to_storage + result[i.name + " percentage to storage"] * result[
                            i.name + " production"] / Factor_due_HP
                        unused = result[i.name + " production"] - to_storage * Factor_due_HP - result["Demand"]

            unused = np.clip(unused, a_min=0, a_max=None)
            unused = unused * (unused > 0.001)
            unused = result["Demand"] + unused
            HP = to_storage * Factor_due_HP + unused
            to_storage = to_storage + unused
            if (HP.sum()) - to_storage.sum() != 0:
                ax.fill_between(result["Time (hours)"], HP / (len_timestep / 3600),
                                result['Demand'] / (len_timestep / 3600), label="HP to storage")
            if (to_storage.sum() - unused.sum()) != 0:
                ax.fill_between(result["Time (hours)"], to_storage / (len_timestep / 3600),
                                result['Demand'] / (len_timestep / 3600), label="To storage")
            if (unused.sum()) - (to_storage.sum()) != 0:
                ax.fill_between(result["Time (hours)"], unused / (len_timestep / 3600),
                                result['Demand'] / (len_timestep / 3600), label="Unused")

        plt.plot(result["Time (hours)"], result["Demand"] / (len_timestep / 3600), label='demand', color='k',
                 linewidth=0.5)
        plt.legend()
        plt.xlim([0, max(result["Time (hours)"])])
        plt.ylim([0, max(result["Demand"]) / (len_timestep / 3600) * 1.1])
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
                    value += max(result[i_list[j].name + " corrected"]) / (len_timestep / 3600)
                # plt.plot(result["Time (hours)"],value,label = i.name)
                ax.fill_between(result["Time (hours)"], value, save_value, label=i.name)
                save_value = value
        if Storage:
            for i in supply:
                for j in storage_obj.supplier:
                    if i.name == j.name:
                        if storage_obj.HP != None:
                            Factor_due_HP = 1 #Changeplothere (old:Factor_due_HP = (j.T_out-(demand.T_out-storage_obj.HP.delta_T_coldside))/(j.T_out-demand.T_out))
                        else:
                            Factor_due_HP = 1
            to_storage = 0
            for i in supply:
                for j in storage_obj.supplier:
                    if j.name == i.name:
                        to_storage = to_storage + result[i.name + " percentage to storage"] * result[
                            i.name + " production"] / Factor_due_HP

            ax.fill_between(result["Time (hours)"], save_value - (to_storage / (len_timestep / 3600)), save_value,
                            label="To storage")

        plt.plot(result["Time (hours)"], result["Demand"] / (len_timestep / 3600), label='demand', color='k',
                 linewidth=0.5)
        plt.legend()
        plt.xlim([0, max(result["Time (hours)"])])
        plt.ylim([0, max(result["Demand"]) / (len_timestep / 3600) * 1.1])
        plt.xlabel("Time (hours)")
        plt.ylabel("Energy (kWh)")
    else:
        raise ValueError("Wrong setting chosen. Choose between demand_met and everything")