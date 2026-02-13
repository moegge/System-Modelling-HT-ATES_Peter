# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 08:55:01 2024

@author: 6100430
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
from scipy.interpolate import interpn
from ATES_obj import ATES_obj
import joblib
import pytest
import time
from main2 import geothermal, demand_class, heat_pump_ATES,gas_boiler, system,\
    economic_analysis, LCOE_calc_Yang, CO2_emissions_calc, Solar_collector, system_plot
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from tqdm import tqdm
import multiprocessing as mp
import random
import traceback


def Optim(x):
    try:
      x=x[0]
      fname = "/path/to/storage/"+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])
      if os.path.isfile(fname):
          return 0,0
      timestep = 3600*6
      # demand = demand_class(T_in =75,T_out = 55,example_demand="Delft Total")
      demand = demand_class(T_in =75,T_out = 55,example_demand=x[0])
      gas = gas_boiler(lifetime=15,gas_price=0.055)
      if x[1] == "Solar-HT-ATES":
          df_weather = pd.read_parquet("/path/to/weatherfile/"+x[0])
          output = np.array(df_weather["G(i)"])
          heat_per_kWp = 600*sum(output)/1047954
          solar = Solar_collector(peak_power = x[2], T_out = 75,heat_per_kWp=heat_per_kWp,output_array=output)
          solar.capex = solar.capex*x[5]
          solar.fixed_opex = solar.fixed_opex*x[5]
          solar.var_opex=solar.var_opex*x[5]
          ATES = ATES_obj([solar],CO2_elec=200,max_V = x[4],HP=None,thickness=50,kh=10,ani=5,T_ground=15,N_wells=6,depth=130,lifetime=30)
          supply = [solar,ATES,gas]
  
      elif x[1] == "Geo-HT-ATES":
          geo = geothermal(flow_rate = x[3],T_out=75,lifetime = 30,CO2_kg=12.5)
          geo.costperkW = geo.costperkW * x[5]
          geo.fixed_opex = geo.fixed_opex *x[5]
          geo.var_opex = geo.var_opex *x[5]
          ATES = ATES_obj([geo],max_V = x[4],HP=None,CO2_elec=200,thickness=50,kh=10,ani=5,T_ground=15,N_wells=6,depth=130,lifetime=30)
          supply = [geo,ATES,gas]
  
          #geo.capex = 22000000
          #ATES.capex = 3408000-options_df_row["Subsidy amount ATES"]
  
      
      #ATES.set_reff(0.85)
  
      # supply = [geo,ATES,gas]
      result,df_flow = system(demand,supply,len_timestep=timestep, control = None)
      
      ATES_fixed = True
      df_eco = economic_analysis(result, supply,disc_rate=0.05,incorporate_CO2=True,CO2_price=75,opex_ATES_fixed=ATES_fixed)
      Network_length = 23 #km
      capex_network = Network_length*1157*1000
      opex_as_capex_network = 0.02
      lifetime_network = 60
      LCOE_Yang = LCOE_calc_Yang(result,supply,df_eco,lifetime_system=60,capex_network=capex_network,opex_network_perc=opex_as_capex_network,lifetime_network=lifetime_network)
  
      RES_lib = ["Geothermal well corrected","ATES corrected","Heat pump corrected", "Solar boiler corrected"]
      RES_share = 0
      for j in result.columns:
          if any(RES in j for RES in RES_lib):
              RES_share = RES_share+sum(result[j])
              try:
                  if not np.isnan(sum(result[j.replace('corrected',"percentage to storage")]*result[j.replace(" corrected"," production")])):
                      RES_share = RES_share - sum(result[j.replace('corrected',"percentage to storage")]*result[j.replace(" corrected"," production")])
                  pass
              except:
                  pass
      RES_share = RES_share/sum(demand.data)
      if hasattr(ATES,"Reff"):
          pass
      else:
          ATES.Reff = 0
      #print(df_eco)
      sum_cost=0
      for i in supply:
          if np.isnan(df_eco.loc[i.name]["LCOE"]):
              pass
          else:
              sum_cost = sum_cost + df_eco.loc[i.name]["LCOE"]*sum(result[str(i.name)+" corrected"])
      df_eco["sum_cost"]=sum_cost
      df_eco["RES_share"]=RES_share
      df_CO2 = CO2_emissions_calc(result, supply, CO2_price=150)
      df_eco["Yearly_CO2_emissions [kg]"]=df_CO2["CO2_emission [kg]"]
      df_eco["Yearly_kWh"]=sum(demand.data)
      df = pd.DataFrame(data={"Size_Geo": [x[3]],"Size_solar":[x[2]],"Size_ATES":[x[4]],"Config":[x[1]],"Demand":[x[0]],"Ratio": [x[5]],"System_LCOH": [LCOE_Yang],"RES":[RES_share],"HT-ATES efficiency":[ATES.Reff]})    
      save_file = "/path/to/storage/"
      df.to_parquet(save_file+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5]))
      return LCOE_Yang, RES_share
    except:
      """Except clause to show which inpus made the program crash and print the inputs"""
      traceback.print_exc()
      print("one crashed", flush=True)
      return 0,0




granularity = 120
max_geo = 2100
max_HTATES = 3000
max_solar = 450000
config =["Solar-HT-ATES","Geo-HT-ATES"]
Demand = ["Berlin","Chongqing","Bagdad"]
ratio = [0.3333,0.5,1,2]
x_df=pd.DataFrame()
x = pd.DataFrame(0,index = [0], columns = ["demand","config","solar size","geo size","HT-ATES size","ratio"])
x_list=[]

for z in Demand:
    for q in ratio:
        for j in range(granularity):
            for k in range(granularity):
                for i in config:
                    if i == "Solar-HT-ATES":
                        if q==0.3333:
                            continue
                        else:
                            x["demand"]=z
                            x["config"]=i
                            x["solar size"]=j*(max_solar/granularity)
                            x["HT-ATES size"]=k*(max_HTATES/granularity)
                            x["geo size"]=0
                            x["ratio"] = q
                    elif i == "Geo-HT-ATES":
                        x["demand"]=z
                        x["config"]=i
                        x["geo size"]=j*(max_geo/granularity)
                        x["solar size"]=0
                        x["HT-ATES size"]=k*(max_HTATES/granularity)
                        x["ratio"] = q
                    x_list.append(np.array(x))


print("prep done",flush=True)                              
n_cores = mp.cpu_count()    #Get core count
if __name__ == "__main__":
    pool = mp.Pool(n_cores)     #Use all cores for multiprocessing pool
    #A,B = Optim(x_list[479])
    random.shuffle(x_list)
    results = pool.map_async(Optim, x_list)    #Run the parallell pool. Everything is saved in the function itself
    results.get()
# LCOH_matrix = np.zeros((granularity,granularity))
# RES_matrix = np.zeros((granularity,granularity))
# for i in tqdm(range(granularity)):
#     for j in range(granularity):
#         X = np.array([i*(max_x0/granularity),j*(max_x1/granularity)])
#         LCOH_matrix[i,j], RES_matrix[i,j] = Optim(X)

# np.save("LCOH_array_optimization",LCOH_matrix)
# np.save("RES_array_optimization",RES_matrix)
# LCOH_matrix = np.load("LCOH_array_optimization.npy")
# RES_matrix = np.load("RES_array_optimization.npy")



# if __name__ == '__main__':
#     runs = []
#     granularity = 10
#     max_x0 = 1500
#     max_x1 = 1000
#     runs = []
#     for i in range(granularity):
#         for j in range(granularity):
#             runs.append( np.array([i*(max_x0/granularity),j*(max_x1/granularity)]))
#     #results2 = Set_MODFLOW(5)
#     n_cores = mp.cpu_count()    #Get core count
#     pool = mp.Pool(n_cores)     #Use all cores for multiprocessing pool
#     results = pool.map_async(Optim, runs)    #Run the parallell pool. Everything is saved in the function itself
#     results = results.get()
#     df = pd.DataFrame(runs)
    
#     df.rename(columns={0:"Size_Geo",1:"Size_ATES"},inplace=True)
#     df1 = pd.DataFrame(results)
#     df1.rename(columns={0:"system LCOH",1:"RES"},inplace=True)
#     df = pd.concat([df,df1],axis=1)
#     RES_matrix = df.pivot(index = "Size_ATES",columns = "Size_Geo",values="RES")
#     LCOH_matrix = df.pivot(index = "Size_ATES",columns = "Size_Geo",values="system LCOH")
#     save_file = r"C:\Users\6100430\OneDrive - Universiteit Utrecht\PhD project\PhD python\Figures 4th paper"
    
#     # RES_matrix.to_excel(save_file+"/RES_geo_ATES.xlsx")
#     # LCOH_matrix.to_excel(save_file+"/LCOH_geo_ATES.xlsx")

    
    
    
    
    
    
    