#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
QSDsan: Quantitative Sustainable Design for sanitation and resource recovery systems
Copyright (C) 2020, Quantitative Sustainable Design Group

This module is developed by:
    Lewis Rowles <stetsonsc@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/QSDsan/blob/master/LICENSE.txt
for license details.
'''
# %%


import numpy as np
from qsdsan import SanUnit, Construction
from qsdsan.utils.loading import load_data, data_path
import os

__all__ = ('CarbonizerBase',)

#path to csv with all the inputs
data_path = os.path.abspath(os.path.dirname('_carbonizer_base.csv'))


### 
class CarbonizerBase(SanUnit):
    '''
    Carbonizer Base is where feedstock is continuously fed into the pyrolysis 
    pot where it is exposed to high temperature pyrolysis. This process 
    produces biochar and hot gases.
    
    The Carbonizer Base is the central location for the combined pyrolysis 
    and combustion process. The feedstock is received into the pyrolysis pot 
    where it is flash pyrolyzed releasing volatile gases which are then mixed 
    with air which is pumped in through the Primary Blower causing combustion 
    of the gases. Due to the combustion of gases, the Carbonizer base is the 
    hottest location of the refinery ranging between 550 - 900 deg C. This 
    range is closely monitored as pyrolysis usually starts at 350 deg C and 
    all of the volatile gases are released at 550 deg C. This temperature is 
    also important when confirming treatment of the faecal sludge, as it 
    serves as our evidence for inactivation of the microbes in the sludge. 
    After the gases are combusted, the exhaust travels below a baffle plate 
    to encourage the fallout of particulates before it proceeds to the 
    Pollution Control Device.

    
    Reference documents
    -------------------
    N/A
    
    Parameters
    ----------
    ins : WasteStream
        Dewatered solids with moisture content ≤ 35%.
    outs : WasteStream 
        Biochar, hot gas, fugitive N2O.
            set both as WasteStream 
            others could be:
            Gases consist of SO2_emissions, NOx_emissions, CO_emissions, 
            Hg_emissions, Cd_emissions, As_emissions, Dioxin_Furans_emissions.

        
    References
    ----------
    .. N/A
    
    '''
    

    def __init__(self, ID='', ins=None, outs=(), **kwargs):
        
        SanUnit.__init__(self, ID, ins, outs)

# load data from csv each name will be self.name    
        data = load_data(path=data_path)
        for para in data.index:
            value = float(data.loc[para]['expected'])
            setattr(self, para, value)
        del data
        
        for attr, value in kwargs.items():
            setattr(self, attr, value)




        
# define the number of influent and effluent streams    
    _N_ins = 1
    _N_outs = 3

# in _run: define influent and effluent streams and treatment processes 
    def _run(self):
        waste = self.ins[0]
        biochar, heat_out, N2O = self.outs
        #biochar.copy_like(self.ins[0])
        biochar.phase = 'l'
        heat_out.phase = N2O.phase = 'g'
        

        
        
        
        #define moisture content 
        if waste.F_mass == 0:
            moisture_content = 0
        else:
            moisture_content = waste.imass['H2O'] / waste.F_mass
            
        ## add check for moisture content < 35%?
        
        if moisture_content > 0.351:
            breakpoint()
            # add error to print that says lower moisture content required for pyrolysis
            
        # biochar produced is a function of influent mass
        biochar_prcd = waste.F_mass * (1 - moisture_content) * self.biochar_production_rate # kg biochar /hr
        biochar_thermal_energy = biochar_prcd * self.biochar_calorific_value # MJ/hr
        #biochar._HHV = biochar_thermal_energy
        
        # biochar_C = waste.COD * (1 - self.pyrolysis_C_gas)
        # biochar_N = waste.TN * (1 - self.pyrolysis_N_gas)
        # biochar_K = waste.TK * (1 - self.pyrolysis_K_gas)
        # biochar_P = waste.TP * (1 - self.pyrolysis_P_gas)
        
        #biochar._F_mass = biochar_prcd
        # biochar._COD = biochar_C
        # biochar._TN = biochar_N
        # biochar._TK = biochar_K
        # biochar._TP = biochar_P
        # biochar.imass['H2O'] = 0.01
        
        # biochar.TC = biochar_C
        # biochar.TN = biochar_N
        # biochar.TK = biochar_K
        # biochar.TP = biochar_P
        # biochar.imass['H2O'] = 0.01
        
        # biochar.imass['C'] = biochar_C
        # biochar.imass['N'] = biochar_N
        # biochar.imass['K'] = biochar_K
        # biochar.imass['P'] = biochar_P
        
       # biochar.imass['OtherSS'] = waste.COD * self.carbon_COD_ratio * waste.F_vol / 1e3 * (1 - self.pyrolysis_C_gas) # kg C / hr
        biochar.imass['C'] = waste.COD * self.carbon_COD_ratio * waste.F_vol / 1e3 * (1 - self.pyrolysis_C_gas)
        biochar.imass['N'] = waste.imass['N'] * (1 - self.pyrolysis_N_gas) # kg N / hr
        biochar.imass['K'] = waste.imass['K'] * (1 - self.pyrolysis_K_gas) # kg K / hr
        biochar.imass['P'] = waste.imass['P'] * (1 - self.pyrolysis_P_gas) # kg P / hr
        biochar.imass['OtherSS'] = biochar_prcd - (biochar.imass['C'] + biochar.imass['N'] + biochar.imass['K'] + biochar.imass['P'])
        biochar.imass['H2O'] = 0.025 * biochar.F_mass # kg H20 / hr with 2.5% moisture content
        #add properties of biochar based on cmps (calculated from input properties with mass balance on N and C)
        #biochar.imass['TN'] 
        # set remainder as biochar.imass['OtherSS'] = biochar_prcd
        # check mass balance with what fraction of what enters ends up as N2O
        
        
        # one unit is capable of treating 550 kg/d (35% moisture based on 20 hr of run time) or 18 kg dry/hr
        # daily run time for the influent waste
        self.daily_run_time = waste.F_mass * (1 - moisture_content) * 24 * (1 / self.loading_rate) # hr/d
        self.uptime_ratio = self.daily_run_time / 24 # ratio of uptime (all items are per hour)

        # if daily_run_time > 24 hr:
        # increase number of systems required?
      
        # hot_gas produced
        heat_out.T = self.pyrolysis_temp
       
        thermal_energy_from_feedstock = waste.F_mass * (1 - moisture_content) * self.dry_feces_heat_of_combustion # MJ/hr
 
        # calculate thermal energy entering and thermal energy required to drive off water
        heat_needed_to_dry_0 = waste.F_mass * (moisture_content) * self.energy_required_to_dry_sludge # MJ/hr

        net_thermal_energy_in = thermal_energy_from_feedstock  - heat_needed_to_dry_0 # MJ/hr
        # net_thermal_energy_in can be used to calculate energy balance of the system
        
        # N2O emissions
        N2O_emissions_broad_est = waste.TN * waste.F_vol * self.cb_N2O_emissions / 1e3 # kg N2O / hr
        N_to_gas = waste.imass['N'] * self.pyrolysis_N_gas # kg N / hr
        N2O_from_HCNO = N_to_gas * self.N_to_HCNO * self.HCNO_to_NH3 * self.NH3_to_N2O
        N20_from_NH3 = N_to_gas * self.N_to_NH3 * self.NH3_to_N2O
        N2O_emissions = N2O_from_HCNO + N20_from_NH3 # kg N2O / hr
        N2O.imass['N2O'] = N2O_emissions
             
        
    #_design will include all the construction or captial impacts  
    def _design(self):
        design = self.design_results
        
        # defining the quantities of materials/items
        # note that these items to be to be in the _impacts_items.xlsx
        design['StainlessSteel'] = SS_quant = (self.carbonizer_base_assembly_stainless +
                                               self.carbonizer_base_squarebox_stainless + 
                                               self.carbonizer_base_charbox_stainless)
        design['Steel'] = S_quant = (self.carbonizer_base_assembly_steel + 
                                     self.carbonizer_base_squarebox_steel + 
                                     self.carbonizer_base_charbox_steel)
        design['ElectricMotor'] =  EM_quant = (2.7/5.8) + (6/5.8)
        design['Electronics'] = Elect_quant = 1
        
        self.construction = (
            Construction(item='StainlessSteel', quantity = SS_quant, quantity_unit = 'kg'),
            Construction(item='Steel', quantity = S_quant, quantity_unit = 'kg'),
            Construction(item='ElectricMotor', quantity = EM_quant, quantity_unit = 'ea'),
            Construction(item='Electronics', quantity = Elect_quant, quantity_unit = 'kg'),
            )
        self.add_construction()
        
    
    #_cost based on amount of steel and stainless plus individual components
    def _cost(self):
        
        #purchase_costs is used for capital costs
        #can use quantities from above (e.g., self.design_results['StainlessSteel'])
        #can be broken down as specific items within purchase_costs or grouped (e.g., 'Misc. parts')
        self.purchase_costs['Stainless steel'] = (self.stainless_steel_cost 
                            * self.design_results['StainlessSteel'])
        self.purchase_costs['Steel'] = (self.steel_cost
                            * self.design_results['Steel'])
        self.purchase_costs['Electric motors'] = (self.char_auger_motor_cost_cb + 
                                                 self.fuel_auger_motor_cost_cb)
        self.purchase_costs['Misc. parts'] = (self.pyrolysis_pot_cost_cb +
                                              self.primary_air_blower_cost_cb +
                                              self.thermocouple_cost_cb_pcd +
                                              self.thermistor_cost_cb_pcd +
                                              self.forced_air_fan_cost_cb +
                                              self.airlock_motor_cost_cb +
                                              self.inducer_fan_cost_cb +
                                              self.biochar_collection_box_cost_cb +
                                              self.klinker_basher_cost_cb +
                                              self.drive_chain_cost_cb +
                                              self.chain_guards_cost_cb +
                                              self.door_cost_cb +
                                              self.agitator_cost_cb +
                                              self.combusion_chamber_cost_cb +
                                              self.sprayer_cost_cb +
                                              self.vent_cost_cb)
        
        self._BM = dict.fromkeys(self.purchase_costs.keys(), 1)
        
        
        #certain parts need to be replaced based on an expected lifefime
        #the cost of these parts is considered along with the cost of the labor to replace them
        cb_replacement_parts_annual_cost = (((self.daily_run_time * 365 / self.pyrolysis_pot_lifetime_cb) * self.pyrolysis_pot_cost_cb)
                                            + ((self.daily_run_time * 365 / self.char_auger_motor_lifetime_cb) * self.char_auger_motor_cost_cb)
                                            + ((self.daily_run_time * 365 / self.fuel_auger_motor_lifetime_cb) * self.fuel_auger_motor_cost_cb)
                                            + ((1 / self.primary_air_blower_lifetime_cb) * self.primary_air_blower_cost_cb)
                                            + ((1 / self.thermocouple_lifetime_cb_2pcd) * self.thermocouple_cost_cb_pcd)
                                            + ((1 / self.thermistor_lifetime_cb_2pcd) * self.thermistor_cost_cb_pcd)
                                            + ((1 / self.forced_air_fan_lifetime_cb) * self.forced_air_fan_cost_cb)
                                            + ((1 / self.airlock_motor_lifetime_cb) * self.airlock_motor_cost_cb)
                                            + ((1 / self.inducer_fan_lifetime_cb) * self.inducer_fan_cost_cb)) # USD/yr only accounts for time running
        
        cb_annual_maintenance = (((self.service_team_greasebearing_cb + self.service_team_tighten_cb + self.service_team_adjustdoor_cb) / 60 * 12 * self.service_team_wages)
                          + ((self.service_team_replacegasket_cb + self.service_team_replacedoor_cb + self.service_team_replacechain_cb + self.service_team_changefirepot_cb 
                            + self.service_team_replacecharaugers_cb) / 60 * self.service_team_wages / self.frequency_corrective_maintenance)
                          + (self.service_team_replacecharmotor_cb / 60 * (self.daily_run_time * 365 / self.char_auger_motor_lifetime_cb) * self.service_team_wages)
                          + (self.service_team_replacefuelmotor_cb / 60 * (self.daily_run_time * 365 / self.fuel_auger_motor_lifetime_cb) * self.service_team_wages)
                          + (self.service_team_replaceblower_cb / 60 * (1 / self.primary_air_blower_lifetime_cb) * self.service_team_wages)
                          + (self.service_team_replacefan_cb / 60 * (1 / self.inducer_fan_lifetime_cb) * self.service_team_wages)
                          + (self.service_team_replacepaddleswitch_cb / 60 * (1 / self.frequency_corrective_maintenance) * self.service_team_wages)
                          + (self.service_team_replaceairlock_cb / 60 * (1 / self.airlock_motor_lifetime_cb) * self.service_team_wages)) #USD/yr only accounts for time running
        
        self.add_OPEX =  (cb_replacement_parts_annual_cost + cb_annual_maintenance) / (365 * 24) # USD/hr (all items are per hour)
        
       
        power_demand = (self.carbonizer_biochar_auger_power + self.carbonizer_fuel_auger_power +
                        self.carbonizer_primary_air_blower_power) #kWh
        self.power_utility(power_demand)  #kWh
        #breakpoint()
        # costs associated with full time opperators can be added in the TEA as staff
      




       