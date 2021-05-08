#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is developed by:
    Lewis Rowles <stetsonsc@gmail.com>

Code is based on Bwaise systems.py developed by:
    Yalin Li <zoe.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/QSDsan/blob/master/LICENSE.txt
for license details.
"""
import numpy as np
import biosteam as bst
import qsdsan as qs
from sklearn.linear_model import LinearRegression as LR
from qsdsan import sanunits as su
from qsdsan import WasteStream, ImpactIndicator, ImpactItem, StreamImpactItem, SimpleTEA, LCA

#from qsdsan.systems import bwaise as bw

from _cmps import cmps

print(qs.__version__)

#import all units
# from _housing_biogenic_refinery import HousingBiogenicRefinery
# from _industrial_control_panel import IndustrialControlPanel
# from _screw_press import ScrewPress
# from _carbonizer_base import CarbonizerBase
# from _pollution_control_device import PollutionControlDevice
# from _oil_heat_exchanger import OilHeatExchanger
# from _hydronic_heat_exchanger import HydronicHeatExchanger
# from _dryer_from_hhx import DryerFromHHX
# from _ion_exchange_NH3 import IonExchangeNH3
# from _struvite_precipitation import StruvitePrecipitation
# from _grinder import Grinder
# =============================================================================
# Unit parameters
# =============================================================================

bst.settings.set_thermo(cmps)
#items = ImpactItem._items

currency = qs.currency = 'USD'
bst.speed_up()

household_size = 4
get_household_size = lambda: household_size
household_per_toilet = 4
get_household_per_toilet = lambda: household_per_toilet
get_toilet_user = lambda: get_household_size()*get_household_per_toilet()

# Number of people served by the one Biogenic Refinery 1018
ppl = 12000

discount_rate = 0.05
get_discount_rate = lambda: discount_rate

# Time take for full degradation, [yr]
tau_deg = 2
# Log reduction at full degradation
log_deg = 3
# Get reduction rate constant k for COD and N, use a function so that k can be
# changed during uncertainty analysis
def get_decay_k(tau_deg=2, log_deg=3):
    k = (-1/tau_deg)*np.log(10**-log_deg)
    return k


max_CH4_emission = 0.25
get_max_CH4_emission = lambda: max_CH4_emission



# Model for tanker truck cost based on capacity (m3)
# price = a*capacity**b -> ln(price) = ln(a) + bln(capacity)
USD_price_dct = np.array((21.62, 32.43, 54.05, 67.57))
capacities = np.array((3, 4.5, 8, 15))
emptying_fee = 0.15
get_emptying_fee = lambda: emptying_fee
def get_tanker_truck_fee(capacity):
    price_dct = USD_price_dct*(1+get_emptying_fee())
    ln_p = np.log(price_dct)
    ln_cap = np.log(capacities)
    model = LR().fit(ln_cap.reshape(-1,1), ln_p.reshape(-1,1))
    [[predicted]] = model.predict(np.array((np.log(capacity))).reshape(1, -1)).tolist()
    cost = np.exp(predicted)
    return cost

# Nutrient loss during applciation
app_loss = dict.fromkeys(('NH3', 'NonNH3', 'P', 'K', 'Mg', 'Ca'), 0.02)
app_loss['NH3'] = 0.05

# Energetic content of the biogas
biogas_energy = 803 # kJ/mol CH4
get_biogas_energy = lambda: biogas_energy
LPG_energy = 50 # MJ/kg
get_LPG_energy = lambda: LPG_energy
get_biogas_factor = lambda: get_biogas_energy()/cmps.CH4.MW/get_LPG_energy()

# =============================================================================
# Prices and GWP CFs
# =============================================================================

# Recycled nutrients are sold at a lower price than commercial fertilizers

price_factor = 0.25
get_price_factor = lambda: price_factor

operator_daily_wage = 29
get_operator_daily_wage = lambda: operator_daily_wage

price_dct = {
    'Electricity': 0.17,
    'Concrete': 194,
    'Steel': 2.665,
    'N': 1.507*get_price_factor(),
    'P': 3.983*get_price_factor(),
    'K': 1.333*get_price_factor(),
    'Polymer': 0.75,
    'Resin': 3.335,
    'FilterBag': 4.08,
    'MgOH2': 0.145,
    'MgCO3': 0.9,
    'H2SO4': 0.3,
    }


GWP_dct = {
    'Electricity': 0.15,
    'CH4': 28,
    'N2O': 265,
    'N': -5.4,
    'P': -4.9,
    'K': -1.5,
    'Polymer':2.8,
    'Resin': 1.612,
    'FilterBag': 1,
    'MgOH2': 1.176277921,
    'MgCO3': 1.176277921,
    'H2SO4': 0.158899487,
    }

items = ImpactItem.get_all_items()

if not items.get('Excavation'): # prevent from reloading
    import os
    path = os.path.dirname(os.path.realpath(__file__)) + '/data'
    ImpactIndicator.load_indicators_from_file(path+'/impact_indicators.tsv')
    item_path = path+'/impact_items.xlsx'
    ImpactItem.load_items_from_excel(item_path)
    del os

GWP = qs.ImpactIndicator.get_indicator('GWP')

bst.PowerUtility.price = price_dct['Electricity']
items['Concrete'].price = price_dct['Concrete']
items['Steel'].price = price_dct['Steel']

# =============================================================================
# Universal units and functions
# =============================================================================

CH4_item = StreamImpactItem(ID='CH4_item', GWP=GWP_dct['CH4'])
N2O_item = StreamImpactItem(ID='N2O_item', GWP=GWP_dct['N2O'])
N_item = StreamImpactItem(ID='N_item', GWP=GWP_dct['N'])
P_item = StreamImpactItem(ID='P_item', GWP=GWP_dct['P'])
K_item = StreamImpactItem(ID='K_item', GWP=GWP_dct['K'])
e_item = ImpactItem(ID='e_item', functional_unit='kWh', GWP=GWP_dct['Electricity'])
polymer_item = StreamImpactItem(ID='polymer_item', GWP=GWP_dct['Polymer'])
resin_item = StreamImpactItem(ID='resin_item', GWP=GWP_dct['Resin'])
filter_bag_item = StreamImpactItem(ID='filter_bag_item', GWP=GWP_dct['FilterBag'])
MgOH2_item = StreamImpactItem(ID='MgOH2_item', GWP=GWP_dct['MgOH2'])
MgCO3_item = StreamImpactItem(ID='MgCO3_item', GWP=GWP_dct['MgCO3'])
H2SO4_item = StreamImpactItem(ID='H2SO4_item', GWP=GWP_dct['H2SO4'])

def batch_create_streams(prefix):
    stream_dct = {}
    stream_dct['CH4'] = WasteStream(f'{prefix}_CH4', phase='g',
                                    impact_item=CH4_item.copy(set_as_source=True))
    stream_dct['N2O'] = WasteStream(f'{prefix}_N2O', phase='g',
                                    impact_item=N2O_item.copy(set_as_source=True))
    stream_dct['liq_N'] = WasteStream(f'{prefix}_liq_N', phase='l', price=price_dct['N'],
                                      impact_item=N_item.copy(set_as_source=True))
    stream_dct['sol_N'] = WasteStream(f'{prefix}_sol_N', phase='l', price=price_dct['N'],
                                      impact_item=N_item.copy(set_as_source=True))
    stream_dct['liq_P'] = WasteStream(f'{prefix}_liq_P', phase='l', price=price_dct['P'],
                                      impact_item=P_item.copy(set_as_source=True))
    stream_dct['sol_P'] = WasteStream(f'{prefix}_sol_P', phase='l', price=price_dct['P'],
                                      impact_item=P_item.copy(set_as_source=True))
    stream_dct['liq_K'] = WasteStream(f'{prefix}_liq_K', phase='l', price=price_dct['K'],
                                      impact_item=K_item.copy(set_as_source=True))
    stream_dct['sol_K'] = WasteStream(f'{prefix}_sol_K', phase='l', price=price_dct['K'],
                                      impact_item=K_item.copy(set_as_source=True))
    stream_dct['polymer'] = WasteStream(f'{prefix}_polymer', phase='s', price=price_dct['Polymer'],
                                      impact_item=resin_item.copy(set_as_source=True))
    stream_dct['resin'] = WasteStream(f'{prefix}_resin', phase='s', price=price_dct['Resin'],
                                      impact_item=resin_item.copy(set_as_source=True))
    stream_dct['filter_bag'] = WasteStream(f'{prefix}_filter_bag', phase='s', price=price_dct['FilterBag'],
                                      impact_item=filter_bag_item.copy(set_as_source=True))
    stream_dct['MgOH2'] = WasteStream(f'{prefix}_MgOH2', phase='s', price=price_dct['MgOH2'],
                                      impact_item=MgOH2_item.copy(set_as_source=True))
    stream_dct['MgCO3'] = WasteStream(f'{prefix}_MgCO3', phase='s', price=price_dct['MgCO3'],
                                      impact_item=MgCO3_item.copy(set_as_source=True))
    stream_dct['H2SO4'] = WasteStream(f'{prefix}_H2SO4', phase='s', price=price_dct['H2SO4'],
                                      impact_item=H2SO4_item.copy(set_as_source=True))
    

    return stream_dct
    

def add_fugitive_items(unit, item):
    unit._run()
    for i in unit.ins:
        i.impact_item = item.copy(set_as_source=True)
        


# %%

# =============================================================================
# Scenario A (sysA): pit latrine with Biogenic Refinery
# =============================================================================

flowsheetA = bst.Flowsheet('sysA')
bst.main_flowsheet.set_flowsheet(flowsheetA)

streamsA = batch_create_streams('A')

#################### Human Inputs ####################
# !!! how to change excrestion based on location (e.g., calorie and protein)
A1 = su.Excretion('A1', outs=('urine','feces'))

################### User Interface ###################
# !!! how to change inputs based on location (e.g., flushing water, cleaning water, toilet paper)
A2 = su.PitLatrine('A2', ins=(A1-0, A1-1,
                              'toilet_paper', 'flushing_water',
                              'cleansing_water', 'desiccant'),
                   outs=('mixed_waste', 'leachate', 'A2_CH4', 'A2_N2O'),
                   N_user=get_toilet_user(), N_toilet=ppl/get_toilet_user(),
                   if_flushing=False, if_desiccant=False, if_toilet_paper=False,
                   OPEX_over_CAPEX=0.05,
                  decay_k_COD=get_decay_k(tau_deg, log_deg),
                   decay_k_N=get_decay_k(tau_deg, log_deg),
                   max_CH4_emission=get_max_CH4_emission())
 
##################### Conveyance of Waste #####################
#!!! add conveyance

A3 = su.Trucking('A3', ins=A2-0, outs=('transported', 'conveyance_loss'),
                  load_type='mass', distance=5, distance_unit='km',
                  interval=A2.emptying_period, interval_unit='yr',
                  loss_ratio=0.02)
def update_A3_param():
    A3._run()
    truck = A3.single_truck
    truck.interval = A2.emptying_period*365*24
    truck.load = A3.F_mass_in*truck.interval/A2.N_toilet
    rho = A3.F_mass_in/A3.F_vol_in
    vol = truck.load/rho
    A3.fee = get_tanker_truck_fee(vol)
    A3._design()
A3.specification = update_A3_param

###################### Treatment ######################
# !!! How to add housing and industral control panel for capital, opex, and energy

# !!! add liquid treatment and change effluent from screw press accordingly

A4 = su.ScrewPress('A4', ins=(A3-0, streamsA['polymer']), outs=('liq', 'cake_sol'))
A5 = su.CarbonizerBase('A5', outs = ('biochar', 'A5_hot_gas', 'A5_N2O'))
 
A6 = su.PollutionControlDevice('A6', ins = (A5-1, A5-2), outs = ('A6_hot_gas_pcd', 'A6_N2O'))

# updating uptime_ratio in all units to follow carbonizer base
old_cost = A6._cost
def update_uptime_ratio():
    A9.uptime_ratio = A8.uptime_ratio = A7.uptime_ratio = A6.uptime_ratio = A5.uptime_ratio 
    old_cost()
A6._cost = update_uptime_ratio

A7 = su.OilHeatExchanger('A7', ins = A6-0, outs = ('A7_hot_gas'))
A8 = su.HydronicHeatExchanger('A8', ins = A7-0, outs = ('A8_hot_gas'))
A9 = su.DryerFromHHX('A9', ins = (A4-1, A8-0), outs = ('waste_out', 'A9_N2O', 'A9_CH4'))

A9-0-A5

A10 = su.Mixer('A10', ins=(A2-2, A9-2), outs=streamsA['CH4'])
A10.specification = lambda: add_fugitive_items(A10, CH4_item)
A10.line = 'fugitive CH4 mixer' 
        
A11 = su.Mixer('A11', ins=(A2-3, A6-1, A9-1), outs=streamsA['N2O'])
A11.specification = lambda: add_fugitive_items(A11, N2O_item)
A11.line = 'fugitive N2O mixer'

#!!! How to add unit for energy balance based on unit from Biosteam
#stack_heat_loss = net_thermal_energy_in - power_delivery_orc - pipe_heat_loss - heat_output_water - heat_loss_water_pipe - jacket_heat_loss_sum
#**add which units each energy is coming from, e.g., A4.net_thermal_energy_in

################## Reuse or Disposal ##################
#!!! add conveyance of biochar and crop application 



############### Simulation, TEA, and LCA ###############
sysA = bst.System('sysA', path= (A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11))

sysA.simulate()

power = sum([u.power_utility.rate for u in sysA.units])

#!!! update labor to input country specific data and be a distribution
teaA = SimpleTEA(system=sysA, discount_rate=get_discount_rate(), 
                  start_year=2020, lifetime=20, uptime_ratio=1, 
                  lang_factor=None, annual_maintenance=0, 
                  annual_labor=(get_operator_daily_wage() * 3*365), construction_schedule=None)

lcaA = LCA(system=sysA, lifetime=20, lifetime_unit='yr', uptime_ratio=1,
            e_item=lambda: power*(365*24)*20)

# %%

# =============================================================================
# Scenario B (sysB): UDDT with Biogenic Refinery
# =============================================================================

flowsheetB = bst.Flowsheet('sysB')
bst.main_flowsheet.set_flowsheet(flowsheetB)

streamsB = batch_create_streams('B')

#################### Human Inputs ####################
# !!! how to change excrestion based on location (e.g., calorie and protein)
B1 = su.Excretion('B1', outs=('urine','feces'))

################### User Interface ###################
# !!! how to change inputs based on location (e.g., flushing water, cleaning water, toilet paper)

B2 = su.UDDT('B2', ins=(B1-0, B1-1,
                        'toilet_paper', 'flushing_water',
                        'cleaning_water', 'desiccant'),
             outs=('liq_waste', 'sol_waste',
                   'struvite', 'HAP', 'B2_CH4', 'B2_N2O'),
             N_user=get_toilet_user(), N_toilet=ppl/get_toilet_user(), 
             if_flushing=False, if_desiccant=False, if_toilet_paper=False,
             OPEX_over_CAPEX=0.1,
             decay_k_COD=get_decay_k(tau_deg, log_deg),
             decay_k_N=get_decay_k(tau_deg, log_deg),
             max_CH4_emission=get_max_CH4_emission())
 
##################### Conveyance of Waste #####################
# Liquid waste
handcart_fee = 0.01 # USD/cap/d
get_handcart_fee = lambda: handcart_fee
truck_fee = 6.21 # USD/m3
get_truck_fee = lambda: truck_fee

get_handcart_and_truck_fee = \
    lambda vol, ppl: get_truck_fee()*vol \
        + get_handcart_fee()*ppl*B2.collection_period
B3 = su.Trucking('B3', ins=B2-0, outs=('transported_l', 'loss_l'),
                 load_type='mass', distance=5, distance_unit='km',
                 loss_ratio=0.02)

# Solid waste
B4 = su.Trucking('B4', ins=B2-1, outs=('transported_s', 'loss_s'),
                 load_type='mass', load=1, load_unit='tonne',
                 distance=5, distance_unit='km',
                 loss_ratio=0.02)
def update_B3_B4_param():
    B4._run()
    truck3, truck4 = B3.single_truck, B4.single_truck
    hr = truck3.interval = truck4.interval = B2.collection_period*24
    ppl_t = ppl / B2.N_toilet
    truck3.load = B3.F_mass_in * hr / B2.N_toilet
    truck4.load = B4.F_mass_in * hr / B2.N_toilet
    rho3 = B3.F_mass_in/B3.F_vol_in
    rho4 = B4.F_mass_in/B4.F_vol_in
    B3.fee = get_handcart_and_truck_fee(truck3.load/rho3, ppl_t)
    B4.fee = get_handcart_and_truck_fee(truck4.load/rho4, ppl_t)
    B3._design()
    B4._design()
B4.specification = update_B3_B4_param



###################### Treatment ######################
B5 = su.StruvitePrecipitation('B5', ins=(B3-0, streamsB['MgOH2'], streamsB['MgCO3'], 
                                      streamsB['filter_bag']), outs=('treated', 'Struvite'))


B6 = su.IonExchangeNH3('B6', ins=(B5-0, streamsB['resin'], streamsB['H2SO4']), 
                    outs=('treated', 'resin_out', 'conc_NH3'))


B7 = su.Grinder('B7', ins=B4-0, outs=('waste'))
B8 = su.CarbonizerBase('B8', outs = ('biochar', 'B8_hot_gas', 'B8_N2O'))
 
B9 = su.PollutionControlDevice('B9', ins = (B8-1, B8-2), outs = ('B9_hot_gas_pcd', 'B9_N2O'))

# updating uptime_ratio in all units to follow carbonizer base
old_cost = B9._cost
def update_uptime_ratio():
    B12.uptime_ratio = B11.uptime_ratio = B10.uptime_ratio = B9.uptime_ratio = B8.uptime_ratio 
    old_cost()
B9._cost = update_uptime_ratio

B10 = su.OilHeatExchanger('B10', ins = B9-0, outs = ('B10_hot_gas'))
B11 = su.HydronicHeatExchanger('B11', ins = B10-0, outs = ('B11_hot_gas'))
B12 = su.DryerFromHHX('B12', ins = (B7-0, B11-0), outs = ('waste_out', 'B12_N2O', 'B12_CH4'))

B12-0-B8

B13 = su.Mixer('B13', ins=(B2-4, B12-2), outs=streamsB['CH4'])
B13.specification = lambda: add_fugitive_items(B13, CH4_item)
B13.line = 'fugitive CH4 mixer' 
        
B14 = su.Mixer('B14', ins=(B2-5, B9-1, B12-1), outs=streamsB['N2O'])
B14.specification = lambda: add_fugitive_items(B14, N2O_item)
B14.line = 'fugitive N2O mixer'

################## Reuse or Disposal ##################
#!!! add conveyance of biochar and crop application 



############### Simulation, TEA, and LCA ###############
sysB = bst.System('sysB', path= (B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13))

sysB.simulate()

power = sum([u.power_utility.rate for u in sysB.units])

#!!! update labor to input country specific data and be a distribution
teaB = SimpleTEA(system=sysB, discount_rate=get_discount_rate(), 
                  start_year=2020, lifetime=20, uptime_ratio=1, 
                  lang_factor=None, annual_maintenance=0, 
                  annual_labor=(get_operator_daily_wage() * 3*365), construction_schedule=None)

lcaB = LCA(system=sysB, lifetime=20, lifetime_unit='yr', uptime_ratio=1,
            e_item=lambda: power*(365*24)*20)

# %%

# =============================================================================
# Summarizing Functions
# =============================================================================

#!!! user cost for each component by capital, O&M, and energy
#!!! impacts for each component by capital, direct from waste, O&M, and energy
def get_total_inputs(unit):
    if len(unit.ins) == 0: # Excretion units do not have ins
        ins = unit.outs
    else:
        ins = unit.ins
    inputs = {}
    inputs['COD'] = sum(i.COD*i.F_vol/1e3 for i in ins)
    inputs['N'] = sum(i.TN*i.F_vol/1e3 for i in ins)
    inputs['NH3'] = sum(i.imass['NH3'] for i in ins)
    inputs['P'] = sum(i.TP*i.F_vol/1e3 for i in ins)
    inputs['K'] = sum(i.TK*i.F_vol/1e3 for i in ins)
    hr = 365 * 24
    for i, j in inputs.items():
        inputs[i] = j * hr
    return inputs

def get_recovery(ins=None, outs=None, hr=365*24, ppl=1, if_relative=True):
    try: iter(outs)
    except: outs = (outs,)
    non_g = tuple(i for i in outs if i.phase != 'g')
    recovery = {}
    recovery['COD'] = sum(i.COD*i.F_vol/1e3 for i in non_g)
    recovery['N'] = sum(i.TN*i.F_vol/1e3 for i in non_g)
    recovery['NH3'] = sum(i.imass['NH3'] for i in non_g)
    recovery['P'] = sum(i.TP*i.F_vol/1e3 for i in non_g)
    recovery['K'] = sum(i.TK*i.F_vol/1e3 for i in non_g)
    for i, j in recovery.items():
        if if_relative:
            inputs = get_total_inputs(ins)
            recovery[i] /= inputs[i]/hr * ppl
        else:
            recovery[i] /= 1/hr * ppl
    return recovery

def get_stream_emissions(streams=None, hr=365*24, ppl=1):
    try: iter(streams)
    except: streams = (streams,)
    emission = {}
    factor = hr / ppl
    for i in streams:
        if not i.impact_item: continue
        emission[f'{i.ID}'] = i.F_mass*i.impact_item.CFs['GlobalWarming']*factor
    return emission

sys_dct = {
    'ppl': dict(sysA=ppl, sysB=ppl),
    'input_unit': dict(sysA=A1, sysB=B1),
    'liq_unit': dict(sysA=None, sysB=None),
    'sol_unit': dict(sysA=None, sysB=None),
    'gas_unit': dict(sysA=None, sysB=None),
    'stream_dct': dict(sysA=streamsA, sysB=streamsB),
    'TEA': dict(sysA=teaA, sysB=teaB),
    'LCA': dict(sysA=lcaA, sysB=lcaB),
    'cache': dict(sysA={}, sysB={}),
    }

def cache_recoveries(sys):
    total_COD = get_total_inputs(sys_dct['input_unit'][sys.ID])['COD']
    ppl = sys_dct['ppl'][sys.ID]
    if sys_dct['gas_unit'][sys.ID]:
        gas_mol = sys_dct['gas_unit'][sys.ID].outs[0].imol['CH4']
        gas_COD = gas_mol*1e3*get_biogas_energy()*365*24/14e3/ppl/total_COD
        # breakpoint()
    else:
        gas_COD = 0
    cache = {
        'liq': get_recovery(ins=sys_dct['input_unit'][sys.ID],
                            outs=sys_dct['liq_unit'][sys.ID].ins,
                            ppl=ppl),
        'sol': get_recovery(ins=sys_dct['input_unit'][sys.ID],
                            outs=sys_dct['sol_unit'][sys.ID].ins,
                            ppl=ppl),
        'gas': dict(COD=gas_COD, N=0, P=0, K=0)
        }
    return cache

def update_cache(sys):
    last_u = sys.path[-1]
    last_u._run()
    sys_dct['cache'][sys.ID] = cache_recoveries(sys)

#!!! ComponentSplitter or Mixer?
# B15 is biogas and A13 and C13 are liq nuetrients 
# A13.specification = lambda: update_cache(sysA)
# B15.specification = lambda: update_cache(sysB)
# C13.specification = lambda: update_cache(sysC)


def get_summarizing_fuctions():
    func_dct = {}
    func_dct['get_annual_cost'] = lambda tea, ppl: tea.EAC/ppl
    func_dct['get_annual_CAPEX'] = lambda tea, ppl: tea.annualized_CAPEX/ppl
    func_dct['get_annual_OPEX'] = lambda tea, ppl: tea.AOC/ppl
    ind = 'GlobalWarming'
    func_dct['get_annual_GWP'] = \
        lambda lca, ppl: lca.total_impacts[ind]/lca.lifetime/ppl
    func_dct['get_constr_GWP'] = \
        lambda lca, ppl: lca.total_construction_impacts[ind]/lca.lifetime/ppl
    func_dct['get_trans_GWP'] = \
        lambda lca, ppl: lca.total_transportation_impacts[ind]/lca.lifetime/ppl  
    func_dct['get_direct_emission_GWP'] = \
        lambda lca, ppl: lca.get_stream_impacts(stream_items=lca.stream_inventory, kind='direct_emission')[ind] \
            /lca.lifetime/ppl
    func_dct['get_offset_GWP'] = \
        lambda lca, ppl: lca.get_stream_impacts(stream_items=lca.stream_inventory, kind='offset')[ind] \
            /lca.lifetime/ppl
    func_dct['get_other_GWP'] = \
        lambda lca, ppl: lca.total_other_impacts[ind]/lca.lifetime/ppl
    for i in ('COD', 'N', 'P', 'K'):
        func_dct[f'get_liq_{i}_recovery'] = \
            lambda sys, i: sys_dct['cache'][sys.ID]['liq'][i]
        func_dct[f'get_sol_{i}_recovery'] = \
            lambda sys, i: sys_dct['cache'][sys.ID]['sol'][i]
        func_dct[f'get_gas_{i}_recovery'] = \
            lambda sys, i: sys_dct['cache'][sys.ID]['gas'][i]
        func_dct[f'get_tot_{i}_recovery'] = \
            lambda sys, i: \
                sys_dct['cache'][sys.ID]['liq'][i] + \
                sys_dct['cache'][sys.ID]['sol'][i] + \
                sys_dct['cache'][sys.ID]['gas'][i]
    return func_dct


def print_summaries(systems):
    try: iter(systems)
    except: systems = (systems, )
    func = get_summarizing_fuctions()
    for sys in systems:
        sys.simulate()
        ppl = sys_dct['ppl'][sys.ID]
        print(f'\n---------- Summary for {sys} ----------\n')
        tea = sys_dct['TEA'][sys.ID]
        tea.show()
        print('\n')
        lca = sys_dct['LCA'][sys.ID]
        lca.show()
        
        unit = f'{currency}/cap/yr'
        print(f'\nNet cost: {func["get_annual_cost"](tea, ppl):.1f} {unit}.')
        print(f'Capital: {func["get_annual_CAPEX"](tea, ppl):.1f} {unit}.')
        print(f'Operating: {func["get_annual_OPEX"](tea, ppl):.1f} {unit}.')
        
        unit = f'{GWP.unit}/cap/yr'
        print(f'\nNet emission: {func["get_annual_GWP"](lca, ppl):.1f} {unit}.')
        print(f'Construction: {func["get_constr_GWP"](lca, ppl):.1f} {unit}.')
        print(f'Transportation: {func["get_trans_GWP"](lca, ppl):.1f} {unit}.')
        print(f'Direct emission: {func["get_direct_emission_GWP"](lca, ppl):.1f} {unit}.')
        print(f'Offset: {func["get_offset_GWP"](lca, ppl):.1f} {unit}.')
        print(f'Other: {func["get_other_GWP"](lca, ppl):.1} {unit}.\n')

        for i in ('COD', 'N', 'P', 'K'):
            print(f'Total {i} recovery is {func[f"get_tot_{i}_recovery"](sys, i):.1%}, '
                  f'{func[f"get_liq_{i}_recovery"](sys, i):.1%} in liquid, '
                  f'{func[f"get_sol_{i}_recovery"](sys, i):.1%} in solid, '
                  f'{func[f"get_gas_{i}_recovery"](sys, i):.1%} in gas.')

def save_all_reports():
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/results'
    if not os.path.isdir(path):
        os.path.mkdir(path)
    del os
    for i in (sysA, sysB, lcaA, lcaB):
        if isinstance(i, bst.System):
            i.simulate()
            i.save_report(f'{path}/{i.ID}.xlsx')
        else:
            i.save_report(f'{path}/{i.system.ID}_lca.xlsx')

__all__ = ('sysA', 'sysB', 'teaA', 'teaB', 'lcaA', 'lcaB',
            'print_summaries', 'save_all_reports',
            *(i.ID for i in sysA.units),
            *(i.ID for i in sysB.units),
            )



