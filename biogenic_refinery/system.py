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

from _cmps_Biogenic_Refinery import cmps

print(qs.__version__)

#import all units
from _housing_biogenic_refinery import HousingBiogenicRefinery
from _industrial_control_panel import IndustrialControlPanel
from _screw_press import ScrewPress
from _carbonizer_base import CarbonizerBase
from _pollution_control_device import PollutionControlDevice
from _oil_heat_exchanger import OilHeatExchanger
from _hydronic_heat_exchanger import HydronicHeatExchanger
from _dryer_from_hhx import DryerFromHHX

# =============================================================================
# Unit parameters
# =============================================================================

bst.settings.set_thermo(cmps)
items = ImpactItem._items

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


# =============================================================================
# Prices and GWP CFs
# =============================================================================

# Recycled nutrients are sold at a lower price than commercial fertilizers

price_factor = 0.25
get_price_factor = lambda: price_factor

operator_daily_wage = 20
get_operator_daily_wage = lambda: operator_daily_wage

price_dct = {
    'Electricity': 0.06,
    'Concrete': 194,
    'Steel': 2.665,
    'N': 1.507*get_price_factor(),
    'P': 3.983*get_price_factor(),
    'K': 1.333*get_price_factor(),
    'Polymer': 5*get_price_factor(),
    }


GWP_dct = {
    'Electricity': 0.15,
    'CH4': 28,
    'N2O': 265,
    'N': -5.4,
    'P': -4.9,
    'K': -1.5,
    'Polymer':2.8,
    }

items = ImpactItem.get_all_items()

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
polymer_item = ImpactItem(ID='polymer_item', functional_unit='kg', GWP=GWP_dct['Polymer'])

#polymer = SanStream('polymer', phase='s', units = 'kg/hr', 
 #                   price=price_dct['Polymer'], impact_item=polymer_item.copy(set_as_source=True))

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
                   CAPEX= 0,
                   OPEX_over_CAPEX=0.0,
                  decay_k_COD=get_decay_k(tau_deg, log_deg),
                   decay_k_N=get_decay_k(tau_deg, log_deg),
                   max_CH4_emission=get_max_CH4_emission())
 
##################### Conveyance of Waste #####################
#!!! add conveyance

# AT = su.Trucking('AT', ins=A2-0, outs=('transported', 'conveyance_loss'),
#                  load_type='mass', distance=5, distance_unit='km',
#                  interval=A2.emptying_period, interval_unit='yr',
#                  loss_ratio=0.02)
# def update_AT_param():
#     AT._run()
#     truck = A3.single_truck
#     truck.interval = A2.emptying_period*365*24
#     truck.load = AT.F_mass_in*truck.interval/A2.N_toilet
#     rho = AT.F_mass_in/AT.F_vol_in
#     vol = truck.load/rho
#     AT.fee = get_tanker_truck_fee(vol)
#     AT._design()
# AT.specification = update_AT_param

###################### Treatment ######################
# !!! How to add housing and industral control panel for capital, opex, and energy

# !!! add liquid treatment and change effluent from screw press accordingly

A3 = ScrewPress('A3', ins=A2-0, out=('liq', 'cake_sol'))
A4 = CarbonizerBase('A4', outs = ('biochar', 'A4_hot_gas', 'A4_N2O'))
 
A5 = PollutionControlDevice('A5', ins = (A4-1, A4-2), outs = ('A5_hot_gas_pcd', 'A5_N2O'))

# updating uptime_ratio in all units to follow carbonizer base
old_cost = A5._cost
def update_uptime_ratio():
    A8.uptime_ratio = A7.uptime_ratio = A6.uptime_ratio = A5.uptime_ratio = A4.uptime_ratio 
    old_cost()
A5._cost = update_uptime_ratio

A6 = OilHeatExchanger('A6', ins = A5-0, outs = ('A6_hot_gas'))
A7 = HydronicHeatExchanger('A7', ins = A6-0, outs = ('A7_hot_gas'))
A8 = DryerFromHHX('A8', ins = (A3-1, A7-0), outs = ('waste_out', 'A8_N2O', 'A8_CH4'))

A8-0-A4

A9 = su.Mixer('A9', ins=(A2-2, A8-2), outs=streamsA['CH4'])
A9.specification = lambda: add_fugitive_items(A9, CH4_item)
A9.line = 'fugitive CH4 mixer' 
        
A10 = su.Mixer('A10', ins=(A2-3, A5-1, A8-1), outs=streamsA['N2O'])
A10.specification = lambda: add_fugitive_items(A10, N2O_item)
A10.line = 'fugitive N2O mixer'

#!!! How to add unit for energy balance based on unit from Biosteam
#stack_heat_loss = net_thermal_energy_in - power_delivery_orc - pipe_heat_loss - heat_output_water - heat_loss_water_pipe - jacket_heat_loss_sum
#**add which units each energy is coming from, e.g., A4.net_thermal_energy_in

################## Reuse or Disposal ##################
#!!! add conveyance of biochar and crop application 

# AT = su.Trucking('AT', ins=A2-0, outs=('transported', 'conveyance_loss'),
#                  load_type='mass', distance=5, distance_unit='km',
#                  interval=A2.emptying_period, interval_unit='yr',
#                  loss_ratio=0.02)
# def update_AT_param():
#     AT._run()
#     truck = A3.single_truck
#     truck.interval = A2.emptying_period*365*24
#     truck.load = AT.F_mass_in*truck.interval/A2.N_toilet
#     rho = AT.F_mass_in/AT.F_vol_in
#     vol = truck.load/rho
#     AT.fee = get_tanker_truck_fee(vol)
#     AT._design()
# AT.specification = update_AT_param

############### Simulation, TEA, and LCA ###############
sysA = bst.System('sysA', path= (A1, A2, A3, A4, A5, A6, A7, A8, A9, A10))

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
# Scenario B (sysB): container-based sanitation with Biogenic Refinery
# =============================================================================



# %%

# =============================================================================
# Summarizing Functions
# =============================================================================

#!!! user cost for each component by capital, O&M, and energy
#!!! impacts for each component by capital, direct from waste, O&M, and energy


# def get_total_inputs(unit):
#     if len(unit.ins) == 0: # Excretion units do not have ins
#         ins = unit.outs
#     else:
#         ins = unit.ins
#     inputs = {}
#     inputs['COD'] = sum(i.COD*i.F_vol/1e3 for i in ins)
#     inputs['N'] = sum(i.TN*i.F_vol/1e3 for i in ins)
#     inputs['NH3'] = sum(i.imass['NH3'] for i in ins)
#     inputs['P'] = sum(i.TP*i.F_vol/1e3 for i in ins)
#     inputs['K'] = sum(i.TK*i.F_vol/1e3 for i in ins)
#     hr = 365 * 24
#     for i, j in inputs.items():
#         inputs[i] = j * hr
#     return inputs

# def get_recovery(ins=None, outs=None, hr=365*24, ppl=1, if_relative=True):
#     try: iter(outs)
#     except: outs = (outs,)
#     non_g = tuple(i for i in outs if i.phase != 'g')
#     recovery = {}
#     recovery['COD'] = sum(i.COD*i.F_vol/1e3 for i in non_g)
#     recovery['N'] = sum(i.TN*i.F_vol/1e3 for i in non_g)
#     recovery['NH3'] = sum(i.imass['NH3'] for i in non_g)
#     recovery['P'] = sum(i.TP*i.F_vol/1e3 for i in non_g)
#     recovery['K'] = sum(i.TK*i.F_vol/1e3 for i in non_g)
#     for i, j in recovery.items():
#         if if_relative:
#             inputs = get_total_inputs(ins)
#             recovery[i] /= inputs[i]/hr * ppl
#         else:
#             recovery[i] /= 1/hr * ppl
#     return recovery

# def get_stream_emissions(streams=None, hr=365*24, ppl=1):
#     try: iter(streams)
#     except: streams = (streams,)
#     emission = {}
#     factor = hr / ppl
#     for i in streams:
#         if not i.impact_item: continue
#         emission[f'{i.ID}'] = i.F_mass*i.impact_item.CFs['GlobalWarming']*factor
#     return emission

# sys_dct = {
#     'ppl': dict(sysA=get_ppl('exist')),
#     'input_unit': dict(sysA=A1),
#     'liq_unit': dict(sysA=A13),
#     'sol_unit': dict(sysA=A12),
#     'gas_unit': dict(sysA=None),
#     'stream_dct': dict(sysA=streamsA),
#     'TEA': dict(sysA=teaA),
#     'LCA': dict(sysA=lcaA),
#     'cache': dict(sysA={}),
#     }

# def cache_recoveries(sys):
#     total_COD = get_total_inputs(sys_dct['input_unit'][sys.ID])['COD']
#     ppl = sys_dct['ppl'][sys.ID]
#     if sys_dct['gas_unit'][sys.ID]:
#         gas_mol = sys_dct['gas_unit'][sys.ID].outs[0].imol['CH4']
#         gas_COD = gas_mol*1e3*get_biogas_energy()*365*24/14e3/ppl/total_COD
#         # breakpoint()
#     else:
#         gas_COD = 0
#     cache = {
#         'liq': get_recovery(ins=sys_dct['input_unit'][sys.ID],
#                             outs=sys_dct['liq_unit'][sys.ID].ins,
#                             ppl=ppl),
#         'sol': get_recovery(ins=sys_dct['input_unit'][sys.ID],
#                             outs=sys_dct['sol_unit'][sys.ID].ins,
#                             ppl=ppl),
#         'gas': dict(COD=gas_COD, N=0, P=0, K=0)
#         }
#     return cache

# def update_cache(sys):
#     last_u = sys.path[-1]
#     last_u._run()
#     sys_dct['cache'][sys.ID] = cache_recoveries(sys)

# A13.specification = lambda: update_cache(sysA)
# B15.specification = lambda: update_cache(sysB)
# C13.specification = lambda: update_cache(sysC)


# def get_summarizing_fuctions():
#     func_dct = {}
#     func_dct['get_annual_cost'] = lambda tea, ppl: tea.EAC/ppl
#     func_dct['get_annual_CAPEX'] = lambda tea, ppl: tea.annualized_CAPEX/ppl
#     func_dct['get_annual_OPEX'] = lambda tea, ppl: tea.AOC/ppl
#     ind = 'GlobalWarming'
#     func_dct['get_annual_GWP'] = \
#         lambda lca, ppl: lca.total_impacts[ind]/lca.lifetime/ppl
#     func_dct['get_constr_GWP'] = \
#         lambda lca, ppl: lca.total_construction_impacts[ind]/lca.lifetime/ppl
#     func_dct['get_trans_GWP'] = \
#         lambda lca, ppl: lca.total_transportation_impacts[ind]/lca.lifetime/ppl  
#     func_dct['get_direct_emission_GWP'] = \
#         lambda lca, ppl: lca.get_stream_impacts(stream_items=lca.stream_inventory, kind='direct_emission')[ind] \
#             /lca.lifetime/ppl
#     func_dct['get_offset_GWP'] = \
#         lambda lca, ppl: lca.get_stream_impacts(stream_items=lca.stream_inventory, kind='offset')[ind] \
#             /lca.lifetime/ppl
#     func_dct['get_other_GWP'] = \
#         lambda lca, ppl: lca.total_other_impacts[ind]/lca.lifetime/ppl
#     for i in ('COD', 'N', 'P', 'K'):
#         func_dct[f'get_liq_{i}_recovery'] = \
#             lambda sys, i: sys_dct['cache'][sys.ID]['liq'][i]
#         func_dct[f'get_sol_{i}_recovery'] = \
#             lambda sys, i: sys_dct['cache'][sys.ID]['sol'][i]
#         func_dct[f'get_gas_{i}_recovery'] = \
#             lambda sys, i: sys_dct['cache'][sys.ID]['gas'][i]
#         func_dct[f'get_tot_{i}_recovery'] = \
#             lambda sys, i: \
#                 sys_dct['cache'][sys.ID]['liq'][i] + \
#                 sys_dct['cache'][sys.ID]['sol'][i] + \
#                 sys_dct['cache'][sys.ID]['gas'][i]
#     return func_dct


# def print_summaries(systems):
#     try: iter(systems)
#     except: systems = (systems, )
#     func = get_summarizing_fuctions()
#     for sys in systems:
#         sys.simulate()
#         ppl = sys_dct['ppl'][sys.ID]
#         print(f'\n---------- Summary for {sys} ----------\n')
#         tea = sys_dct['TEA'][sys.ID]
#         tea.show()
#         print('\n')
#         lca = sys_dct['LCA'][sys.ID]
#         lca.show()
        
#         unit = f'{currency}/cap/yr'
#         print(f'\nNet cost: {func["get_annual_cost"](tea, ppl):.1f} {unit}.')
#         print(f'Capital: {func["get_annual_CAPEX"](tea, ppl):.1f} {unit}.')
#         print(f'Operating: {func["get_annual_OPEX"](tea, ppl):.1f} {unit}.')
        
#         unit = f'{GWP.unit}/cap/yr'
#         print(f'\nNet emission: {func["get_annual_GWP"](lca, ppl):.1f} {unit}.')
#         print(f'Construction: {func["get_constr_GWP"](lca, ppl):.1f} {unit}.')
#         print(f'Transportation: {func["get_trans_GWP"](lca, ppl):.1f} {unit}.')
#         print(f'Direct emission: {func["get_direct_emission_GWP"](lca, ppl):.1f} {unit}.')
#         print(f'Offset: {func["get_offset_GWP"](lca, ppl):.1f} {unit}.')
#         print(f'Other: {func["get_other_GWP"](lca, ppl):.1} {unit}.\n')

#         for i in ('COD', 'N', 'P', 'K'):
#             print(f'Total {i} recovery is {func[f"get_tot_{i}_recovery"](sys, i):.1%}, '
#                   f'{func[f"get_liq_{i}_recovery"](sys, i):.1%} in liquid, '
#                   f'{func[f"get_sol_{i}_recovery"](sys, i):.1%} in solid, '
#                   f'{func[f"get_gas_{i}_recovery"](sys, i):.1%} in gas.')

# def save_all_reports():
#     import os
#     path = os.path.dirname(os.path.realpath(__file__))
#     path += '/results'
#     if not os.path.isdir(path):
#         os.path.mkdir(path)
#     del os
#     for i in (sysA, sysB, sysC, lcaA, lcaB, lcaC):
#         if isinstance(i, bst.System):
#             i.simulate()
#             i.save_report(f'{path}/{i.ID}.xlsx')
#         else:
#             i.save_report(f'{path}/{i.system.ID}_lca.xlsx')

# __all__ = ('sysA', 'sysB', 'sysC', 'teaA', 'teaB', 'teaC', 'lcaA', 'lcaB', 'lcaC',
#            'print_summaries', 'save_all_reports',
#            *(i.ID for i in sysA.units),
#            *(i.ID for i in sysB.units),
#            *(i.ID for i in sysC.units),
#            )



