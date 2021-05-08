#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
EXPOsan: Exposition of sanitation and resource recovery systems

This module is developed by:
    Yalin Li <zoe.yalin.li@gmail.com>

This module is modified for Biogenic Refinery by:
    Lewis Rowles <stetsonsc@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/EXPOsan/blob/main/LICENSE.txt
for license details.
'''


# %%

import numpy as np
import pandas as pd
from chaospy import distributions as shape
from thermosteam.functional import V_to_rho, rho_to_V
from biosteam import PowerUtility
from biosteam.evaluation import Model, Metric
from qsdsan import currency, ImpactItem
from qsdsan.utils import (
    load_data, data_path,
    AttrSetter, AttrFuncSetter, DictAttrSetter,
    FuncGetter,
    time_printer
    )
from exposan import biogenic_refinery as br

getattr = getattr
eval = eval

item_path = br.systems.item_path

__all__ = ('modelA', 'modelB', 'result_dct',
           'run_uncertainty', 'save_uncertainty_results')


# %%

# =============================================================================
# Functions for batch-making metrics and -setting parameters
# =============================================================================

systems = br.systems
sys_dct = systems.sys_dct
price_dct = systems.price_dct
GWP_dct = systems.GWP_dct
GWP = systems.GWP
get_summarizing_fuctions = systems.get_summarizing_fuctions
func = get_summarizing_fuctions()

def add_metrics(system):
    sys_ID = system.ID
    tea = sys_dct['TEA'][sys_ID]
    lca = sys_dct['LCA'][sys_ID]
    ppl = sys_dct['ppl'][sys_ID]
    unit = f'{currency}/cap/yr'
    cat = 'TEA results'
    metrics = [
        Metric('Net cost', lambda: func['get_annual_cost'](tea, ppl), unit, cat),
        Metric('Annual CAPEX', lambda: func['get_annual_CAPEX'](tea, ppl), unit, cat),
        Metric('Annual OPEX', lambda: func['get_annual_OPEX'](tea, ppl), unit, cat),
        ]
    unit = f'{GWP.unit}/cap/yr'
    cat = 'LCA results'
    metrics.extend([
        Metric('Net emission', lambda: func['get_annual_GWP'](lca, ppl), unit, cat),
        Metric('Construction', lambda: func['get_constr_GWP'](lca, ppl), unit, cat),
        Metric('Transportation', lambda: func['get_trans_GWP'](lca, ppl), unit, cat),
        Metric('Direct emission', lambda: func['get_direct_emission_GWP'](lca, ppl), unit, cat),
        Metric('Offset', lambda: func['get_offset_GWP'](lca, ppl), unit, cat),
        Metric('Other', lambda: func['get_other_GWP'](lca, ppl), unit, cat),
        ])
    for i in ('COD', 'N', 'P', 'K'):
        cat = f'{i} recovery'
        metrics.extend([
            Metric(f'Liquid {i}', FuncGetter(func[f'get_liq_{i}_recovery'], (system, i)), '', cat),
            Metric(f'Solid {i}', FuncGetter(func[f'get_sol_{i}_recovery'], (system, i)), '', cat),
            Metric(f'Gas {i}', FuncGetter(func[f'get_gas_{i}_recovery'], (system, i)), '', cat),
            Metric(f'Total {i}', FuncGetter(func[f'get_tot_{i}_recovery'], (system, i)), '', cat)
            ])
    return metrics


def batch_setting_unit_params(df, model, unit, exclude=()):
    for para in df.index:
        if para in exclude: continue
        b = getattr(unit, para)
        lower = float(df.loc[para]['low'])
        upper = float(df.loc[para]['high'])
        dist = df.loc[para]['distribution']
        if dist == 'uniform':
            D = shape.Uniform(lower=lower, upper=upper)
        elif dist == 'triangular':
            D = shape.Triangle(lower=lower, midpoint=b, upper=upper)
        elif dist == 'constant': continue
        else:
            raise ValueError(f'Distribution {dist} not recognized for unit {unit}.')     
        model.parameter(setter=AttrSetter(unit, para),
                        name=para, element=unit, kind='coupled', units=df.loc[para]['unit'],
                        baseline=b, distribution=D)


# %%

# =============================================================================
# Shared by all three systems
# =============================================================================

su_data_path = data_path + 'sanunit_data/'
path = su_data_path + '_drying_bed.tsv'
drying_bed_data = load_data(path)

def add_shared_parameters(sys, model, drying_bed_unit, crop_application_unit):
    ########## Related to multiple units ##########
    unit = sys.path[0]
    param = model.parameter
    
    
    ########## Related to human input ##########
    # Diet and excretion
    path = data_path + 'sanunit_data/_excretion.tsv'
    data = load_data(path)
    batch_setting_unit_params(data, model, unit)
    
    # Household size
    b = systems.get_household_size()
    D = shape.Normal(mu=b, sigma=1.8)
    @param(name='Household size', element=unit, kind='coupled', units='cap/household',
           baseline=b, distribution=D)
    def set_household_size(i):
        systems.household_size = max(1, i)
    
    # Toilet density
    b = systems.get_household_per_toilet()
    D = shape.Uniform(lower=3, upper=5)
    @param(name='Toilet density', element=unit, kind='coupled', units='household/toilet',
           baseline=b, distribution=D)
    def set_toilet_density(i):
        systems.household_per_toilet = i

    ##### Universal degradation parameters #####
    # Max methane emission
    unit = sys.path[1] # the first unit that involves degradation
    b = systems.get_max_CH4_emission()
    D = shape.Triangle(lower=0.175, midpoint=b, upper=0.325)
    @param(name='Max CH4 emission', element=unit, kind='coupled', units='g CH4/g COD',
           baseline=b, distribution=D)
    def set_max_CH4_emission(i):
        systems.max_CH4_emission = i
    
    # Time to full degradation
    b = systems.tau_deg
    D = shape.Uniform(lower=1, upper=3)
    @param(name='Full degradation time', element=unit, kind='coupled', units='yr',
           baseline=b, distribution=D)
    def set_tau_deg(i):
        systems.tau_deg = i
    
    # Reduction at full degradation
    b = systems.log_deg
    D = shape.Uniform(lower=2, upper=4)
    @param(name='Log degradation', element=unit, kind='coupled', units='-',
           baseline=b, distribution=D)
    def set_log_deg(i):
        systems.log = i
    
    ##### Toilet material properties #####
    density = unit.density_dct
    b = density['Plastic']
    D = shape.Uniform(lower=0.31, upper=1.24)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Plastic'),
          name='Plastic density', element=unit, kind='isolated', units='kg/m2',
          baseline=b, distribution=D)
    
    b = density['Brick']
    D = shape.Uniform(lower=1500, upper=2000)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Brick'),
          name='Brick density', element=unit, kind='isolated', units='kg/m3',
          baseline=b, distribution=D)
    
    b = density['StainlessSteelSheet']
    D = shape.Uniform(lower=2.26, upper=3.58)
    param(setter=DictAttrSetter(unit, 'density_dct', 'StainlessSteelSheet'),
          name='SS sheet density', element=unit, kind='isolated', units='kg/m2',
          baseline=b, distribution=D)
        
    b = density['Gravel']
    D = shape.Uniform(lower=1520, upper=1680)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Gravel'),
          name='Gravel density', element=unit, kind='isolated', units='kg/m3',
          baseline=b, distribution=D)

    b = density['Sand']
    D = shape.Uniform(lower=1281, upper=1602)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Sand'),
          name='Sand density', element=unit, kind='isolated', units='kg/m3',
          baseline=b, distribution=D)
        
    b = density['Steel']
    D = shape.Uniform(lower=7750, upper=8050)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Steel'),
          name='Steel density', element=unit, kind='isolated', units='kg/m3',
          baseline=b, distribution=D)

    # ########## Drying bed ##########
    # unit = drying_bed_unit
    # D = shape.Uniform(lower=0, upper=0.1)
    # batch_setting_unit_params(drying_bed_data, model, unit, exclude=('sol_frac', 'bed_H'))
    
    # b = unit.sol_frac
    # if unit.design_type == 'unplanted':
    #     D = shape.Uniform(lower=0.3, upper=0.4)
    # elif unit.design_type == 'planted':
    #     D = shape.Uniform(lower=0.4, upper=0.7)
    # param(setter=DictAttrSetter(unit, '_sol_frac', getattr(unit, 'design_type')),
    #       name='sol_frac', element=unit, kind='coupled', units='fraction',
    #       baseline=b, distribution=D)
    
    # b = unit.bed_H['covered']
    # D = shape.Uniform(lower=0.45, upper=0.75)
    # param(setter=DictAttrSetter(unit, 'bed_H', ('covered', 'uncovered')),
    #       name='non_storage_bed_H', element=unit, kind='coupled', units='m',
    #       baseline=b, distribution=D)
    
    # b = unit.bed_H['storage']
    # D = shape.Uniform(lower=1.2, upper=1.8)
    # param(DictAttrSetter(unit, 'bed_H', 'storage'),
    #       name='storage_bed_H', element=unit, kind='coupled', units='m',
    #       baseline=b, distribution=D)

    # ########## Crop application ##########
    # unit = crop_application_unit
    # D = shape.Uniform(lower=0, upper=0.1)
    # param(setter=DictAttrSetter(unit, 'loss_ratio', 'NH3'),
    #       name='NH3 application loss', element=unit, kind='coupled',
    #       units='fraction of applied', baseline=0.05, distribution=D)
    
    # # Mg, Ca, C actually not affecting results
    # D = shape.Uniform(lower=0, upper=0.05)
    # param(setter=DictAttrSetter(unit, 'loss_ratio', ('NonNH3', 'P', 'K', 'Mg', 'Ca')),
    #       name='Other application losses', element=unit, kind='coupled',
    #       units='fraction of applied', baseline=0.02, distribution=D)

    ######## General TEA settings ########
    # Discount factor for the excreta-derived fertilizers
    get_price_factor = systems.get_price_factor
    b = get_price_factor()
    D = shape.Uniform(lower=0.1, upper=0.4)
    @param(name='Price factor', element='TEA', kind='isolated', units='-',
           baseline=b, distribution=D)
    def set_price_factor(i):
        systems.price_factor = i
    
    D = shape.Uniform(lower=1.164, upper=2.296)
    @param(name='N fertilizer price', element='TEA', kind='isolated', units='USD/kg N',
           baseline=1.507, distribution=D)
    def set_N_price(i):
        price_dct['N'] = i * get_price_factor()
        
    D = shape.Uniform(lower=2.619, upper=6.692)
    @param(name='P fertilizer price', element='TEA', kind='isolated', units='USD/kg P',
           baseline=3.983, distribution=D)
    def set_P_price(i):
        price_dct['P'] = i * get_price_factor()
        
    D = shape.Uniform(lower=1.214, upper=1.474)
    @param(name='K fertilizer price', element='TEA', kind='isolated', units='USD/kg K',
           baseline=1.333, distribution=D)
    def set_K_price(i):
        price_dct['K'] = i * get_price_factor()
    
    # Money discount rate
    b = systems.get_discount_rate()
    D = shape.Uniform(lower=0.03, upper=0.06)
    @param(name='Discount rate', element='TEA', kind='isolated', units='fraction',
           baseline=b, distribution=D)
    def set_discount_rate(i):
        systems.discount_rate = i
    
    # Electricity price
    b = price_dct['Electricity']
    D = shape.Triangle(lower=0.08, midpoint=b, upper=0.21)
    @param(name='Electricity price', element='TEA', kind='isolated',
           units='$/kWh', baseline=b, distribution=D)
    def set_electricity_price(i):
        PowerUtility.price = i
    
    ######## General LCA settings ########
    b = GWP_dct['CH4']
    D = shape.Uniform(lower=28, upper=34)
    @param(name='CH4 CF', element='LCA', kind='isolated', units='kg CO2-eq/kg CH4',
           baseline=b, distribution=D)
    def set_CH4_CF(i):
        GWP_dct['CH4'] = i    
    
    b = GWP_dct['N2O']
    D = shape.Uniform(lower=265, upper=298)
    @param(name='N2O CF', element='LCA', kind='isolated', units='kg CO2-eq/kg N2O',
           baseline=b, distribution=D)
    def set_N2O_CF(i):
        GWP_dct['N2O'] = i

    b = GWP_dct['Electricity']
    D = shape.Uniform(lower=0.106, upper=0.121)
    @param(name='Electricity CF', element='LCA', kind='isolated',
           units='kg CO2-eq/kWh', baseline=b, distribution=D)
    def set_electricity_CF(i):
        GWP_dct['Electricity'] = i

    b = -GWP_dct['N']
    D = shape.Triangle(lower=1.8, midpoint=b, upper=8.9)
    @param(name='N fertilizer CF', element='LCA', kind='isolated',
           units='kg CO2-eq/kg N', baseline=b, distribution=D)
    def set_N_fertilizer_CF(i):
        GWP_dct['N'] = -i
        
    b = -GWP_dct['P']
    D = shape.Triangle(lower=4.3, midpoint=b, upper=5.4)
    @param(name='P fertilizer CF', element='LCA', kind='isolated',
           units='kg CO2-eq/kg P', baseline=b, distribution=D)
    def set_P_fertilizer_CF(i):
        GWP_dct['P'] = -i
        
    b = -GWP_dct['K']
    D = shape.Triangle(lower=1.1, midpoint=b, upper=2)
    @param(name='K fertilizer CF', element='LCA', kind='isolated',
           units='kg CO2-eq/kg K', baseline=b, distribution=D)
    def set_K_fertilizer_CF(i):
        GWP_dct['K'] = -i

    data = load_data(item_path, sheet='GWP')    
    for p in data.index:
        item = ImpactItem.get_item(p)
        b = item.CFs['GlobalWarming']
        lower = float(data.loc[p]['low'])
        upper = float(data.loc[p]['high'])
        dist = data.loc[p]['distribution']
        if dist == 'uniform':
            D = shape.Uniform(lower=lower, upper=upper)
        elif dist == 'triangular':
            D = shape.Triangle(lower=lower, midpoint=b, upper=upper)
        elif dist == 'constant': continue
        else:
            raise ValueError(f'Distribution {dist} not recognized.')
        model.parameter(name=p,
                        setter=DictAttrSetter(item, 'CFs', 'GlobalWarming'),
                        element='LCA', kind='isolated',
                        units=f'kg CO2-eq/{item.functional_unit}',
                        baseline=b, distribution=D)
    
    return model


# # =============================================================================
# # For the same processes in sysA and sysB
# # =============================================================================

path = su_data_path + '_toilet.tsv'
toilet_data = load_data(path)
path = su_data_path + '_pit_latrine.tsv'
pit_latrine_data = load_data(path)
MCF_lower_dct = eval(pit_latrine_data.loc['MCF_decay']['low'])
MCF_upper_dct = eval(pit_latrine_data.loc['MCF_decay']['high'])
N2O_EF_lower_dct = eval(pit_latrine_data.loc['N2O_EF_decay']['low'])
N2O_EF_upper_dct = eval(pit_latrine_data.loc['N2O_EF_decay']['high'])

def add_pit_latrine_parameters(sys, model):
    unit = sys.path[1]
    param = model.parameter
    ######## Related to the toilet ########
    data = pd.concat((toilet_data, pit_latrine_data))
    batch_setting_unit_params(data, model, unit, exclude=('MCF_decay', 'N2O_EF_decay'))

    # MCF and N2O_EF decay parameters, specified based on the type of the pit latrine
    b = unit.MCF_decay
    kind = unit._return_MCF_EF()
    D = shape.Triangle(lower=MCF_lower_dct[kind], midpoint=b, upper=MCF_upper_dct[kind])
    param(setter=DictAttrSetter(unit, '_MCF_decay', kind),
          name='MCF_decay', element=unit, kind='coupled',
          units='fraction of anaerobic conversion of degraded COD',
          baseline=b, distribution=D)
        
    b = unit.N2O_EF_decay
    D = shape.Triangle(lower=N2O_EF_lower_dct[kind], midpoint=b, upper=N2O_EF_upper_dct[kind])
    param(setter=DictAttrSetter(unit, '_N2O_EF_decay', kind),
          name='N2O_EF_decay', element=unit, kind='coupled',
          units='fraction of N emitted as N2O',
          baseline=b, distribution=D)

    # Costs
    b = unit.CAPEX
    D = shape.Uniform(lower=386, upper=511)
    param(setter=AttrSetter(unit, 'CAPEX'),
          name='Pit latrine capital cost', element=unit, kind='cost',
          units='USD', baseline=b, distribution=D)
        
    b = unit.OPEX_over_CAPEX
    D = shape.Uniform(lower=0.02, upper=0.08)
    param(setter=AttrSetter(unit, 'OPEX_over_CAPEX'),
          name='Pit latrine operating cost', element=unit, kind='cost',
          units='fraction of capital cost', baseline=b, distribution=D)
    
    ######## Related to conveyance ########
    unit = sys.path[2]
    b = unit.loss_ratio
    D = shape.Uniform(lower=0.02, upper=0.05)
    param(setter=AttrSetter(unit, 'loss_ratio'),
          name='Transportation loss', element=unit, kind='coupled', units='fraction',
          baseline=b, distribution=D)
    
    b = unit.single_truck.distance
    D = shape.Uniform(lower=2, upper=10)
    param(setter=AttrSetter(unit.single_truck, 'distance'),
          name='Transportation distance', element=unit, kind='coupled', units='km',
          baseline=b, distribution=D)
    
    b = systems.emptying_fee
    D = shape.Uniform(lower=0, upper=15)
    @param(name='Emptying fee', element=unit, kind='coupled', units='USD',
            baseline=b, distribution=D)
    def set_emptying_fee(i):
        systems.emptying_fee = i
    
    return model

# path = su_data_path + '_sludge_separator.tsv'
# sludge_separator_data = load_data(path)
# split_lower_dct = eval(sludge_separator_data.loc['split']['low'])
# split_upper_dct = eval(sludge_separator_data.loc['split']['high'])
# split_dist_dct = eval(sludge_separator_data.loc['split']['distribution'])

# def add_sludge_separator_parameters(unit, model):
#     param = model.parameter
    
#     b = unit.settled_frac
#     D = shape.Uniform(lower=0.1, upper=0.2)
#     @param(name='Settled frac', element=unit, kind='coupled', units='fraction',
#            baseline=b, distribution=D)
#     def set_settled_frac(i):
#         unit.settled_frac = i
    
#     for key in split_lower_dct.keys():
#         b = getattr(unit, 'split')[key]
#         lower = split_lower_dct[key]
#         upper = split_upper_dct[key]
#         dist = split_dist_dct[key]
#         if dist == 'uniform':
#             D = shape.Uniform(lower=lower, upper=upper)
#         elif dist == 'triangular':
#             D = shape.Triangle(lower=lower, midpoint=b, upper=upper)
#         param(setter=DictAttrSetter(unit, 'split', key),
#               name='Frac of settled'+key, element=unit, kind='coupled',
#               units='fraction',
#               baseline=b, distribution=D)
    
#     return model

# def add_lagoon_parameters(unit, model):
#     param = model.parameter
#     b = systems.get_sewer_flow()
#     D = shape.Uniform(lower=2500, upper=3000)
#     @param(name='Sewer flow', element=unit, kind='coupled', units='m3/d',
#            baseline=b, distribution=D)
#     def set_sewer_flow(i):
#         systems.sewer_flow = i
#     return model

# def add_existing_plant_parameters(toilet_unit, cost_unit, tea, model):
#     param = model.parameter
#     b = systems.ppl_exist_sewer
#     D = shape.Uniform(lower=3e4, upper=5e4)
#     @param(name='Sewer ppl', element=toilet_unit, kind='coupled', units='-',
#            baseline=b, distribution=D)
#     def set_sewer_ppl(i):
#         systems.ppl_exist_sewer = i
        
#     b = systems.ppl_exist_sludge
#     D = shape.Triangle(lower=416667, midpoint=b, upper=458333)
#     @param(name='Sludge ppl', element=toilet_unit, kind='coupled', units='-',
#            baseline=b, distribution=D)
#     def set_sludge_ppl(i):
#         systems.ppl_exist_sludge = i
    
#     b = cost_unit.lifetime
#     D = shape.Triangle(lower=8, midpoint=b, upper=11)
#     param(setter=AttrSetter(cost_unit, 'lifetime'),
#           name='Plant lifetime', element='TEA/LCA', kind='isolated', units='yr',
#           baseline=b, distribution=D)
    
#     b = tea.annual_labor
#     D = shape.Uniform(lower=1e6, upper=5e6)
#     param(setter=AttrFuncSetter(tea, 'annual_labor',
#                                 lambda salary: salary*12*12),
#           name='Staff salary', element='TEA', kind='isolated', units='USD',
#           baseline=b, distribution=D)

#     return model


# %%

# =============================================================================
# Scenario A (sysA)
# =============================================================================

sysA = systems.sysA
sysA.simulate()
modelA = Model(sysA, add_metrics(sysA))
paramA = modelA.parameter

# Shared parameters

# Pit latrine and conveyance
modelA = add_pit_latrine_parameters(sysA, modelA)

# Screw press
A4 = systems.A4
path = su_data_path + '_screw_press.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelA, A4)


# Carbonizer base
A5 = systems.A5
path = su_data_path + '_carbonizer_base.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelA, A5)

# Pollution control device
A6 = systems.A6
path = su_data_path + '_pollution_control_device.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelA, A6)

# Oil heat exchanger
A7 = systems.A7
path = su_data_path + '_oil_heat_exchanger.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelA, A7)

# Hydronic heat exchanger
A8 = systems.A8
path = su_data_path + '_hydronic_heat_exchanger.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelA, A8)

# Dryer from HHx
A9 = systems.A9
path = su_data_path + '_dryer_from_hhx.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelA, A9)

all_paramsA = modelA.get_parameters()


# %%

# =============================================================================
# Scenario B (sysB)
# =============================================================================

sysB = systems.sysB
sysB.simulate()
modelB = Model(sysB, add_metrics(sysB))
paramB = modelB.parameter


# Shared parameters

# Pit latrine and conveyance
B2 = systems.B2
path = su_data_path + '_uddt.tsv'
uddt_data = load_data(path)
data = pd.concat((toilet_data, uddt_data))
batch_setting_unit_params(data, modelB, B2)

b = B2.CAPEX
D = shape.Uniform(lower=476, upper=630)
@paramB(name='UDDT capital cost', element=B2, kind='cost',
       units='USD', baseline=b, distribution=D)
def set_UDDT_CAPEX(i):
    B2.CAPEX = i
    
b = B2.OPEX_over_CAPEX
D = shape.Uniform(lower=0.05, upper=0.1)
@paramB(name='UDDT operating cost', element=B2, kind='cost',
       units='fraction of capital cost', baseline=b, distribution=D)
def set_UDDT_OPEX(i):
    B2.OPEX_over_CAPEX = i

# Conveyance
B3 = systems.B3
B4 = systems.B4
b = B3.loss_ratio
D = shape.Uniform(lower=0.02, upper=0.05)
@paramB(name='Transportation loss', element=B3, kind='coupled', units='fraction',
       baseline=b, distribution=D)
def set_trans_loss(i):
    B3.loss_ratio = B4.loss_ratio = i

b = B3.single_truck.distance
D = shape.Uniform(lower=2, upper=10)
@paramB(name='Transportation distance', element=B3, kind='coupled', units='km',
       baseline=b, distribution=D)
def set_trans_distance(i):
    B3.single_truck.distance = B4.single_truck.distance = i

b = systems.handcart_fee
D = shape.Uniform(lower=0.004, upper=0.015)
@paramB(name='Handcart fee', element=B3, kind='cost', units='USD',
       baseline=b, distribution=D)
def set_handcart_fee(i):
    systems.handcart_fee = i

b = systems.truck_fee
D = shape.Uniform(lower=4.82, upper=8.5)
@paramB(name='Truck fee', element=B3, kind='cost', units='USD',
       baseline=b, distribution=D)
def set_truck_fee(i):
    systems.truck_fee = i

# Struvite precipitation
B5 = systems.B5
path = su_data_path + '_struvite_precipitation.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelB, B5)


# Ion exchange NH3
B6 = systems.B6
path = su_data_path + '_ion_exchange_NH3.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelB, B6)

# Grinder
B7 = systems.B7
path = su_data_path + '_grinder.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelB, B7)

# Carbonizer base
B8 = systems.B8
path = su_data_path + '_carbonizer_base.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelB, B8)

# Pollution control device
B9 = systems.B9
path = su_data_path + '_pollution_control_device.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelB, B9)

# Oil heat exchanger
B10 = systems.B10
path = su_data_path + '_oil_heat_exchanger.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelB, B10)

# Hydronic heat exchanger
B11 = systems.B11
path = su_data_path + '_hydronic_heat_exchanger.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelB, B11)

# Dryer from HHX
B12 = systems.B12
path = su_data_path + '_dryer_from_hhx.tsv'
data = load_data(path)
batch_setting_unit_params(data, modelB, B12)


all_paramsB = modelB.get_parameters()




# %%

# =============================================================================
# Functions to run simulation and generate plots
# =============================================================================

result_dct = {
        'sysA': dict.fromkeys(('parameters', 'data', 'percentiles', 'spearman')),
        'sysB': dict.fromkeys(('parameters', 'data', 'percentiles', 'spearman')),
        }

@time_printer
def run_uncertainty(model, seed=None, N=1000, rule='L',
                    percentiles=(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1),
                    print_time=False):
    global result_dct
    if seed:
        np.random.seed(seed)

    samples = model.sample(N, rule)
    model.load_samples(samples)
    model.evaluate()

    # Data organization
    dct = result_dct[model._system.ID]
    index_p = len(model.get_parameters())
    dct['parameters'] = model.table.iloc[:, :index_p].copy()
    dct['data'] = model.table.iloc[:, index_p:].copy()
    if percentiles:
        dct['percentiles'] = dct['data'].quantile(q=percentiles)

    # Spearman's rank correlation
    spearman_metrics = [model.metrics[i] for i in (0, 3, 12, 16, 20, 24)]
    spearman_results = model.spearman(model.get_parameters(), spearman_metrics)
    spearman_results.columns = pd.Index([i.name_with_units for i in spearman_metrics])
    dct['spearman'] = spearman_results
    return dct

def save_uncertainty_results(model, path=''):
    if not path:
        import os
        path = os.path.dirname(os.path.realpath(__file__))
        path += '/results'
        if not os.path.isdir(path):
            os.mkdir(path)
        path += f'/model{model._system.ID[-1]}.xlsx'
        del os
    elif not (path.endswith('xlsx') or path.endswith('xls')):
        extension = path.split('.')[-1]
        raise ValueError(f'Only "xlsx" and "xls" are supported, not {extension}.')
    
    dct = result_dct[model._system.ID]
    if dct['parameters'] is None:
        raise ValueError('No cached result, run model first.')
    with pd.ExcelWriter(path) as writer:
        dct['parameters'].to_excel(writer, sheet_name='Parameters')
        dct['data'].to_excel(writer, sheet_name='Uncertainty results')
        if 'percentiles' in dct.keys():
            dct['percentiles'].to_excel(writer, sheet_name='Percentiles')
        dct['spearman'].to_excel(writer, sheet_name='Spearman')
        model.table.to_excel(writer, sheet_name='Raw data')



