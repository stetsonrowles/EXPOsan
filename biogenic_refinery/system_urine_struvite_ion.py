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
from qsdsan import WasteStream, ImpactIndicator, ImpactItem, StreamImpactItem, SimpleTEA, LCA, SanStream

#from qsdsan.systems import bwaise as bw

from _cmps_Biogenic_Refinery import cmps

print(qs.__version__)

#import all units
from _ion_exchange_NH3 import IonExchangeNH3


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


# =============================================================================
# Prices and GWP CFs
# =============================================================================

# Recycled nutrients are sold at a lower price than commercial fertilizers

price_factor = 0.25
get_price_factor = lambda: price_factor

operator_daily_wage = 29
get_operator_daily_wage = lambda: operator_daily_wage

price_dct = {
    'Electricity': 0.06,
    'Concrete': 194,
    'Steel': 2.665,
    'N': 1.507*get_price_factor(),
    'P': 3.983*get_price_factor(),
    'K': 1.333*get_price_factor(),
    'Polymer': 5*get_price_factor(),
    'Resin': 3.335*get_price_factor(),
    }


GWP_dct = {
    'Electricity': 0.15,
    'CH4': 28,
    'N2O': 265,
    'N': -5.4,
    'P': -4.9,
    'K': -1.5,
    'Polymer': 2.8,
    'Resin': 1.612,
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
polymer_item = ImpactItem(ID='polymer_item', functional_unit='kg', GWP=GWP_dct['Polymer'])
resin_item = ImpactItem(ID='resin_item', functional_unit='kg', GWP=GWP_dct['Resin'])

polymer = SanStream('polymer', phase='s', units = 'kg/hr', price=price_dct['Polymer'], impact_item=polymer_item.copy(set_as_source=True))


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
# Scenario A (sysA): UDDT with Biogenic Refinery
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
             OPEX_over_CAPEX=0.1,
             decay_k_COD=get_decay_k(tau_deg, log_deg),
             decay_k_N=get_decay_k(tau_deg, log_deg),
             max_CH4_emission=get_max_CH4_emission())
 
##################### Conveyance of Waste #####################
# Liquid waste
handcart_fee = 0.01 # USD/cap/d
get_handcart_fee = lambda: handcart_fee
truck_fee = 6.21 # UGX/m3
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
B5 = IonExchangeNH3('B4', ins=(B3-0, 'resin_in', 'H2SO4'), outs=('treated', 'resin_out', 'conc_NH3'))


B9 = su.Mixer('B9', ins=(B2-4), outs=streamsB['CH4'])
B9.specification = lambda: add_fugitive_items(B9, CH4_item)
B9.line = 'fugitive CH4 mixer' 
        
B10 = su.Mixer('B10', ins=(B2-5), outs=streamsB['N2O'])
B10.specification = lambda: add_fugitive_items(B10, N2O_item)
B10.line = 'fugitive N2O mixer'



################## Reuse or Disposal ##################
#!!! add conveyance of biochar and crop application 



############### Simulation, TEA, and LCA ###############
sysB = bst.System('sysB', path= (B1, B2, B3, B4, B5, B9, B10))

sysB.simulate()

power = sum([u.power_utility.rate for u in sysB.units])

#!!! update labor to input country specific data and be a distribution
teaB = SimpleTEA(system=sysB, discount_rate=get_discount_rate(), 
                  start_year=2020, lifetime=20, uptime_ratio=1, 
                  lang_factor=None, annual_maintenance=0, 
                  annual_labor=(get_operator_daily_wage() * 0*365), construction_schedule=None)

lcaB = LCA(system=sysB, lifetime=20, lifetime_unit='yr', uptime_ratio=1,
            e_item=lambda: power*(365*24)*20)
