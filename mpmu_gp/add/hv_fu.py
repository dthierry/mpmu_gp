# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax import grad, jacfwd

__author__ = "David Thierry @dthierry"

rho_fact = 1e-3  # kg/L / kg/m3
# kg/m3
std_den = {
    0: 0.7, # ng
    1: 0.0899  # h2
}
# -> kg/L
std_den = {k:std_den[k]*rho_fact for k in std_den.keys()}


"""
For natural gas, according to
https://h2tools.org/hyarc/calculator-tools/lower-and-higher-heating-values-fuels
Hydrogen:
LHV = 119.96 MJ/kg
HHV = 141.88 MJ/kg

Natural gas:
LHV = 47.13 MJ/kg
HHV = 52.21 MJ/kg
"""

hhv_fact = 1_055.055853  # MJ/MMBTU
# MJ/kg -> MMBTU
hhv = {0: 52.2,
       1: 141.7}
# in MMBTU/kg
hhv = {k: hhv[k]/hhv_fact for k in hhv.keys()}

def mass_conversion(x):
    """returns (residual) kg/h"""
    ng_slpm = x[0]
    h2_slpm = x[1]
    ng_kg = x[2]
    h2_kg = x[3]
    # kg/h, 60 at the end is min to hr
    ng_kg_ = ng_slpm * std_den[0] * 60.
    h2_kg_ = h2_slpm * std_den[1] * 60.
    # return residual
    return jnp.array([ng_kg_ - ng_kg, h2_kg_ - h2_kg])

j_mass_conv = jacfwd(mass_conversion)

def heat_conversion(x):
    """returns (residual) MMBTU/h"""
    ng_kg = x[0]
    h2_kg = x[1]
    ng_h = x[2]
    h2_h = x[3]
    #
    ng_h_ = ng_kg * hhv[0]
    h2_h_ = h2_kg * hhv[1]
    # residual
    return jnp.array([ng_h_ - ng_h, h2_h_ - h2_h])

j_heat_conv = jacfwd(heat_conversion)

def jac_mass_conv(x):
    j = j_mass_conv(x)
    row0 = jnp.array([j[0,0], -1])
    row1 = jnp.array([j[1,1], -1])
    return [row0, row1]

def jac_heat_conv(x):
    j = j_heat_conv(x)
    row0 = jnp.array([j[0,0], -1])
    row1 = jnp.array([j[1,1], -1])
    return [row0, row1]


