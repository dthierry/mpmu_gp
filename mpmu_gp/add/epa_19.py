# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax import grad, jacfwd

# stream of computations
# 0: compute average mw of fuel (ng + h2)
# 1: compute mass fractions
# 2: compute Xs
# 3: compute GCV
# 4: fd

# conversion factor 1 MWh = 3.4095106405145 MMBTU
mmbtuToMwh = 3.4095106405145

# f0
ch4 = {"name":"Methane",
       "symbol":"CH4",
       "C":1., "H":4., "O":0.,"N":0.,
       "MW":16.04,"y_ng":0.949, "y_h": 0.}
# f1
c2h6 = {"name":"Ethane",
        "symbol":"C2H6",
        "C":2., "H":6., "O":0.,"N":0,
        "MW":30.07, "y_ng":0.025, "y_h": 0.}

# f2
c3h8 = {"name":"Propane",
        "symbol":"C3H8",
        "C":3., "H":8., "O":0.,"N":0.,
        "MW":44.1, "y_ng":0.003, "y_h": 0.}

# f3
n2 = {"name":"Nitrogen",
      "symbol":"N2",
      "C":0., "H":0., "O":0.,"N":2,
      "MW":28.02, "y_ng":0.016, "y_h": 0.}

# f4
co2 = {"name":"Carbon Dioxide",
       "symbol":"CO2",
       "C":1., "H":0., "O":2.,"N":0.,
       "MW":44.01, "y_ng":0.007, "y_h": 0.}

# f5
h2 = {"name":"Hydrogen",
      "symbol":"H2",
      "C":0., "H":2., "O":0.,"N":0.,
      "MW":2.016, "y_ng":0.0, "y_h": 1.}

f_ = [ch4, c2h6, c3h8, n2, co2, h2]
f_ng = [ch4, c2h6, c3h8, n2, co2]
# composition of natural gas
K = 1.

# molecular weight
mw_ = jnp.array([i["MW"] for i in f_])
mw_ng = jnp.array([i["MW"] for i in f_ng])

# composition of ng and h2 (param)
fmf_0_ = jnp.array([[i["y_ng"], i["y_h"]] for i in f_])
fmf_0_ng = jnp.array([i["y_ng"] for i in f_ng])

# atoms (param)
atom_n0 = jnp.array([[i["C"] for i in f_],
                    [i["H"] for i in f_],
                    [i["O"] for i in f_]])

#
def c0_avg_mw_in(x):
    """R^2 -> R"""
    # R2 -> R
    # returns the residual
    xh2 = x[0]
    avg_mw_f = x[1]

    xng = 1.-(xh2/100.)
    #return sum(sp["MW"] for sp in f_ng)*xng \
    return jnp.dot(mw_ng, fmf_0_ng)*xng \
        + h2["MW"] * xh2/100. \
        - avg_mw_f

j0_avg_mw_in = jacfwd(c0_avg_mw_in)

#
def c1_mass_f_ng(x):
    """R^2 -> R^5
    mass fraction of the fuel components"""
    xh2 = x[0]
    avg_mw_f = x[1]
    mass_f_ng_i = x[2:]
    return jnp.multiply(fmf_0_ng, mw_ng/avg_mw_f)*(1. - xh2/100.) \
        - mass_f_ng_i

#return fmf_0_ng * (1. - xh2/100.) \
#return jnp.multiply(fmf_0_ng, jnp.ones(5)) * (1. - xh2/100.) \
j1_mass_f_ng = jacfwd(c1_mass_f_ng)

def c2_mass_f_h2(x):
    """R^6 -> R
    h2 mass fraction"""
    mass_f_ng_i = x[0:5]
    mass_f_h2 = x[5]
    return 1.0 - jnp.sum(mass_f_ng_i) - mass_f_h2

j2_mass_f_h2 = jacfwd(c2_mass_f_h2)

atom_C = jnp.array([atm["C"] for atm in f_])
atom_H = jnp.array([atm["H"] for atm in f_])
atom_O = jnp.array([atm["O"] for atm in f_])
atom_N = jnp.array([atm["N"] for atm in f_])

# c h o n
molar_mass_atom = jnp.array([12.01070, 1.00794, 15.9994, 14.0067])

# c h o n
def c3_mass_c(x):
    """R^5 -> R
    h mass"""
    # four atoms with c
    mass_f_ng_ch4 = x[0]
    mass_f_ng_c2h6 = x[1]
    mass_f_ng_c3h8 = x[2]
    mass_f_ng_co2 = x[3]
    m_c = x[4]
    return (mass_f_ng_ch4 * atom_C[0] \
            + mass_f_ng_c2h6 * atom_C[1] \
            + mass_f_ng_c3h8 * atom_C[2] \
            + mass_f_ng_co2 * atom_C[4]) * molar_mass_atom[0] - m_c

j3_mass_c = jacfwd(c3_mass_c)


def c4_mass_h(x):
    """R^5 -> R
    c mass
    """
    # four atoms with h
    mass_f_ng_ch4 = x[0]
    mass_f_ng_c2h6 = x[1]
    mass_f_ng_c3h8 = x[2]
    mass_f_h2 = x[3]
    m_h = x[4]
    return (mass_f_ng_ch4 * atom_H[0]
            + mass_f_ng_c2h6 * atom_H[1]
            + mass_f_ng_c3h8 * atom_H[2]
            + mass_f_h2 * 2.0) * molar_mass_atom[1] - m_h

j4_mass_h = jacfwd(c4_mass_h)


def c5_mass_o(x):
    """R^2 -> R
    o mass
    """
    mass_f_ng_co2 = x[0]
    m_o = x[1]
    return mass_f_ng_co2*atom_O[4] * molar_mass_atom[2] - m_o

j5_mass_o = jacfwd(c5_mass_o)

def c6_mass_n(x):
    """R^2 -> R
    n mass
    """
    mass_f_ng_n2 = x[0]
    m_n = x[1]
    return mass_f_ng_n2*atom_N[3] * molar_mass_atom[3] - m_n

j6_mass_n = jacfwd(c6_mass_n)

#### ####
def c7_mass_frac_c(x):
    """R^5 -> R
    c atom mass frac
    """
    m_c = x[0]
    m_h = x[1]
    m_o = x[2]
    m_n = x[3]
    mf_c = x[4]
    return m_c / (m_c + m_h + m_o + m_n) - mf_c
j7_mass_frac_c = jacfwd(c7_mass_frac_c)

def c8_mass_frac_h(x):
    """R^5 -> R
    h atom mass frac
    """
    m_c = x[0]
    m_h = x[1]
    m_o = x[2]
    m_n = x[3]
    mf_h = x[4]
    return m_h / (m_c + m_h + m_o + m_n) - mf_h
j8_mass_frac_h = jacfwd(c8_mass_frac_h)

def c9_mass_frac_o(x):
    """R^5 -> R
    o atom mass frac
    """
    m_c = x[0]
    m_h = x[1]
    m_o = x[2]
    m_n = x[3]
    mf_o = x[4]
    return m_o / (m_c + m_h + m_o + m_n) - mf_o
j9_mass_frac_o = jacfwd(c9_mass_frac_o)

def c10_mass_frac_n(x):
    """R^5 -> R
    n atom mass frac
    """
    m_c = x[0]
    m_h = x[1]
    m_o = x[2]
    m_n = x[3]
    mf_n = x[4]
    return m_n / (m_c + m_h + m_o + m_n) - mf_n
j10_mass_frac_n = jacfwd(c10_mass_frac_n)

# 80 ### #### #### #### #### #### #### #### # #### #### #### #### #### #### ####

def c11_gcv(x):
    """R^2 -> R
    """
    mass_f_h2 = x[0]
    gcv = x[1]
    return 22446.0 * (1.0-mass_f_h2) + 60920.0 * mass_f_h2 - gcv

j11_gcv = jacfwd(c11_gcv)


def c12_fd(x):
    """R^6 -> R
    we do multiply the molefracs by a 100
    """
    mf_c = x[0]  # a.k.a. Xc
    mf_h = x[1]  # a.k.a. Xh
    mf_o = x[2]  # a.k.a. Xo
    mf_n = x[3]  # a.k.a. Xn

    gcv = x[4]
    fd = x[5]
    return (3.64*mf_h*100.
            + 1.53*mf_c*100.
            + 0.14*mf_n*100.
            + 0.46*mf_o*100.)*1e+06/gcv \
        - fd

j12_fd = jacfwd(c12_fd)

def c13_cco(x):
    """R^2 -> R
    """
    yco = x[0] * 1e-06
    cco = x[1]
    return yco * 28.010 / (0.730240*528.0) - cco
    #return yco * 44.01 / (0.720240*528.0) - cco

j13_cco = jacfwd(c13_cco)

def c14_cnox(x):
    """R^2 -> R
    """
    ynox = x[0] * 1e-06
    cnox = x[1]
    return ynox * 46.0055 / (0.730240*528.0) - cnox

j14_cnox = jacfwd(c14_cnox)

def c15_eco(x):
    """R^4 -> R
    returns in lb/MWh
    """
    yo2 = x[0] * 1e-06 * 1e+02
    fd = x[1]
    cco = x[2]
    eco = x[3]
    return mmbtuToMwh * cco * fd * 20.9/(20.9-yo2) - eco

j15_eco = jacfwd(c15_eco)

def c16_enox(x):
    """R^4 -> R
    returns in lb/MWh
    """
    yo2 = x[0] * 1e-06 * 1e+02
    fd = x[1]
    cnox = x[2]
    enox = x[3]
    return mmbtuToMwh * cnox * fd * 20.9/(20.9-yo2) - enox

j16_enox = jacfwd(c16_enox)

def epa_19(xi):

    xh2 = xi[0]
    yco = xi[1]
    ynox = xi[2]
    yo2 = xi[3]
    ##
    avg_mw_f = xi[4]
    mass_f_ng_i = xi[5:10] # (5) 0:c 1:c2 2:c3 3:n2 4:co2 5:h2
    mass_f_h2 = xi[10]
    ##
    m_c = xi[11]
    m_h = xi[12]
    m_o = xi[13]
    m_n = xi[14]
    ##
    mf_c = xi[15]
    mf_h = xi[16]
    mf_o = xi[17]
    mf_n = xi[18]
    ##
    gcv = xi[19]
    fd = xi[20]
    ##
    cco = xi[21]
    cnox = xi[22]
    ##
    eco = xi[23]
    enox = xi[24]

    r0 = c0_avg_mw_in(jnp.array([xh2, avg_mw_f]))

    r1 = c1_mass_f_ng(
        jnp.concatenate([jnp.array([xh2]),
                         jnp.array([avg_mw_f]), mass_f_ng_i])
    )

    #print(mass_f_ng_i)
    #print(mass_f_h2)

    r2 = c2_mass_f_h2(
        jnp.concatenate([mass_f_ng_i,
                         jnp.array([mass_f_h2])])
    )
    #print("r2:")
    #print(r2)
    # 76 #######################################################################
    r3 = c3_mass_c(jnp.array([mass_f_ng_i[0],
                              mass_f_ng_i[1],
                              mass_f_ng_i[2],
                              mass_f_ng_i[4],
                              m_c]))

    r4 = c4_mass_h(jnp.array([mass_f_ng_i[0],
                              mass_f_ng_i[1],
                              mass_f_ng_i[2],
                              mass_f_h2,
                              m_h]))

    r5 = c5_mass_o(jnp.array([
        mass_f_ng_i[4],
        m_o]))

    r6 = c6_mass_n(jnp.array([
        mass_f_ng_i[3],
        m_n]))
    # 76 #######################################################################
    r7 = c7_mass_frac_c(jnp.array([m_c, m_h, m_o, m_n, mf_c]))
    r8 = c8_mass_frac_h(jnp.array([m_c, m_h, m_o, m_n, mf_h]))
    r9 = c9_mass_frac_o(jnp.array([m_c, m_h, m_o, m_n, mf_o]))
    r10 = c10_mass_frac_n(jnp.array([m_c, m_h, m_o, m_n, mf_n]))
    # 76 #######################################################################
    r11 = c11_gcv(jnp.array([mass_f_h2,
                             gcv]))

    r12 = c12_fd(jnp.array([mf_c,
                            mf_h,
                            mf_o,
                            mf_n,
                            gcv,
                            fd]))

    r13 = c13_cco(jnp.array([yco, cco]))

    r14 = c14_cnox(jnp.array([ynox, cnox]))

    r15 = c15_eco(jnp.array([yo2, fd, cco, eco]))

    r16 = c16_enox(jnp.array([yo2, fd, cnox, enox]))
    # 76 #######################################################################

    # 0
    #xh2 = x[0]
    #avg_mw_f = x[1]

    # 1
    #avg_mw_f = x[0]
    #mass_f_ng_i = x[1:]

    # 2
    #mass_f_ng_i = x[0:4]
    #mass_f_h2 = x[5]

    # 3
    #avg_mw_f = x[0]
    #mass_f_ng_ch4 = x[1]
    #mass_f_ng_c2h6 = x[2]
    #mass_f_ng_c3h8 = x[3]
    #mass_f_ng_co2 = x[4]
    #m_c = x[5]

    # 4
    #avg_mw_f = x[0]
    #mass_f_ng_ch4 = x[1]
    #mass_f_ng_c2h6 = x[2]
    #mass_f_ng_c3h8 = x[3]
    #mass_f_h2 = x[4]
    #m_h = x[5]

    # 5
    #avg_mw_f = x[0]
    #mass_f_ng_co2 = x[1]
    #m_o = x[2]

    # 6
    #avg_mw_f = x[0]
    #mass_f_ng_n2 = x[1]
    #m_n = x[3]

    # 7
    #mass_f_h2 = x[0]
    #gcv = x[1]

    # 8
    #Xc = x[0]
    #Xh = x[1]
    #Xo = x[2]
    #Xn = x[3]
    #gcv = x[4]
    #fd = x[5]

    # 9
    #ynox = x[0] * 1e-06
    #cnox = x[1]

    # 10
    #yco = x[0] * 1e-06
    #cco = x[1]

    # 11
    #yo2 = x[0] * 1e-06 * 1e+03
    #fd = x[1]
    #cnox = x[2]
    #enox = x[3]

    # 12
    #yo2 = x[0] * 1e-06 * 1e+03
    #fd = x[1]
    #cco = x[2]
    #eco = x[3]a
    return jnp.concatenate(
        [jnp.array([r0]),
            r1,
         jnp.array([r2,
                    r3, r4, r5, r6,
                    r7, r8, r9, r10,
                    r11, r12,
                    r13, r14, r15, r16])])

def jac_epa_19(xi):
    xh2 = xi[0]
    yco = xi[1]
    ynox = xi[2]
    yo2 = xi[3]

    avg_mw_f = xi[4]
    mass_f_ng_i = xi[5:10] # (5) 0:c 1:c2 2:c3 3:n2 4:co2 5:h2
    mass_f_h2 = xi[10]

    m_c = xi[11]
    m_h = xi[12]
    m_o = xi[13]
    m_n = xi[14]
    ##
    mf_c = xi[15]
    mf_h = xi[16]
    mf_o = xi[17]
    mf_n = xi[18]
    ##
    gcv = xi[19]
    fd = xi[20]
    ##
    cco = xi[21]
    cnox = xi[22]
    ##
    eco = xi[23]
    enox = xi[24]

    j0 = j0_avg_mw_in(jnp.array([xh2, avg_mw_f]))

    j1 = j1_mass_f_ng(
        jnp.concatenate([jnp.array([xh2]),
                         jnp.array([avg_mw_f]), mass_f_ng_i])
    )



    j2 = j2_mass_f_h2(
        jnp.concatenate([mass_f_ng_i,
                         jnp.array([mass_f_h2])])
    )

    # 76 #######################################################################
    j3 = j3_mass_c(jnp.array([mass_f_ng_i[0],
                              mass_f_ng_i[1],
                              mass_f_ng_i[2],
                              mass_f_ng_i[4],
                              m_c]))

    j4 = j4_mass_h(jnp.array([mass_f_ng_i[0],
                              mass_f_ng_i[1],
                              mass_f_ng_i[2],
                              mass_f_h2,
                              m_h]))

    j5 = j5_mass_o(jnp.array([mass_f_ng_i[4],
                              m_o]))

    j6 = j6_mass_n(jnp.array([mass_f_ng_i[3],
                              m_n]))
    # 76 #######################################################################
    j7 = j7_mass_frac_c(jnp.array([m_c, m_h, m_o, m_n, mf_c]))
    j8 = j8_mass_frac_h(jnp.array([m_c, m_h, m_o, m_n, mf_h]))
    j9 = j9_mass_frac_o(jnp.array([m_c, m_h, m_o, m_n, mf_o]))
    j10 = j10_mass_frac_n(jnp.array([m_c, m_h, m_o, m_n, mf_n]))
    # 76 #######################################################################
    j11 = j11_gcv(jnp.array([mass_f_h2,
                             gcv]))

    j12 = j12_fd(jnp.array([mf_c,
                            mf_h,
                            mf_o,
                            mf_n,
                            gcv,
                            fd]))

    j13 = j13_cco(jnp.array([yco, cco]))

    j14 = j14_cnox(jnp.array([ynox, cnox]))

    j15 = j15_eco(jnp.array([yo2, fd, cco, eco]))

    j16 = j16_enox(jnp.array([yo2, fd, cnox, enox]))
    # 76 #######################################################################

    j1_ = []
    for row in j1:
        nrow = jnp.array([row[0], row[1], -1.0])
        j1_.append(nrow)

    return [j0]+j1_+[j2,
                     j3, j4, j5, j6,
                     j7, j8, j9, j10,
                     j11, j12, j13, j14, j15, j16]

