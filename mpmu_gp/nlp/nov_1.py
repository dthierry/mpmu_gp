# -*- coding: utf-8 -*-

import numpy as np
import cyipopt
# required for the gradients
import jax.numpy as jnp
from chp_ed.ipopt_nlp.noxlb.epa_19_nov_1 import epa_19, jac_epa_19
from chp_ed.ipopt_nlp.noxlb.hv_fu import mass_conversion, heat_conversion
from chp_ed.ipopt_nlp.noxlb.hv_fu import jac_mass_conv, jac_heat_conv

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

import subprocess
import os
import json
import sys

__author__ = "David Thierry"

"""
notes: I am trying to implement a multi-unit scheme, e.g. 10 units that combine
their power output. However, this will only consider identical units.
This is based on the multi-period version 4.

v1: considers the aggregated production, etc. as opposed of n times the original
system
082523: this has the nox block (long method)
"""


class gauss_reg_nlp(cyipopt.Problem):
    def __init__(self, surrogate, thorz: int, nunits: int,
                 load_array: np.ndarray,
                 lmp_array: np.ndarray,
                 co2_array: np.ndarray,
                 fregfile,
                 cNg=1, cH2=1,
                 const_lmp=False, lmp_val=1.,
                 const_load=False, load_val=65.,
                 const_co2f=False, grid_co2ef_val=1.,
                 load_fact=1.,
                 co2Tax=1., ckCuFtNg=1., cMmbtuNg=1., ckgH2=1.,
                 **kwargs):
        """
        make sure the arrays are consistent with thorz
        load_array [=] kWh
        lmp_array [=] USD/kWh

        """

        frgdf = pd.read_csv(fregfile, index_col=0)  # regressions
        #
        self.mffi = frgdf.loc["m_i", :].to_numpy() # molar flow f i
        #self.mfai = frgdf.loc["mA_i", :].to_numpy()
        #
        self.mffc = frgdf.loc["m_c", :].to_numpy()
        #self.mfac = frgdf.loc["mA_c", :].to_numpy()

        date = datetime.now()  # debugging stuff
        self.datestr = date.strftime("%b_%d-%H-%M-%S")
        os.mkdir(self.datestr)
        self.debug_filename = self.datestr + "/debug-"
        self.generate_debug_file = False

        f_path = os.path.realpath(__file__)
        with open(self.debug_filename + "script.txt", "w") as t_file:
            subprocess.call(["cat", f_path], stdout=t_file)

        self.lmp = lmp_array  # grid import elec. price

        self.load = load_array  # load (demand)

        self.grid_co2ef = co2_array  # emission factor for the grid

        self.lmp *= 1

        # artificial lmp
        if const_lmp:
            print("Constant LMP")
            lmp = np.ones(self.lmp.size)
            lmp[:] = lmp_val
            self.lmp = lmp

        # artificial load
        if const_load:
            print("Constant Load")
            self.load = np.ones(self.load.size)
            self.load[:] = load_val

        # artificial co2 factor
        if const_co2f:
            print("Const CO2")
            self.grid_co2ef = np.ones(self.grid_co2ef.size)
            self.grid_co2ef[:] = grid_co2ef_val
        print(f"Load factor = {load_fact}")
        self.load *= load_fact

        self.s_0 = surrogate.s_0
        self.s_1 = surrogate.s_1

        #self.xp = np.array([])
        self.thorz = thorz
        self.nunits = nunits

        # 72 ###################################################################
        # var info
        # we divide variables per unit block first and then in time.
        self.s_input_v = 2  # surrogate input shared by all

        self.s_output_v_0 = 4  # surrogate output 0
        self.s_output_v_1 = 5  # surrogate output 1

        self.nox_lb_n_block = 21  # epa 19

        self.nox_lb_pow_block = 2  # correction with efficiency

        self.mass_conv_block = 2  # 2 eqn and con
        self.heat_conv_block = 2

        self.co2_w_block = 4

        self.s_output = [self.s_output_v_0, self.s_output_v_1]
        self.s_output_v_o = sum(self.s_output)

        self.add_v_u_block = 0  # additional variables in the unit? block

        self.add_v_u_block += 1  # var o&m
        self.add_v_u_block += 1  # var total SLP
        self.add_v_u_block += 1  # var h2 SLP
        self.add_v_u_block += 1  # var total mass flow
        #: 220823
        #: noxlb block
        self.add_v_u_block += self.nox_lb_n_block # additional variables
        #: 101023
        self.add_v_u_block += self.nox_lb_pow_block
        #: 041023
        self.add_v_u_block += self.mass_conv_block
        self.add_v_u_block += self.heat_conv_block
        self.add_v_u_block += self.co2_w_block

        # variables in unit-block
        self.v_u_block = self.s_input_v \
                         + self.s_output_v_o \
                         + self.add_v_u_block
        print(f"v_u_block={self.v_u_block}")

        self.add_t_vars = 0  # additional variables within the context of the...
        # ...time block
        self.add_t_vars += 1  # total generation
        self.add_t_vars += 1  # import (positive slack)
        self.add_t_vars += 1  # neg slack
        self.add_t_vars += 1  # import cost
        self.add_t_vars += 1  # import co2
        self.add_t_vars += 1  # total vonm
        self.add_t_vars += 1  # total co
        self.add_t_vars += 1  # total nox
        # total unit-block variables (single time-block)

        self.v_tu_block = self.v_u_block * self.nunits + self.add_t_vars
        print(f"v_tu_block={self.v_tu_block}")

        # additional out-of-time-block variables
        self.add_vars = 0
        self.v_block = self.v_tu_block * self.thorz + self.add_vars

        # 72 ###################################################################
        # con info
        self.surr_block_0 = 4  # surrogate block 0
        self.surr_block_1 = 5  # surrogate block 1

        self.surr_block = [self.surr_block_0,
                           self.surr_block_1]

        self.surr_block_o = sum(self.surr_block)

        self.add_c_u_block = 0  # add constraints in the unit? block

        self.add_c_u_block += 1  # var o&m   #0
        self.add_c_u_block += 1  # total SLP #1
        self.add_c_u_block += 1  # h2 SLP    #2
        self.add_c_u_block += 1  # total mass fr    #3
        #: 220823
        #: noxlb block
        self.add_c_u_block += self.nox_lb_n_block  # new equations
        self.add_c_u_block += self.nox_lb_pow_block  # new equations
        self.add_c_u_block += self.mass_conv_block  # new equations
        self.add_c_u_block += self.heat_conv_block  # new equations
        self.add_c_u_block += self.co2_w_block # new equations

        self.add_t_cons = 0  # additional constraints in the time block
        self.add_t_cons += 1  # total gen
        self.add_t_cons += 1  # demand
        self.add_t_cons += 1  # imp cost
        self.add_t_cons += 1  # imp co2
        self.add_t_cons += 1  # total vonm
        self.add_t_cons += 1  # total co
        self.add_t_cons += 1  # total nox


        # constraints in unit-block
        self.c_u_block = self.surr_block_o + self.add_c_u_block

        # total unit-block constraints (single time-block)
        self.c_tu_block = self.c_u_block * self.nunits + self.add_t_cons


        # out-of-time-block constraints
        self.add_cons = 0

        self.c_block = self.c_tu_block * self.thorz + self.add_cons

        #54429091390059111319955091733536609023614493455356060337724161134189978
        # nonzeros from the surrogate (dense)
        # surrogate block
        nz_u = self.s_input_v * (self.surr_block_0+self.surr_block_1)
        nz_u += (self.surr_block_0 + self.surr_block_1)
        # add u block
        nz_u += 3 # 0th
        nz_u += 3 # 1st
        nz_u += 3 # 2nd
        nz_u += 3 # 3rd
        njn0 = nz_u

        nz_u += 2  # 0
        nz_u += 3  # 1
        nz_u += 3  # 2
        nz_u += 3  # 3
        nz_u += 3  # 4
        nz_u += 3  # 5
        nz_u += 6  # 6

        nz_u += 5  # 7
        nz_u += 5  # 8
        nz_u += 2  # 9
        nz_u += 2  # 10

        nz_u += 5  # 11
        nz_u += 5  # 12
        nz_u += 5  # 13
        nz_u += 5  # 14

        nz_u += 2  # 15
        nz_u += 6  # 16
        nz_u += 2  # 17
        nz_u += 2  # 18

        nz_u += 4  # 19
        nz_u += 4  # 20

        njn1 = nz_u
        #
        nz_u += 3
        nz_u += 3
        #
        njn0 = nz_u
        nz_u += 2
        nz_u += 2
        njn1 = nz_u
        njn0 = nz_u
        nz_u += 2
        nz_u += 2
        njn1 = nz_u
        # co2w
        nz_u += 2
        nz_u += 4
        nz_u += 3
        nz_u += 3

        #

        # add t block
        nz_add_t = 1 + self.nunits # 0th tgen
        nz_add_t += 3  # 1st demand
        nz_add_t += 2  # 2nd imp cost
        nz_add_t += 2  # 3nd imp co2
        nz_add_t += 1 + self.nunits  # 4rd
        nz_add_t += 1 + self.nunits  # 5rd
        nz_add_t += 1 + self.nunits  # 6th

        self.nnzJ = (nz_u * self.nunits + nz_add_t) * self.thorz
        print(f"nnzJ={self.nnzJ}")

        #62693569614482526429594921815729462651307437020096094256770739590555097
        # 72 ###################################################################
        # ID all the variables
        # input to surrogate
        self.id_in_iload = 0
        self.id_in_xh2 = 1

        # 0 ngslpm
        # 1 opow
        # 2 itt
        # 3 eff ##
        # 4 co2
        # 5 co
        # 6 nox
        # 7 thc
        self.names_0 = [
            "DEMPOWKW",  # 0
            "H2%",  # 1
            #
            "NGSLP",  # 2
            "OBSPOWKW",  # 3
            "EFF%",  # 4
            "ITT",  # 5
            #
            "CO2",  # 6
            "CO",  # 7
            "NOx",  # 8
            "THC",  # 9
            "O2",  # 9
            #
            "VO&M",  # 10
            "TOTSLP",  # 14
            "H2SLP",  # 15
            "TMF"
        ]
        self.names_a0 = ["vonm", "tslpm", "h2_slpm", "tmflow"]
        self.names_nox = [
            "avg_mw_in",  # 0
            "mass_f_ng_0",  # 1
            "mass_f_ng_1",  # 2
            "mass_f_ng_2",  # 3
            "mass_f_ng_3",  # 4
            "mass_f_ng_4",  # 5
            "mass_f_h2",  # 6
            #
            "mass_c",  # 7
            "mass_h",  # 8
            "mass_o",  # 9
            "mass_n",  # 10
            #
            "mass_perc_c",  # 11
            "mass_perc_h",  # 12
            "mass_perc_o",  # 13
            "mass_perc_n",  # 14
            #
            "gcv",  # 15
            "fd",  # 16
            #
            "cco",  # 17
            "cnox",  # 18
            "eco",  # 19
            "enox"  # 20
        ]

        self.names_nox_pow = ["eco_pow", "enox_pow"]

        self.names_mc = [
            "ng_kg",  # 0
            "h2_kg",  # 1
        ]
        self.names_hc = [
            "ng_h",  # 0
            "h2_h",  # 1
        ]
        self.names_co2w = [
            "cco2",  # 0
            "eco2",  # 1
            "eco2_pow",  # 1
            "co2w",  # 3
        ]

        self.names_t = [
            "TGEN kWh", # 0
            "E-IMP kWh", # 1
            "E-N-SlLACK kWh", # 2
            "E-IMP_COST", # 3
            "E-IMP_CO2", # 4
            "TVONM", # 5
            "TCO", # 0
            "TNOX"]

        l = []
        for t in range(self.thorz):
            for u in range(self.nunits):
                l += [n + f"_{u}" for n in self.names_0]  \
                    + [n + f"_{u}" for n in self.names_nox] \
                    + [n + f"_{u}" for n in self.names_nox_pow] \
                    + [n + f"_{u}" for n in self.names_mc] \
                    + [n + f"_{u}" for n in self.names_hc] \
                    + [n + f"_{u}" for n in self.names_co2w]
            l += [n + f"_{t}" for n in self.names_t]
            if t == 0:
                dl = pd.DataFrame({k: [0] for k in l})
                dl.to_csv(self.debug_filename + "names_1transpose.csv")


        dl = pd.DataFrame(l)
        dl.to_csv(self.debug_filename + "names.csv")

        con_names_u = ["s0_0", "s0_1", "s0_2", "s0_3", "s1_0", "s1_1",
                       "s1_2", "s1_3", "s1_4", "a0", "a1", "a2", "a3",
                       "an0", "an1", "an2", "an3", "an4", "an5", "an6",
                       "an7", "an8", "an9", "an10",
                       "an11", "an12", "an13", "an14",
                       "an15", "an16", "an17", "an18",
                       "an19", "an20"]
        con_names_nox_pow = ["nlbpow0", "nlbpow1"]
        con_names_mc = ["nkg", "hkg"]
        con_names_hc = ["nh", "hh"]
        con_names_co2w = ["co2w0", "co2w1", "co2w2", "co2w3"]

        con_names_t = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"]

        l = []
        for t in range(self.thorz):
            for u in range(self.nunits):
                l += con_names_u + con_names_nox_pow \
                    + con_names_mc \
                    + con_names_hc \
                    + con_names_co2w
            l += con_names_t

        dl = pd.DataFrame(l)
        dl.to_csv(self.debug_filename + "con_names.csv")

        self.id_surr_0_ng_slpm = 0  # Natural Gas (slpm)
        self.id_surr_0_load = 1  # Load (kW)
        self.id_surr_0_eff = 2  # Eff (%)
        self.id_surr_0_itt = 3  # ITT

        self.id_surr_1_co2 = 0  # CO2 ppm
        self.id_surr_1_co = 1  # CO ppm
        self.id_surr_1_nox = 2  # NOx ppm
        self.id_surr_1_thc = 3  # THC ppm
        self.id_surr_1_o2 = 4  # O2 ppm

        self.id_a_vonm = 0  # var o and m
        self.id_a_tslp = 1  # total flow slp
        self.id_a_h2_slpm = 2  # total flow h2
        self.id_a_tmflow = 3  # total MASS flow
        #: the start of the noxblock
        self.id_a_noxlbptr_ = 4
        #: noxlb block
        self.id_a_avg_mw_in = self.id_a_noxlbptr_ # 4
        self.id_a_mass_f_ng_0 = self.id_a_noxlbptr_+1  # 5 ch4 5
        self.id_a_mass_f_ng_1 = self.id_a_noxlbptr_+2  # 6 c2h6 6
        self.id_a_mass_f_ng_2 = self.id_a_noxlbptr_+3  # 7 c3h8 7
        self.id_a_mass_f_ng_3 = self.id_a_noxlbptr_+4  # 8 n2 8
        self.id_a_mass_f_ng_4 = self.id_a_noxlbptr_+5  # 9 co2 9
        self.id_a_mass_f_h2 = self.id_a_noxlbptr_+6  # 10 h2 10

        self.id_a_mass_c = self.id_a_noxlbptr_+7  # 11
        self.id_a_mass_h = self.id_a_noxlbptr_+8  # 12
        self.id_a_mass_o = self.id_a_noxlbptr_+9  # 13
        self.id_a_mass_n = self.id_a_noxlbptr_+10  # 14

        self.id_a_mass_perc_c = self.id_a_noxlbptr_+11  # 15
        self.id_a_mass_perc_h = self.id_a_noxlbptr_+12  # 16
        self.id_a_mass_perc_o = self.id_a_noxlbptr_+13  # 17
        self.id_a_mass_perc_n = self.id_a_noxlbptr_+14  # 18

        self.id_a_gcv = self.id_a_noxlbptr_+15  # 19
        self.id_a_fd = self.id_a_noxlbptr_+16  # 20

        self.id_a_cco = self.id_a_noxlbptr_+17  # 21
        self.id_a_cnox = self.id_a_noxlbptr_+18  # 22
        self.id_a_eco = self.id_a_noxlbptr_+19  # 23
        self.id_a_enox = self.id_a_noxlbptr_+20  # 24
        #: correction by efficiency
        self.id_a_eco_pow = self.id_a_noxlbptr_ + 21  # 25
        self.id_a_enox_pow = self.id_a_noxlbptr_ + 22  # 26
        #:
        self.id_a_mc_ptr_ = 27
        self.id_a_ng_kg = self.id_a_mc_ptr_  # 27
        self.id_a_h2_kg = self.id_a_mc_ptr_+1  # 28
        #:
        self.id_a_hc_ptr_ = 29
        self.id_a_ng_h = self.id_a_hc_ptr_  # 29
        self.id_a_h2_h = self.id_a_hc_ptr_+1  # 30
        #:
        self.id_a_cco2 = self.id_a_hc_ptr_+2  # 31
        self.id_a_eco2 = self.id_a_hc_ptr_+3  # 32
        self.id_a_eco2_pow = self.id_a_hc_ptr_+4 # 33
        self.id_a_wco2 = self.id_a_hc_ptr_+5  # 34

        self.id_t_tgen = 0
        self.id_t_pimp = 1
        self.id_t_nimp = 2
        self.id_t_imp_c = 3
        self.id_t_imp_co2 = 4
        self.id_t_tvonm = 5
        self.id_t_tco = 6
        self.id_t_tnox = 7

        self.costH2 = cH2
        self.costNg = cNg

        self.costNg = np.linspace(self.costNg, self.costNg * 1.1,
                                  num=self.nunits)
        self.costH2 = np.linspace(self.costH2, self.costH2 * 1.1,
                                  num=self.nunits)

        self.demand = self.load
        dm = pd.Series([], dtype=np.float64)
        for time in range(self.thorz):
            dm = pd.concat([dm, pd.Series(self.demand[time], dtype=np.float64)],
                           ignore_index=True)
        dm = pd.DataFrame(self.demand)
        dm.to_csv(self.debug_filename + "demand.csv")
        dc = pd.DataFrame(self.grid_co2ef)
        dc.to_csv(self.debug_filename + "grid_co2ef.csv")
        dlmp = pd.DataFrame(self.lmp)
        dlmp.to_csv(self.debug_filename + "lmp.csv")
        names = ["cng", "ch2", "ctax", "ckcuftng",  "cmmbtung", "ckgh2"]
        values = [cNg, cH2, co2Tax, ckCuFtNg, cMmbtuNg, ckgH2]
        dparams = pd.DataFrame({"name": names, "values" : values})
        dparams.to_csv(self.debug_filename + "params.csv")

        self._ic = 99
        self._inf_du = 0

        self.print_low_inf_du_single_time = True
        self.print_low_inf_pr_single_time = True

        self.co2Tax = co2Tax  # this should be in USD/kgCo2
        self.ckCuFtNg = ckCuFtNg  # usd cost per 1000 cuftNg
        self.cMmbtuNg = cMmbtuNg  # usd cost per Mmbtu of Ng
        self.ckgH2 = ckgH2  # usd cost per kgH2

        print(f"co2Tax={self.co2Tax}")
        print(f"cKCuFtNg={self.ckCuFtNg}\tcMmbtuNg={self.cMmbtuNg}")
        print(f"ckgH2={self.ckgH2}")

        cyipopt.Problem.__init__(self, **kwargs)


    # 76 #######################################################################
    def objective(self, x):
        retval = 0.0
        # 7.74 USD/kCuFT
        #ckCuFtNg = 7.74  # feb 2018
        # 8.84 USD/kCuFt in july
        # 28.316846592 liter/cuft
        lTokCuFt = 28.316846592 * 1e3  # liter/cuft * cuft/kCuFt = l/kCuFt
        # 100 USD/tCO2
        #cCO2 = 1e2 * (1/1000)

        #cMmbtuNg = 2.67  # usd/mmbtuNg
        #ckgH2 = 4.0  # usd/kgH2

        for time in range(self.thorz):
        #    # 67 ##############################################################
            col0_0 = self.v_tu_block * time
            for unit in range(self.nunits):
                col_id0 = col0_0 + self.v_u_block*unit  # input part
                #retval -= x[col_id0+1]  # max h2
                col_id1_0 = col_id0 + self.s_input_v  # surr output 0
                col_id1_1 = col_id1_0 + self.s_output_v_0  # surr output 1
                #retval += x[col_id1_1 + self.id_surr_1_co]
                #
                col_id2 = col_id1_1 + self.s_output_v_1
                #
                #retval += x[col_id2 + self.id_a_enox] ** 2
                #retval += (x[col_id2 + self.id_a_enox_pow]-0.07)**2

                # hv cost minimization
                #retval += x[col_id2 + self.id_a_ng_h] \
                #    + x[col_id2 + self.id_a_h2_h] * 1e1
                # sl/min * (60min/hr) = sl/hr
                #retval += x[col_id1_0 + self.id_surr_0_ng_slpm] \
                #    *60.*(1./lTokCuFt)*ckCuFtNg
                retval += x[col_id2 + self.id_a_h2_kg] * self.ckgH2 # 4 USD/kgH2
                retval += x[col_id2 + self.id_a_ng_h] * self.cMmbtuNg
                # co2 from the turbine
                retval += x[col_id2 + self.id_a_wco2] * self.co2Tax

            col0 = col0_0 + self.v_u_block * self.nunits
            retval += x[col0 + self.id_t_imp_c]  # imports

            # co2 from the grid
            retval += x[col0 + self.id_t_imp_co2] * self.co2Tax
        return retval

    # 76 #######################################################################
    def gradient(self, x):
        # 7.74 USD/kCuFT
        #ckCuFtNg = 7.74  # feb 2018
        # 28.316846592 liter/cuft
        lTokCuFt = 28.316846592 * 1e3  # liter/cuft * cuft/kCuFt = l/kCuFt

        #cMmbtuNg = 2.67  # usd/mmbtu
        #cKgH2 = 4.0  # usd/kgH2

        retval = np.zeros(self.v_block)
        for time in range(self.thorz):
            col0_0 = self.v_tu_block * time
            for unit in range(self.nunits):
                col_id0 = col0_0 + self.v_u_block*unit  # input part
                #retval[col_id0 + 1] = -1.0  # max h2
                col_id1_0 = col_id0 + self.s_input_v  # surr output 0
                col_id1_1 = col_id1_0 + self.s_output_v_0  # surr output 1
                #retval[col_id1_1 + self.id_surr_1_co] = 1.0

                col_id2 = col_id1_1 + self.s_output_v_1

                #retval[col_id2 + self.id_a_enox] = \
                #    2. * x[col_id2 + self.id_a_enox]

                #retval[col_id2 + self.id_a_enox_pow] = 2.0 \
                #    *(x[col_id2 + self.id_a_enox_pow]-0.07)

                # hv cost minimization
                #retval[col_id2 + self.id_a_ng_h] = 1e0
                #retval[col_id2 + self.id_a_h2_h] = 1e1

                #retval[col_id1_0 + self.id_surr_0_ng_slpm] = \
                #    60.*(1./lTokCuFt)*ckCuFtNg

                retval[col_id2 + self.id_a_h2_kg] = self.ckgH2
                retval[col_id2 + self.id_a_ng_h] = self.cMmbtuNg

                # co2 from the turbine
                retval[col_id2 + self.id_a_wco2] = self.co2Tax

            col0 = col0_0 + self.v_u_block * self.nunits
            retval[col0 + self.id_t_imp_c] = 1.0 # imports

            # co2 from the grid
            retval[col0 + self.id_t_imp_co2] = self.co2Tax
        # ##
        if self.generate_debug_file:
            d = pd.DataFrame(retval)
            d.to_csv(self.debug_filename + "gradient.csv")
        return retval

    # 76 #######################################################################
    def constraints(self, x):
        # 72####################################################################
        lmp = self.lmp
        costH2 = self.costH2
        costNg = self.costNg
        demand = self.demand

        mffi = self.mffi
        #mfai = self.mfai
        #
        mffc = self.mffc
        #mfac = self.mfac

        retval = np.zeros(self.c_block)
        x_i = np.zeros((self.nunits * self.thorz, self.s_input_v))
        # 72 ###################################################################
        for time in range(self.thorz):
            col0_0 = self.v_tu_block * time
            # 68 ###############################################################
            # 68 ###############################################################
            for unit in range(self.nunits):
                # surrogate output block indices (variables)
                col_id0 = col0_0 + self.v_u_block*unit  # input part
                col_id1_0 = col_id0 + self.s_input_v  # surr output 0
                # extract the input values from x
                x_i[unit + time * self.nunits, :] = x[col_id0:col_id1_0]
                # 67 ##########################################################

        # evaluate the surrogate
        fstar_0, _, _, _ = self.s_0.test(x_i, fidelity=1)
        # second surrogate has fidelity of 2
        fstar_1, _, _, _ = self.s_1.test(x_i, fidelity=2)
        for time in range(self.thorz):
            # trying to point out to the first row of the time block
            row0_0 = self.c_tu_block * time
            col0_0 = self.v_tu_block * time
            # 67 ###############################################################
            t_gen = 0e0
            t_vonm = 0e0
            t_coppm = 0e0
            t_noxppm = 0e0
            # 67 ###############################################################
            for unit in range(self.nunits):
                # surrogate block
                # first, the surr_block constraints residuals
                # surrogate output block indices (constraints)
                row0 = row0_0 + self.c_u_block * unit
                row1_0 = row0 + self.surr_block_0 # surr 0
                row1_1 = row1_0 + self.surr_block_1  # surr 1
                row1 = row1_1
                #f"row1: {row1}")
                # surrogate output block indices (variables)
                col_id0 = col0_0 + self.v_u_block*unit  # input part
                col_id1_0 = col_id0 + self.s_input_v  # surr output 0
                col_id1_1 = col_id1_0 + self.s_output_v_0  # surr output 1
                col_id2 = col_id1_1 + self.s_output_v_1
                col_id3 = col_id2 + self.add_v_u_block
                # extract the input values from x
                x_i_surr = x[col_id0:col_id1_0]
                # fetch the surr output variables
                y_0 = x[col_id1_0:col_id1_1]
                y_1 = x[col_id1_1:col_id2]

                # evaluate the surrogate(s)
                # first surrrogate has fidelity of 1

                # residual 0 and 1

                retval[row0:row1_0] = fstar_0[unit + time * self.nunits] \
                    - y_0
                retval[row1_0:row1_1] = fstar_1[unit + time * self.nunits] \
                    - y_1

                #d0 = pd.DataFrame(fstar_0[unit + time * self.nunits])
                #d1 = pd.DataFrame(fstar_1[unit + time * self.nunits])
                #print(pd.concat([d0, d1], axis=1))

                # fetch the remaining variables
                x_add_v_u = x[col_id2:col_id3]


                ## additional u constraints

                # 0 vonm = flowNg*costNg + flowH2*costH2
                retval[row1] = costH2[unit] * x_add_v_u[self.id_a_h2_slpm] \
                               + costNg[unit] * y_0[self.id_surr_0_ng_slpm] \
                               - x_add_v_u[self.id_a_vonm]


                # 1 totalSLP = tSLP = ngSLP / (1 - %h2)
                #retval[row1 + 1] = \
                #    y_0[self.id_surr_0_ng_slpm]/
                # (1-x_i_surr[self.id_in_xh2]/100) \
                #    -x_add_v_u[self.id_a_tslp]
                retval[row1 + 1] = \
                    y_0[self.id_surr_0_ng_slpm] \
                    -x_add_v_u[self.id_a_tslp]*(1-x_i_surr[self.id_in_xh2]/100)

                # 2 h2SLP = tSLP - ngSLP
                retval[row1 + 2] = x_add_v_u[self.id_a_tslp] \
                                   - y_0[self.id_surr_0_ng_slpm] \
                                   - x_add_v_u[self.id_a_h2_slpm]
                # mass flow fuel coeff and mass flow fuel int
                # 3rd linreg(pow, %h2) = tmflow
                retval[row1 + 3] = \
                    mffc[0] * x_i_surr[self.id_in_iload] \
                    + mffc[1] * x_i_surr[self.id_in_xh2]/100 \
                    + mffi[0] \
                    - x_add_v_u[self.id_a_tmflow]

                # lb of nox block ##############################################
                rownox = row1 + 4
                xnox_i = jnp.array([x_i_surr[1],
                           y_1[1],
                           y_1[2],
                           y_1[4],
                           ])
                xnox_i = jnp.concatenate(
                    [xnox_i, x_add_v_u[self.id_a_noxlbptr_:]])

                xnox_o = epa_19(xnox_i)

                retval[rownox:(rownox+xnox_o.size)] = xnox_o

                # lb of nox correction #########################################
                row_nox_pow = rownox + xnox_o.size
                retval[row_nox_pow+0] = \
                    (x_add_v_u[self.id_a_eco] / y_0[self.id_surr_0_eff]) \
                    - x_add_v_u[self.id_a_eco_pow]
                retval[row_nox_pow+1] = \
                    (x_add_v_u[self.id_a_enox] / y_0[self.id_surr_0_eff]) \
                    - x_add_v_u[self.id_a_enox_pow]


                # mass conv ####################################################
                x_mc_i = jnp.array([y_0[self.id_surr_0_ng_slpm],
                                    x_add_v_u[self.id_a_h2_slpm],
                                    x_add_v_u[self.id_a_ng_kg],
                                    x_add_v_u[self.id_a_h2_kg]
                                    ])

                row_mc = row_nox_pow + 2

                x_mc_o = mass_conversion(x_mc_i)

                retval[row_mc:(row_mc+x_mc_o.size)] = x_mc_o

                # heat conv ####################################################
                x_hc_i = jnp.array([x_add_v_u[self.id_a_ng_kg],
                                    x_add_v_u[self.id_a_h2_kg],
                                    x_add_v_u[self.id_a_ng_h],
                                    x_add_v_u[self.id_a_h2_h]])
                row_hc = row_mc + x_mc_o.size

                x_hc_o = heat_conversion(x_hc_i)

                retval[row_hc:(row_hc + x_hc_o.size)] = x_hc_o
                # co2 w ### ####################################################
                row_co2w = row_hc + x_hc_o.size
                # conversion factor 1 MWh = 3.4095106405145 MMBTU
                mmbtuToMwh = 3.4095106405145
                # co2w 0: cco2
                retval[row_co2w+0] = \
                    y_1[self.id_surr_1_co2]*1e-06*44.009/(0.730240*528.0) \
                    - x_add_v_u[self.id_a_cco2]
                #   1e-06*44.009/(0.730240*528.0) # id_surr_1_co2

                # co2w 1: eco2 (in lb/MWh)
                retval[row_co2w+1] = \
                    mmbtuToMwh*(x_add_v_u[self.id_a_cco2]
                                *x_add_v_u[self.id_a_fd]) \
                    * 20.9/(20.9 - y_1[self.id_surr_1_o2]*1e-06*1e+02) \
                    - x_add_v_u[self.id_a_eco2]

                # co2w 2: eco2_pow (in lb/MWhe)
                retval[row_co2w+2] = \
                    (x_add_v_u[self.id_a_eco2]/y_0[self.id_surr_0_eff]) \
                    - x_add_v_u[self.id_a_eco2_pow]
                # 1 kg = 2.204623 lbs
                kgToLb = 2.204623 # lb/kg
                mwhToKwh = 1e3  # kWh/MWh
                #

                # co2w 3: wco2 (kgCO2) a.k.a. id_a_wco2
                retval[row_co2w+3] = \
                    x_add_v_u[self.id_a_eco2_pow] \
                    *y_0[self.id_surr_0_load]/(kgToLb*mwhToKwh) \
                    - x_add_v_u[self.id_a_wco2]

            # 67 ###############################################################
                t_gen += y_0[self.id_surr_0_load]
                t_vonm += x_add_v_u[self.id_a_vonm]
                t_coppm += y_1[self.id_surr_1_co]
                t_noxppm += y_1[self.id_surr_1_nox]

            # 67 ###############################################################
            row0 = row0_0 + self.c_u_block * self.nunits
            # surrogate output block indices (variables)
            col0 = col0_0 + self.v_u_block * self.nunits

            x_t = x[col0:(col0 + self.add_t_vars)]

            # 0 t_gen - sum_u gen_u = 0
            retval[row0 + 0] = x_t[self.id_t_tgen] - t_gen

            # 1 load + x_add_v_u[imports] >= edemand
            #   edemand - load - xadd_t_v[imports] <= 0
            retval[row0 + 1] = -demand[time] \
                + x_t[self.id_t_tgen] \
                + x_t[self.id_t_pimp] \
                - x_t[self.id_t_nimp]

            # 2 eimportcost = lmp * eimport
            retval[row0 + 2] = lmp[time] * x_t[self.id_t_pimp] \
                               - x_t[self.id_t_imp_c]
            # 4 imp_co2 = co2ef * eimport a.k.a. id_t_imp_co2
            retval[row0 + 3] = self.grid_co2ef[time] * x_t[self.id_t_pimp] \
                               - x_t[self.id_t_imp_co2]
            # 5
            retval[row0 + 4] = x_t[self.id_t_tvonm] - t_vonm

            # 6 t_co - sum_u coppm_u = 0
            retval[row0 + 5] = x_t[self.id_t_tco] - t_coppm
            # 7 t_nox - sum_u noxppm_u = 0
            retval[row0 + 6] = x_t[self.id_t_tnox] - t_noxppm

        #if self._ic % 100 == 0:
        #    df.to_csv(f"{self._ic}-neg.csv")
        #    drv = pd.DataFrame(retval)
        #    drv.to_csv(f"{self._ic}-constraints.csv")
        #    drv = pd.DataFrame(x)
        #    drv.to_csv(f"{self._ic}-v.csv")

        if self.generate_debug_file:
            d = pd.DataFrame(retval)
            d.to_csv(self.debug_filename + "constraints.csv")
            d = pd.DataFrame()
            rows = {}
            cols = {names}
            # self.generate_debug_file = False
            dm = pd.Series([], dtype=np.float64)
            for time in range(self.thorz):
                dm = pd.concat([dm, pd.Series(demand[time], dtype=np.float64)],
                               ignore_index=True)
            dm.to_csv(self.debug_filename + "demand.csv")

        return retval

    def jacobianstructure(self):
        row = np.zeros(self.nnzJ)
        col = np.zeros(self.nnzJ)
        k = 0
        for time in range(self.thorz):
            row0_0 = self.c_tu_block * time
            col0_0 = self.v_tu_block * time
            for unit in range(self.nunits):
                row0 = row0_0 + self.c_u_block * unit
                row1_0 = row0 + self.surr_block_0 # surr 0
                row1_1 = row1_0 + self.surr_block_1  # surr 1
                row1 = row1_1
                #
                col_id0 = col0_0 + self.v_u_block*unit  # input part
                col_id1_0 = col_id0 + self.s_input_v  # surr output 0
                col_id1_1 = col_id1_0 + self.s_output_v_0  # surr output 1
                col_id2 = col_id1_1 + self.s_output_v_1
                # add vars t
                #
                for i in range(self.surr_block_0):
                    # every row has 3 nonzeroes
                    row[k] = row0 + i
                    row[k+1] = row0 + i
                    row[k+2] = row0 + i
                    #
                    col[k] = col_id0 # input 1
                    col[k+1] = col_id0 + 1
                    col[k+2] = col_id1_0 + i
                    #
                    k+= 3
                for i in range(self.surr_block_1):
                    row[k] = row1_0 + i
                    row[k+1] = row1_0 + i
                    row[k+2] = row1_0 + i
                    #
                    col[k] = col_id0
                    col[k+1] = col_id0 + 1
                    col[k+2] = col_id1_1 + i
                    k += 3
                    #

                # 0th
                row[k+0] = row1
                row[k+1] = row1
                row[k+2] = row1
                #
                col[k+0] = col_id1_0 + self.id_surr_0_ng_slpm
                col[k+1] = col_id2 + self.id_a_h2_slpm
                col[k+2] = col_id2 + self.id_a_vonm
                #
                k += 3
                #
                # 1st
                row[k] = row1 + 1
                row[k+1] = row1 + 1
                row[k+2] = row1 + 1
                #
                col[k] = col_id0 + self.id_in_xh2
                col[k+1] = col_id1_0 + self.id_surr_0_ng_slpm
                col[k+2] = col_id2 + self.id_a_tslp
                #
                k += 3
                # 2nd
                row[k] = row1 + 2
                row[k+1] = row1 + 2
                row[k+2] = row1 + 2
                #
                col[k] = col_id1_0 + self.id_surr_0_ng_slpm
                col[k+1] = col_id2 + self.id_a_tslp
                col[k+2] = col_id2 + self.id_a_h2_slpm
                #
                k += 3
                # 3rd
                row[k] = row1 + 3
                row[k+1] = row1 + 3
                row[k+2] = row1 + 3
                #
                col[k] = col_id0 + self.id_in_iload
                col[k+1] = col_id0 + self.id_in_xh2
                col[k+2] = col_id2 + self.id_a_tmflow
                k += 3
                # nox block
                #
                noxbc0 = 4 # 4 bc the last row was at three
                # 0
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                #
                col[k+0] = col_id0 + self.id_in_xh2
                col[k+1] = col_id2 + self.id_a_avg_mw_in
                k += 2  # 0
                noxbc0 += 1
                # 1 id_in_xh2
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                #
                col[k+0] = col_id0 + self.id_in_xh2
                col[k+1] = col_id2 + self.id_a_avg_mw_in
                col[k+2] = col_id2 + self.id_a_mass_f_ng_0
                #
                k += 3  # 1
                noxbc0 += 1
                # 2
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0

                col[k+0] = col_id0 + self.id_in_xh2
                col[k+1] = col_id2 + self.id_a_avg_mw_in
                col[k+2] = col_id2 + self.id_a_mass_f_ng_1
                k += 3  # 2
                noxbc0 += 1
                # 3
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0

                col[k+0] = col_id0 + self.id_in_xh2
                col[k+1] = col_id2 + self.id_a_avg_mw_in
                col[k+2] = col_id2 + self.id_a_mass_f_ng_2
                k += 3  # 3
                noxbc0 += 1
                # 4
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0

                col[k+0] = col_id0 + self.id_in_xh2
                col[k+1] = col_id2 + self.id_a_avg_mw_in
                col[k+2] = col_id2 + self.id_a_mass_f_ng_3
                k += 3  # 4
                noxbc0 += 1
                # 5
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0

                col[k+0] = col_id0 + self.id_in_xh2
                col[k+1] = col_id2 + self.id_a_avg_mw_in
                col[k+2] = col_id2 + self.id_a_mass_f_ng_4
                k += 3  # 5
                noxbc0 += 1
                # 6
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0
                row[k+4] = row1 + noxbc0
                row[k+5] = row1 + noxbc0

                col[k+0] = col_id2 + self.id_a_mass_f_ng_0
                col[k+1] = col_id2 + self.id_a_mass_f_ng_1
                col[k+2] = col_id2 + self.id_a_mass_f_ng_2
                col[k+3] = col_id2 + self.id_a_mass_f_ng_3
                col[k+4] = col_id2 + self.id_a_mass_f_ng_4
                col[k+5] = col_id2 + self.id_a_mass_f_h2
                k += 6  # 6
                noxbc0 += 1
                # 7 mass_c
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0
                row[k+4] = row1 + noxbc0
                #
                col[k+0] = col_id2 + self.id_a_mass_f_ng_0
                col[k+1] = col_id2 + self.id_a_mass_f_ng_1
                col[k+2] = col_id2 + self.id_a_mass_f_ng_2
                col[k+3] = col_id2 + self.id_a_mass_f_ng_4
                col[k+4] = col_id2 + self.id_a_mass_c
                k += 5
                noxbc0 += 1
                # 8 mass_h
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0
                row[k+4] = row1 + noxbc0
                #
                col[k+0] = col_id2 + self.id_a_mass_f_ng_0
                col[k+1] = col_id2 + self.id_a_mass_f_ng_1
                col[k+2] = col_id2 + self.id_a_mass_f_ng_2
                col[k+3] = col_id2 + self.id_a_mass_f_h2
                col[k+4] = col_id2 + self.id_a_mass_h
                k += 5
                noxbc0 += 1
                # 9 mass_o
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                #
                col[k+0] = col_id2 + self.id_a_mass_f_ng_4
                col[k+1] = col_id2 + self.id_a_mass_o
                k += 2
                noxbc0 += 1
                # 10 mass_n
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                #
                col[k+0] = col_id2 + self.id_a_mass_f_ng_3
                col[k+1] = col_id2 + self.id_a_mass_n
                k += 2
                noxbc0 += 1
                # 11 mass_perc_c
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0
                row[k+4] = row1 + noxbc0
                #
                col[k+0] = col_id2 + self.id_a_mass_c
                col[k+1] = col_id2 + self.id_a_mass_h
                col[k+2] = col_id2 + self.id_a_mass_o
                col[k+3] = col_id2 + self.id_a_mass_n
                col[k+4] = col_id2 + self.id_a_mass_perc_c
                k += 5
                noxbc0 += 1
                # 12 mass_perc_h
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0
                row[k+4] = row1 + noxbc0
                #
                col[k+0] = col_id2 + self.id_a_mass_c
                col[k+1] = col_id2 + self.id_a_mass_h
                col[k+2] = col_id2 + self.id_a_mass_o
                col[k+3] = col_id2 + self.id_a_mass_n
                col[k+4] = col_id2 + self.id_a_mass_perc_h
                k += 5
                noxbc0 += 1
                # 13 mass_perc_o
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0
                row[k+4] = row1 + noxbc0
                #
                col[k+0] = col_id2 + self.id_a_mass_c
                col[k+1] = col_id2 + self.id_a_mass_h
                col[k+2] = col_id2 + self.id_a_mass_o
                col[k+3] = col_id2 + self.id_a_mass_n
                col[k+4] = col_id2 + self.id_a_mass_perc_o
                k += 5
                noxbc0 += 1
                # 14 mass_perc_n
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0
                row[k+4] = row1 + noxbc0
                #
                col[k+0] = col_id2 + self.id_a_mass_c
                col[k+1] = col_id2 + self.id_a_mass_h
                col[k+2] = col_id2 + self.id_a_mass_o
                col[k+3] = col_id2 + self.id_a_mass_n
                col[k+4] = col_id2 + self.id_a_mass_perc_n
                k += 5
                noxbc0 += 1
                #
                # 15
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0

                col[k+0] = col_id2 + self.id_a_mass_f_h2
                col[k+1] = col_id2 + self.id_a_gcv
                k += 2
                noxbc0 += 1
                # 16
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0
                row[k+4] = row1 + noxbc0
                row[k+5] = row1 + noxbc0

                col[k+0] = col_id2 + self.id_a_mass_perc_c
                col[k+1] = col_id2 + self.id_a_mass_perc_h
                col[k+2] = col_id2 + self.id_a_mass_perc_o
                col[k+3] = col_id2 + self.id_a_mass_perc_n
                col[k+4] = col_id2 + self.id_a_gcv
                col[k+5] = col_id2 + self.id_a_fd
                k += 6
                noxbc0 += 1
                # 17
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0

                col[k+0] = col_id1_1 + self.id_surr_1_co
                col[k+1] = col_id2 + self.id_a_cco
                k += 2
                noxbc0 += 1
                # 18
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0

                col[k+0] = col_id1_1 + self.id_surr_1_nox
                col[k+1] = col_id2 + self.id_a_cnox
                k += 2
                noxbc0 += 1
                # 19
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0

                col[k+0] = col_id1_1 + self.id_surr_1_o2
                col[k+1] = col_id2 + self.id_a_fd
                col[k+2] = col_id2 + self.id_a_cco
                col[k+3] = col_id2 + self.id_a_eco
                k += 4
                noxbc0 += 1
                # 20
                row[k+0] = row1 + noxbc0
                row[k+1] = row1 + noxbc0
                row[k+2] = row1 + noxbc0
                row[k+3] = row1 + noxbc0

                col[k+0] = col_id1_1 + self.id_surr_1_o2
                col[k+1] = col_id2 + self.id_a_fd
                col[k+2] = col_id2 + self.id_a_cnox
                col[k+3] = col_id2 + self.id_a_enox
                k += 4
                noxbc0 += 1
                # 64 ###########################################################
                # 0
                nox_pow_row = noxbc0
                row[k+0] = row1 + nox_pow_row
                row[k+1] = row1 + nox_pow_row
                row[k+2] = row1 + nox_pow_row
                col[k+0] = col_id1_0 + self.id_surr_0_eff
                col[k+1] = col_id2 + self.id_a_eco
                col[k+2] = col_id2 + self.id_a_eco_pow
                k += 3
                nox_pow_row += 1
                # 1
                row[k+0] = row1 + nox_pow_row
                row[k+1] = row1 + nox_pow_row
                row[k+2] = row1 + nox_pow_row
                col[k+0] = col_id1_0 + self.id_surr_0_eff
                col[k+1] = col_id2 + self.id_a_enox
                col[k+2] = col_id2 + self.id_a_enox_pow
                k += 3
                nox_pow_row += 1
                # 64 ###########################################################
                mc_row = nox_pow_row
                # 0
                row[k+0] = row1 + mc_row
                row[k+1] = row1 + mc_row
                #
                col[k+0] = col_id1_0 + self.id_surr_0_ng_slpm
                col[k+1] = col_id2 + self.id_a_ng_kg
                k += 2
                mc_row += 1
                # 1
                row[k+0] = row1 + mc_row
                row[k+1] = row1 + mc_row
                #
                col[k+0] = col_id2 + self.id_a_h2_slpm
                col[k+1] = col_id2 + self.id_a_h2_kg
                k += 2
                mc_row += 1
                # 64 ###########################################################
                hc_row = mc_row
                # 0
                row[k+0] = row1 + hc_row
                row[k+1] = row1 + hc_row
                #
                col[k+0] = col_id2 + self.id_a_ng_kg
                col[k+1] = col_id2 + self.id_a_ng_h
                k += 2
                hc_row += 1
                # 1
                row[k+0] = row1 + hc_row
                row[k+1] = row1 + hc_row
                #
                col[k+0] = col_id2 + self.id_a_h2_kg
                col[k+1] = col_id2 + self.id_a_h2_h
                k += 2
                hc_row += 1
                # co2 w ### ####################################################
                co2w_row = hc_row
                # 0: cco2
                row[k+0] = row1 + co2w_row
                row[k+1] = row1 + co2w_row
                #
                col[k+0] = col_id1_1 + self.id_surr_1_co2
                col[k+1] = col_id2 + self.id_a_cco2
                k += 2
                co2w_row += 1
                # 1: eco2
                row[k+0] = row1 + co2w_row
                row[k+1] = row1 + co2w_row
                row[k+2] = row1 + co2w_row
                row[k+3] = row1 + co2w_row
                #
                col[k+0] = col_id1_1 + self.id_surr_1_o2
                col[k+1] = col_id2 + self.id_a_fd
                col[k+2] = col_id2 + self.id_a_cco2
                col[k+3] = col_id2 + self.id_a_eco2
                k += 4
                co2w_row += 1
                # 2: eco2_pow
                row[k+0] = row1 + co2w_row
                row[k+1] = row1 + co2w_row
                row[k+2] = row1 + co2w_row
                col[k+0] = col_id1_0 + self.id_surr_0_eff
                col[k+1] = col_id2 + self.id_a_eco2
                col[k+2] = col_id2 + self.id_a_eco2_pow
                k += 3
                co2w_row += 1
                # 3: wco2
                row[k+0] = row1 + co2w_row
                row[k+1] = row1 + co2w_row
                row[k+2] = row1 + co2w_row
                col[k+0] = col_id1_0 + self.id_surr_0_load
                col[k+1] = col_id2 + self.id_a_eco2_pow
                col[k+2] = col_id2 + self.id_a_wco2
                k += 3
                co2w_row += 1


            # 68 ###############################################################
            row_addt = row0_0 + self.c_u_block * self.nunits
            col_addt = col0_0 + self.v_u_block * self.nunits  #
            #
            # add_t 0th
            for u in range(self.nunits):
                cs0u = col0_0 + self.s_input_v + self.v_u_block * u
                row[k] = row_addt
                col[k] = cs0u + self.id_surr_0_load
                k += 1
            row[k] = row_addt
            col[k] = col_addt + self.id_t_tgen
            k += 1
            #
            # add_t 1st
            row[k] = row_addt + 1
            row[k+1] = row_addt + 1
            row[k+2] = row_addt + 1
            col[k] = col_addt + self.id_t_tgen
            col[k+1] = col_addt + self.id_t_pimp
            col[k+2] = col_addt + self.id_t_nimp
            k += 3
            #
            # add_t 2nd
            row[k] = row_addt + 2
            row[k+1] = row_addt + 2
            col[k] = col_addt + self.id_t_pimp
            col[k+1] = col_addt + self.id_t_imp_c
            k += 2
            # add_t 3nd imp_co2
            row[k+0] = row_addt + 3
            row[k+1] = row_addt + 3
            col[k+0] = col_addt + self.id_t_pimp
            col[k+1] = col_addt + self.id_t_imp_co2
            k += 2
            #
            # add_t 4rd
            for u in range(self.nunits):
                cau = col0_0 + self.s_input_v \
                    + self.s_output_v_0  + self.s_output_v_1 \
                    + self.v_u_block * u
                row[k] = row_addt + 4
                col[k] = cau + self.id_a_vonm
                k += 1
            row[k] = row_addt + 4
            col[k] = col_addt + self.id_t_tvonm
            k += 1
            #
            # add_t 5th
            for u in range(self.nunits):
                cs1u = col0_0 + self.s_input_v \
                    + self.s_output_v_0 + self.v_u_block * u
                row[k] = row_addt + 5
                col[k] = cs1u + self.id_surr_1_co
                k += 1
            row[k] = row_addt + 5
            col[k] = col_addt + self.id_t_tco
            k += 1
            #
            # add_t 6th
            for u in range(self.nunits):
                cs1u = col0_0 + self.s_input_v \
                    + self.s_output_v_0 + self.v_u_block * u
                row[k] = row_addt + 6
                col[k] = cs1u + self.id_surr_1_nox
                k += 1
            row[k] = row_addt + 6
            col[k] = col_addt + self.id_t_tnox
            k += 1


        #if self.generate_debug_file:
        #d = pd.DataFrame(row)
        #d.to_csv(self.debug_filename + "r_jacobian.csv")
        #d = pd.DataFrame(col)
        #d.to_csv(self.debug_filename + "c_jacobian.csv")

        return (row, col)

    def jacobian(self, x):
        inf = np.inf
        lmp = self.lmp
        costH2 = self.costH2
        costNg = self.costNg
        demand = self.demand

        x_i = np.zeros((self.nunits * self.thorz, self.s_input_v))
        #y0 = np.zeros((self.nunits * self.thorz, self.s_output_v_0))

        ##
        for time in range(self.thorz):
            col0_0 = self.v_tu_block * time
            for unit in range(self.nunits):
                col_id0 = col0_0 + self.v_u_block * unit
                # surrogate output block indices (variables)
                col_id1_0 = col_id0 + self.s_input_v  # surr output 0
                col_id1_1 = col_id1_0 + self.s_output_v_0  # surr output 1
                col_id2 = col_id1_1 + self.s_output_v_1
                col_id3 = col_id2 + self.add_v_u_block
                # extract the input values from x
                x_i[unit + time * self.nunits, :] = x[col_id0:col_id1_0]
                #y0[unit + time * self.nunits, :] = x[col_id1_0:col_id1_1]

        # evaluate the surrogate
        j0 = self.s_0.Jacobian(x_i, fidelity=1)

        delta = 1e-08

        x_i0p = x_i[:, 0] + delta
        x_i0m = x_i[:, 0] - delta

        x_i1p = x_i[:, 1] + delta
        x_i1m = x_i[:, 1] - delta


        ypp, _, _, _ = self.s_1.test(np.transpose(np.stack((x_i0p, x_i[:, 1]))),
                                    fidelity=2)
        ypm, _, _, _ = self.s_1.test(np.transpose(np.stack((x_i0m, x_i[:, 1]))),
                                    fidelity=2)

        yhp, _, _, _ = self.s_1.test(np.transpose(np.stack((x_i[:, 0], x_i1p))),
                                    fidelity=2)
        yhm, _, _, _ = self.s_1.test(np.transpose(np.stack((x_i[:, 0], x_i1m))),
                                    fidelity=2)


        dp = (ypp - ypm)/(2*delta)
        dh = (yhp - yhm)/(2*delta)

        j1 = np.stack((dp, dh))

        data = np.zeros(self.nnzJ)
        k = 0
        for time in range(self.thorz):
            col0_0 = self.v_tu_block * time
            for unit in range(self.nunits):
                for i in range(self.surr_block_0):
                    data[k] = j0[unit + time * self.nunits][i,0]
                    data[k+1] = j0[unit + time * self.nunits][i,1]
                    data[k+2] = -1
                    k += 3
                for i in range(self.surr_block_1):
                    data[k] = j1[0][unit + time * self.nunits,i]
                    data[k+1] = j1[1][unit + time * self.nunits,i]

                    data[k+2] = -1
                    k += 3
                col_id0 = col0_0 + self.v_u_block*unit  # input part
                col_id1_0 = col_id0 + self.s_input_v  # surr output 0
                col_id1_1 = col_id1_0 + self.s_output_v_0  # surr output 1
                col_id2 = col_id1_1 + self.s_output_v_1
                col_id3 = col_id2 + self.add_v_u_block
                # extract the input values from x
                x_i_surr = x[col_id0:col_id1_0]
                y_0 = x[col_id1_0:col_id1_1] # s0
                y_1 = x[col_id1_1:col_id2] # s1

                # add u
                x_add_v_u = x[col_id2:col_id3]
                # 0th
                data[k] = costNg[unit]  # id_surr_0_ng_slpm
                data[k+1] = costH2[unit] # id_a_h2_slpm
                data[k+2] = -1 # id_a_vonm
                k += 3
                # 1st
                #a = (1/100)*y0[unit+time*self.nunits,self.id_surr_0_ng_slpm]
                #b = pow(1-x_i[unit+time*self.nunits,self.id_in_xh2]/100, 2)
                #data[k] = a/b
                #data[k+1] = 1/(1-x_i[unit+time*self.nunits,self.id_in_xh2]/100)
                #data[k+2] = -1

                data[k+0] = x_add_v_u[self.id_a_tslp]/100. # self.id_in_xh2
                data[k+1] = 1. # self.id_surr_0_ng_slpm
                data[k+2] = x_i_surr[self.id_in_xh2]/100. - 1.
                # id_a_tslp

                k += 3

                # 2nd
                data[k] = -1 # id_surr_0_ng_slpm
                data[k+1] = 1 # id_a_tslp
                data[k+2] = -1 # id_a_h2_slpm
                k += 3
                # 3rd
                data[k] = self.mffc[0] #+ self.mfac[0] # iload
                data[k+1] = self.mffc[1]/100 # xh2
                data[k+2] = -1. # a_tmflow
                k += 3

                # nox lb #######################################################

                xnox_i = jnp.array([x_i_surr[1],
                           y_1[1],
                           y_1[2],
                           y_1[4],
                           ])
                nox_b_end = self.id_a_noxlbptr_ + self.nox_lb_n_block
                xnox_i = jnp.concatenate(
                    [xnox_i, x_add_v_u[self.id_a_noxlbptr_:nox_b_end]])

                j_list = jac_epa_19(xnox_i)
                nrow = 0
                for jrow in j_list:
                    for jidx in range(jrow.size):
                        data[k] = jrow[jidx] # value_j
                        k += 1
                    nrow += 1

                # nox lb pow ###################################################
                # enox_pow
                data[k] = -x_add_v_u[self.id_a_eco] \
                    / pow(y_0[self.id_surr_0_eff], 2)
                data[k+1] = 1e0/y_0[self.id_surr_0_eff]
                data[k+2] = -1.
                k += 3
                # co_pow
                data[k] = -x_add_v_u[self.id_a_enox] \
                    / pow(y_0[self.id_surr_0_eff], 2)
                data[k+1] = 1e0/y_0[self.id_surr_0_eff]
                data[k+2] = -1.
                k += 3

                # 64 ###########################################################
                x_mc_i = jnp.array([y_0[self.id_surr_0_ng_slpm],
                                    x_add_v_u[self.id_a_h2_slpm],
                                    x_add_v_u[self.id_a_ng_kg],
                                    x_add_v_u[self.id_a_h2_kg]
                                    ])

                j_mc = jac_mass_conv(x_mc_i)

                for jrow in j_mc:
                    for v in jrow:
                        data[k] = v
                        k += 1

                # 64 ###########################################################
                x_hc_i = jnp.array([x_add_v_u[self.id_a_ng_kg],
                                    x_add_v_u[self.id_a_h2_kg],
                                    x_add_v_u[self.id_a_ng_h],
                                    x_add_v_u[self.id_a_h2_h]])

                j_hc = jac_heat_conv(x_hc_i)

                for jrow in j_hc:
                    for v in jrow:
                        data[k] = v
                        k += 1

                # co2 w ### ####################################################
                # co2w 0: cco2
                data[k+0] = 1e-06*44.009/(0.730240*528.0) # id_surr_1_co2
                data[k+1] = -1.  # id_a_cco2
                k += 2

                # 1: eco2
                mmbtuToMwh = 3.4095106405145
                # id_surr_1_o2 0
                data[k+0] = \
                    mmbtuToMwh*1e-06*1e+02*x_add_v_u[self.id_a_cco2] \
                    *x_add_v_u[self.id_a_fd] \
                    *20.9/pow(20.9-y_1[self.id_surr_1_o2]*1e-06*1e+02, 2)
                # id_a_fd 1
                data[k+1] = mmbtuToMwh*x_add_v_u[self.id_a_cco2] \
                    *20.9/(20.9-y_1[self.id_surr_1_o2]*1e-06*1e+02)
                # id_a_cco2 2
                data[k+2] = mmbtuToMwh*x_add_v_u[self.id_a_fd] \
                    *20.9/(20.9-y_1[self.id_surr_1_o2]*1e-06*1e+02)
                # id_a_eco2 3
                data[k+3] = -1.
                k += 4

                # 2: eco2_pow
                # id_surr_0_eff 0
                data[k+0] = -x_add_v_u[self.id_a_eco2] \
                    /pow(y_0[self.id_surr_0_eff], 2)
                # id_a_eco2 1
                data[k+1] = 1e0 / y_0[self.id_surr_0_eff]
                # id_a_eco2_pow 2
                data[k+2] = -1.
                k += 3

                # 3: wco2
                kgToLb = 2.204623 # lb/kg
                mwhToKwh = 1e3  # kWh/MWh
                # id_surr_0_load 0
                data[k+0] = x_add_v_u[self.id_a_eco2_pow]/(kgToLb*mwhToKwh)
                # id_a_eco2_pow 1
                data[k+1] = y_0[self.id_surr_0_load]/(kgToLb*mwhToKwh)
                # id_a_wco2 2
                data[k+2] = -1.
                k += 3

            #
            # add_t 0th
            for u in range(self.nunits):
                data[k] = -1
                k += 1
            data[k] = 1
            k += 1
            #
            # add_t 1st
            data[k] = 1
            data[k+1] = 1
            data[k+2] = -1
            k += 3
            #
            # add_t 2nd
            data[k] = lmp[time]
            data[k+1] = -1
            k += 2
            #
            # add_t 3nd imp_co2
            data[k+0] = self.grid_co2ef[time]
            data[k+1] = -1.
            k += 2
            # add_t 4rd
            for u in range(self.nunits):
                data[k] = -1
                k += 1
            data[k] = 1
            k += 1
            #
            # add_t 5th
            for u in range(self.nunits):
                data[k] = -1
                k += 1
            data[k] = 1
            k += 1
            #
            # add_t 6th
            for u in range(self.nunits):
                data[k] = -1
                k += 1
            data[k] = 1
            k += 1

        #if self.generate_debug_file:
        #d = pd.DataFrame(data)
        #d.to_csv(self.debug_filename + "jacobian.csv")
            # self.generate_debug_file = False
        return data

    def hessian(self, x, lagrange, obj_factor):
        return None

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm, regularization_size,
                     alpha_du, alpha_pr, ls_trials):
        self._ic = iter_count
        self._inf_du = inf_du
        iterate = self.get_current_iterate()

        mxl = iterate["mult_x_L"]
        mxu = iterate["mult_x_U"]

        nl = np.linalg.norm(mxl, ord=np.inf)
        nu = np.linalg.norm(mxu, ord=np.inf)

        if nu > 1e8:
            print(f"nu = {nu}")
            #d = pd.DataFrame(mxu)
            #d.to_csv(self.debug_filename + f"nu-{iter_count}.csv")
            #d = pd.DataFrame(iterate["x"])
            #d.to_csv(self.debug_filename + f"xu-{iter_count}.csv")
        if nl > 1e8:
            print(f"nl = {nl}")
            #d = pd.DataFrame(mxl)
            #d.to_csv(self.debug_filename + f"nl-{iter_count}.csv")
            #d = pd.DataFrame(iterate["x"])
            #d.to_csv(self.debug_filename + f"xl-{iter_count}.csv")

        if iter_count % 100 == 0 and iter_count > 0:
            d = pd.DataFrame(iterate["x"])
            d.to_csv(self.debug_filename + f"x-{iter_count}.csv")
            infes = self.get_current_violations()
            d = pd.DataFrame(infes["g_violation"])
            d.to_csv(self.debug_filename + f"g-{iter_count}.csv")
            d = pd.DataFrame(infes["grad_lag_x"])
            d.to_csv(self.debug_filename + f"grad_lag-{iter_count}.csv")

        #``"mult_x_L"``, ``"mult_x_U"``, ``"g"``, and ``"mult_g"``
        if inf_du <= 1e-08 and self.print_low_inf_du_single_time:
            d = pd.DataFrame(iterate["x"])
            d.to_csv(self.debug_filename + f"infdu-x-{iter_count}.csv") #

            infes = self.get_current_violations()
            d = pd.DataFrame(infes["g_violation"])
            d.to_csv(self.debug_filename + f"infdu-g-{iter_count}.csv") #

            d = pd.DataFrame(infes["grad_lag_x"])
            d.to_csv(self.debug_filename + f"infdu-grad_lag-{iter_count}.csv") #
            self.print_low_inf_du_single_time = False
        #
        if inf_pr <= 1e-06 and self.print_low_inf_pr_single_time:
            d = pd.DataFrame(iterate["x"])
            d.to_csv(self.debug_filename + f"infpr-x-{iter_count}.csv") #

            infes = self.get_current_violations()
            d = pd.DataFrame(infes["g_violation"])
            d.to_csv(self.debug_filename + f"infpr-g-{iter_count}.csv") #

            d = pd.DataFrame(infes["grad_lag_x"])
            d.to_csv(self.debug_filename + f"infpr-grad_lag-{iter_count}.csv") #
            self.print_low_inf_pr_single_time = False

        if inf_pr <= 1e-08:
            d = pd.DataFrame(iterate["x"])
            d.to_csv(self.debug_filename + f"wow-x-{iter_count}.csv") #

        #if iter_count % 10 == 0:
        #    violation = self.get_current_violations()
        #    #xlv = violation["x_L_violation"]
        #    cxl = violation["compl_x_L"]
        #    print(cxl)
        #    #sys.exit()




