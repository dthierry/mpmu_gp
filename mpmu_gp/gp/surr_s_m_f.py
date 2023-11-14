# -*- coding: utf-8 -*-

import numpy as np
import cyipopt


from chp_ed.mult_mod.MultiFidelity_Limit3 import MultiFidelity as mF

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as nrm

from datetime import datetime

import subprocess
import os
import json

__author__ = "David Thierry"

colours = ['lightcoral', 'orangered', 'darkolivegreen', 'steelblue']


class two_mod_surr(object):
    """this should load the surrogate(s)"""
    def __init__(self):
        pass
        #self.load_surrogate(file_0, file_1)

    def load_existing_surrogate(self, file_0: str, file_1: str):
        """Contains the trained model 0 and 1"""
        self.s_f_0 = file_0  # surrogate 0 (ngslpm, obspow, itt, eff)
        self.s_f_1 = file_1  # surrogate 1 (co2, co, nox, thc)
        self.s_0 = mF.loadModel(self.s_f_0)
        self.s_1 = mF.loadModel(self.s_f_1)

    def train_new_surrogate_1(self, input_data: str, out_f1: str):
        """Train new model"""
        train_perc_lf = 100 # Low-fidelity (CFD) training data percentage
        train_perc_hf = 100 # High-fidelity (Exp) training data percentage

        smoke_ci = False # Reduces training epochs to 200 if true

        deep = False

        encoder = False

        data = pd.read_excel(input_data,
                             sheet_name=1,
                             header=[0, 1])

        data_np = data.to_numpy()

        locs_of_interest = \
        data.loc[:,'Demand Power']==data.loc[:,'Demand Power']
        locs_of_interest = locs_of_interest.to_numpy()

        locs_of_interest = np.argwhere(locs_of_interest==True)[:,0]


        load_demand = np.zeros((len(locs_of_interest),1))
        load_actual = np.zeros((len(locs_of_interest),1))
        perc_h2 = np.zeros((len(locs_of_interest),1))

        co2_exp = np.zeros((len(locs_of_interest),1))
        co2_cfd_s = np.zeros((len(locs_of_interest),1))

        co_exp = np.zeros((len(locs_of_interest),1))
        co_cfd_s = np.zeros((len(locs_of_interest),1))

        nox_exp = np.zeros((len(locs_of_interest),1))
        nox_cfd_s = np.zeros((len(locs_of_interest),1))

        thc_exp = np.zeros((len(locs_of_interest),1))
        thc_cfd_s = np.zeros((len(locs_of_interest),1))

        o2_exp = np.zeros((len(locs_of_interest),1))
        o2_cfd_s = np.zeros((len(locs_of_interest),1))

        for i in range(len(locs_of_interest)):
            load_demand[i] = data_np[int(locs_of_interest[i]),5]
            load_actual[i] = data_np[int(locs_of_interest[i]),6]
            perc_h2[i] = data_np[int(locs_of_interest[i]),3]

            co2_exp[i] = data_np[int(locs_of_interest[i]),9]
            co2_cfd_s[i] = data_np[int(locs_of_interest[i]),16]

            co_exp[i] = data_np[int(locs_of_interest[i]),10]
            co_cfd_s[i] = data_np[int(locs_of_interest[i]),17]

            nox_exp[i] = data_np[int(locs_of_interest[i]),12]
            nox_cfd_s[i] = data_np[int(locs_of_interest[i]),19]

            thc_exp[i] = data_np[int(locs_of_interest[i]),13]
            thc_cfd_s[i] = data_np[int(locs_of_interest[i]),20]

            o2_exp[i] = data_np[int(locs_of_interest[i]),14]
            o2_cfd_s[i] = data_np[int(locs_of_interest[i]),21]

        data_x = np.column_stack((load_demand, perc_h2))

        param = ['$\mathregular{CO_2}$ (ppm)',
                 'CO (ppm)',
                 'NOx (ppm)',
                 'THC (ppm)']
        data_y_f1 = np.column_stack((co2_cfd_s[co2_cfd_s == co2_cfd_s],
                                      co_cfd_s[co2_cfd_s == co2_cfd_s],
                                      nox_cfd_s[co2_cfd_s == co2_cfd_s],
                                      thc_cfd_s[co2_cfd_s == co2_cfd_s],
                                      o2_cfd_s[co2_cfd_s == co2_cfd_s]))

        data_x_f1 = np.column_stack((data_x[:,None,0][co2_cfd_s == co2_cfd_s],
                                      data_x[:,None,1][co2_cfd_s == co2_cfd_s]))

        data_y_f2 = np.column_stack((co2_exp[co2_exp == co2_exp],
                                      co_exp[co2_exp == co2_exp],
                                      nox_exp[co2_exp == co2_exp],
                                      thc_exp[co2_exp == co2_exp],
                                      o2_exp[co2_exp == co2_exp]))

        data_x_f2 = np.column_stack((data_x[:,None,0][co2_exp == co2_exp],
                                      data_x[:,None,1][co2_exp == co2_exp]))


        train_set_f1 = np.random.permutation(data_y_f1.shape[0])
        test_set_f1 = train_set_f1[int(data_y_f1.shape[0]*train_perc_lf/100):]
        train_set_f1 = train_set_f1[:int(data_y_f1.shape[0]*train_perc_lf/100)]

        train_x_f1 = data_x_f1[train_set_f1,:]
        train_y_f1 = data_y_f1[train_set_f1,:]
        test_x_f1 = data_x_f1[test_set_f1,:]
        test_y_f1 = data_y_f1[test_set_f1,:]

        train_set_f2 = np.random.permutation(data_y_f2.shape[0])
        test_set_f2 = train_set_f2[int(data_y_f2.shape[0]*train_perc_hf/100):]
        train_set_f2 = train_set_f2[:int(data_y_f2.shape[0]*train_perc_hf/100)]

        train_x_f2 = data_x_f2[train_set_f2,:]
        train_y_f2 = data_y_f2[train_set_f2,:]
        test_x_f2 = data_x_f2[test_set_f2,:]
        test_y_f2 = data_y_f2[test_set_f2,:]

        num_inputs = data_x.shape[1]
        num_tasks = data_y_f1.shape[1]


        max_epochs = 200 if smoke_ci else 5000

        early_stopping_tol = [100, 0.00001]
        lrs = [0.01, 0.00001]
        restarts = [0, 400]

        #%% TRAINING MF MODEL

        self.s_1 = mF(num_inputs, num_tasks,
                                 deep_gp = deep,
                                 encoder = encoder, encoder_size = [4],
                      lbound=[0,0,0,0,0])

        self.s_1.train(train_x_f1, train_y_f1,
                       fidelity = 1,
                       epochs = max_epochs,
                       verbose = True,
                       early_stopping_tol = early_stopping_tol,
                       lrs = lrs, restarts = restarts)

        self.s_1.train(train_x_f2, train_y_f2,
                       fidelity = 2,
                       epochs = max_epochs,
                       verbose = True,
                       early_stopping_tol = early_stopping_tol,
                       lrs = lrs,
                       restarts = restarts)

        self.s_1.saveModel(filename=out_f1)

    def test_1(self):
        power_demand = 65
        h2_percentage = 100

        jacobian = self.s_1.Jacobian(np.array([power_demand,
                                               h2_percentage]).reshape((1,-1)),
                                     fidelity = 2)

        pred_y, *_ = self.s_1.test(np.array([power_demand,
                                             h2_percentage]).reshape((1,-1)),
                                   fidelity = 2)

        print(pred_y)

        print(jacobian)

        ##
        f, ax = plt.subplots(2,2, dpi=pow(2,8))
        f_jac_0, ax_jac_0 = plt.subplots(2,2, dpi=pow(2,8))
        f_jac_1, ax_jac_1 = plt.subplots(2,2, dpi=pow(2,8))
        n_points = 100
        cols = 4
        dpow_lb = 15
        dpow_ub = 65
        num = 5
        dpow_range = np.linspace(dpow_lb, dpow_ub, n_points)
        for i in np.linspace(1e-06, 70, num=num): # this is hydrogen
            h2p_range = np.ones(n_points) * i
            dpow_range = np.ones(n_points) * 50.

            test_dpow = np.transpose(np.stack([dpow_range, h2p_range]))
            m, _, _, _ = self.s_1.test(test_dpow, fidelity=2)
            d = pd.DataFrame(m)
            d.to_csv(f"m_{i}.csv")

            jac = self.s_1.Jacobian(
                test_dpow,
                fidelity=2)
            print(jac.size)

            dj0 = pd.DataFrame([jac[:][k, 0] for k in range(cols)])
            dj0.to_csv(f"dj0_{i}.csv")
            dj1 = pd.DataFrame([jac[:][k, 1] for k in range(cols)])
            dj1.to_csv(f"dj1_{i}.csv")

            #dj1 = pd.DataFrame(jac[:])
            #dj1.to_csv(f"dj1_{i}.csv")

            for col in range(cols):
                if col == 0:
                    name = 'co2'
                    l = (0, 0)
                    j0a = 0
                    j0b = 0

                    j1a = 0
                    j1b = 1
                elif col == 1:
                    name = 'co'
                    l = (0, 1)
                    j0a = 1
                    j0b = 0

                    j1a = 1
                    j1b = 1
                elif col == 2:
                    name = 'nox'
                    l = (1, 0)
                    j0a = 2
                    j0b = 0

                    j1a = 2
                    j1b = 1
                elif col == 3:
                    name = 'thc'
                    l = (1, 1)
                    j0a = 3
                    j0b = 0

                    j1a = 3
                    j1b = 1

                a = ax[l]

                xv = dpow_range

                a.plot(m[:, col],
                       # label='{:4.1} kW'.format(i))
                       ls="solid",
                       label='{:6.1} %h2'.format(i))

                aj_0 = ax_jac_0[l]
                aj_1 = ax_jac_1[l]

                aj_0.plot(jac[:, j0a, j0b], ls="solid",
                          label="{:6.1} %h2".format(i))
                aj_1.plot(jac[:, j1a, j1b], ls="solid",
                          label="{:6.1} %h2".format(i))

        for col in range(cols):
            if col == 0:
                name = 'co2'
                l = (0, 0)
            elif col == 1:
                name = 'co'
                l = (0, 1)
            elif col == 2:
                name = "nox"
                l = (1, 0)
            elif col == 3:
                name = 'thc'
                l = (1, 1)

            a = ax[l]
            a.legend()
            a.set_title(name)

            aj_0 = ax_jac_0[l]
            aj_1 = ax_jac_1[l]

            aj_0.legend()
            aj_0.set_title(name)

            aj_1.legend()
            aj_1.set_title(name)

        f.tight_layout()

        f_jac_0.tight_layout()
        f_jac_1.tight_layout()

        f.savefig("output_1_0.png")

        f_jac_0.savefig("jac10_0.png")
        f_jac_1.savefig("jac11_0.png")


    ####
    def train_new_surrogate_0(self, input_data: str, out_f0: str):
        train_perc_lf = 100 # Low-fidelity (CFD) training data percentage
        train_perc_hf = 100 # High-fidelity (Exp) training data percentage

        smoke_ci = False #


        deep = False

        encoder = False

        data = pd.read_excel(input_data,
                             sheet_name=1,
                             header=[0, 1])

        data_np = data.to_numpy()

        locs_of_interest = \
        data.loc[:,'Demand Power']==data.loc[:,'Demand Power']
        locs_of_interest = locs_of_interest.to_numpy()

        locs_of_interest = np.argwhere(locs_of_interest==True)[:,0]


        load_demand = np.zeros((len(locs_of_interest), 1)) # input 0
        perc_h2 = np.zeros((len(locs_of_interest), 1)) # input 1

        load_actual = np.zeros((len(locs_of_interest), 1)) # 0
        ng_slpm = np.zeros((len(locs_of_interest), 1)) # 1
        itt = np.zeros((len(locs_of_interest), 1)) # 2
        eff = np.zeros((len(locs_of_interest), 1)) # 3


        for i in range(len(locs_of_interest)):
            load_demand[i] = data_np[int(locs_of_interest[i]), 5]
            perc_h2[i] = data_np[int(locs_of_interest[i]), 3]

            load_actual[i] = data_np[int(locs_of_interest[i]), 6]
            ng_slpm[i] = data_np[int(locs_of_interest[i]), 2]
            eff[i] = data_np[int(locs_of_interest[i]), 7]
            itt[i] = data_np[int(locs_of_interest[i]), 8]


        data_x = np.column_stack((load_demand, perc_h2))


        data_y_f1 = np.column_stack((ng_slpm[eff == eff],
                                      load_actual[eff == eff],
                                      eff[eff == eff],
                                      itt[eff == eff]))

        data_x_f1 = np.column_stack((data_x[:,None,0][eff == eff],
                                      data_x[:,None,1][eff == eff]))



        train_set_f1 = np.random.permutation(data_y_f1.shape[0])
        test_set_f1 = train_set_f1[int(data_y_f1.shape[0]*train_perc_lf/100):]
        train_set_f1 = train_set_f1[:int(data_y_f1.shape[0]*train_perc_lf/100)]

        train_x_f1 = data_x_f1[train_set_f1,:]
        train_y_f1 = data_y_f1[train_set_f1,:]

        test_x_f1 = data_x_f1[test_set_f1,:]
        test_y_f1 = data_y_f1[test_set_f1,:]

        num_inputs = data_x.shape[1]
        num_tasks = data_y_f1.shape[1]


        max_epochs = 200 if smoke_ci else 5000

        early_stopping_tol = [100, 0.00001]
        lrs = [0.01, 0.00001]
        restarts = [0, 400]


        self.s_0 = mF(num_inputs,
                                 num_tasks,
                                 deep_gp=deep,
                                 encoder=encoder,
                                 encoder_size=[4],
                      lbound=None)

        self.s_0.train(train_x_f1, train_y_f1,
                       fidelity = 1,
                       epochs = max_epochs,
                       verbose = True,
                       early_stopping_tol = early_stopping_tol,
                       lrs = lrs, restarts = restarts)
                       #lbound_trans=None)


        self.s_0.saveModel(filename=out_f0)

    # 76
    # 76 #######################################################################
    def test_0(self):

        num = 5
        # initialize subplots
        f, ax = plt.subplots(2,2, dpi=pow(2,8))
        f_jac_0, ax_jac_0 = plt.subplots(2,2, dpi=pow(2,8))
        f_jac_1, ax_jac_1 = plt.subplots(2,2, dpi=pow(2,8))
        n_points = 100

        dpow_lb = 15
        dpow_ub = 65
        cols = 4
        #h2p_range = np.linspace(1e-5, 80, n_points)
        dpow_range = np.linspace(dpow_lb, dpow_ub, n_points)

        #for i in np.linspace(15, 65, num=num):
        for i in np.linspace(1e-06, 70, num=num): # this is hydrogen
            h2p_range = np.ones(len(dpow_range)) * i
            #dpow_range = np.ones(n_points) * i
            test_dpow = np.transpose(np.stack([dpow_range, h2p_range]))
            m, _, _, _ = self.s_0.test(test_dpow, fidelity=1)

            jac = self.s_0.Jacobian(
                test_dpow,
                fidelity=1)

            for col in range(cols):
                if col == 0:
                    name = 'ng'
                    l = (0, 0)
                    j0a = 0
                    j0b = 0

                    j1a = 0
                    j1b = 1
                elif col == 1:
                    name = 'load_actual'
                    l = (0, 1)
                    j0a = 1
                    j0b = 0

                    j1a = 1
                    j1b = 1
                elif col == 2:
                    name = 'eff'
                    l = (1, 0)
                    j0a = 2
                    j0b = 0

                    j1a = 2
                    j1b = 1
                elif col == 3:
                    name = 'itt'
                    l = (1, 1)
                    j0a = 3
                    j0b = 0

                    j1a = 3
                    j1b = 1

                a = ax[l]

                xv = dpow_range

                a.plot(xv, m[:, col],
                       # label='{:4.1} kW'.format(i))
                       label='{:6.1} %h2'.format(i))

                aj_0 = ax_jac_0[l]
                aj_1 = ax_jac_1[l]

                aj_0.plot(xv, jac[:, j0a, j0b], label="{:6.1} %h2".format(i))
                aj_1.plot(xv, jac[:, j1a, j1b], label="{:6.1} %h2".format(i))

        for col in range(cols):
            if col == 0:
                name = 'ng'
                l = (0, 0)
            elif col == 1:
                name = 'load_actual'
                l = (0, 1)
            elif col == 2:
                name = 'eff'
                l = (1, 0)
            elif col == 3:
                name = 'itt'
                l = (1, 1)

            a = ax[l]
            a.legend()
            a.set_title(name)

            aj_0 = ax_jac_0[l]
            aj_1 = ax_jac_1[l]

            aj_0.legend()
            aj_0.set_title(name)

            aj_1.legend()
            aj_1.set_title(name)

        f.tight_layout()

        f_jac_0.tight_layout()
        f_jac_1.tight_layout()

        f.savefig("output__0.png")

        f_jac_0.savefig("jac0_0.png")
        f_jac_1.savefig("jac1_0.png")


    # 76
    # 76 #######################################################################
    def test_all(self):
        power_demand = 65
        h2_percentage = 100

        jac_0 = self.s_0.Jacobian(np.array([power_demand,
                                               h2_percentage]).reshape((1,-1)),
                                     fidelity=1)

        pred_y0, *_ = self.s_0.test(np.array([power_demand,
                                             h2_percentage]).reshape((1,-1)),
                                   fidelity=1)

        print(pred_y0)
        print(jac_0)

        jac_1 = self.s_1.Jacobian(np.array([power_demand,
                                               h2_percentage]).reshape((1,-1)),
                                     fidelity = 2)

        pred_y1, *_ = self.s_1.test(np.array([power_demand,
                                             h2_percentage]).reshape((1,-1)),
                                   fidelity = 2)

        print(pred_y1)
        print(jac_1)

    def compare_grads(self):
        cmap = plt.get_cmap("Oranges")
        norm = nrm(vmin=-5, vmax=70)

        n_points = 100
        f, ax = plt.subplots(1,5, dpi=pow(2,8))
        f_jac_0, ax_jac_0 = plt.subplots(1,5, dpi=pow(2,8))
        f_jac_1, ax_jac_1 = plt.subplots(1,5, dpi=pow(2,8))

        dpow_lb = 15
        dpow_ub = 65
        cols = 4
        #h2p_range = np.linspace(1e-5, 80, n_points)
        dpow_range = np.linspace(dpow_lb, dpow_ub, n_points)
        delta = 1e-08
        num = 5
        fidelity=2
        #for i in np.linspace(15, 65, num=num):
        for i in np.linspace(1e-06, 70, num=num): # this is hydrogen
            c = cmap(norm(i))
            h2p_range = np.ones(len(dpow_range)) * i
            #dpow_range = np.ones(n_points) * i
            x0 = np.transpose(np.stack([dpow_range, h2p_range]))
            xp = np.transpose(np.stack([dpow_range+delta, h2p_range]))
            xh = np.transpose(np.stack([dpow_range, h2p_range+delta]))


            y0, _, _, _ = self.s_1.test(x0, fidelity=fidelity)
            yp, _, _, _ = self.s_1.test(xp, fidelity=fidelity)
            yh, _, _, _ = self.s_1.test(xh, fidelity=fidelity)

            dp = (yp - y0)/delta  # n by 4
            dh = (yh - y0)/delta

            jac = self.s_1.Jacobian(
                x0,
                fidelity=fidelity)  # n by (4, 2)
            dj0 = pd.DataFrame({k: jac[:, k, 0] for k in range(cols)})
            dj0.to_csv(f"dj0_{i}.csv")
            dj1 = pd.DataFrame({k: jac[:, k, 1] for k in range(cols)})
            dj1.to_csv(f"dj1_{i}.csv")


            for col in range(cols):
                if col == 0:
                    name = 'ng'
                    name = 'co2'
                    l = (0, 0)
                    j0a = 0
                    j0b = 0

                    j1a = 0
                    j1b = 1
                elif col == 1:
                    name = 'load_actual'
                    name = 'co'
                    l = (0, 1)
                    j0a = 1
                    j0b = 0

                    j1a = 1
                    j1b = 1
                elif col == 2:
                    name = 'eff'
                    name = 'nox'
                    l = (0, 2)
                    j0a = 2
                    j0b = 0

                    j1a = 2
                    j1b = 1
                elif col == 3:
                    name = 'itt'
                    name = 'thc'
                    l = (0, 3)
                    j0a = 3
                    j0b = 0

                    j1a = 3
                    j1b = 1

                a = ax[l]

                xv = dpow_range

                a.plot(xv, y0[:, col],
                       # label='{:4.1} kW'.format(i))
                       label='{:6.1} %h2'.format(i), color=c)

                aj_0 = ax_jac_0[l]
                aj_1 = ax_jac_1[l]

                aj_0.plot(xv,
                          jac[:, j0a, j0b], label="{:6.1} %h2".format(i),
                          ls="--", color=c)
                aj_0.plot(xv, dp[:, col], label="{:6.1} %h2 FD".format(i),
                          marker="x", markersize=1, color=c,
                          ls="None")

                aj_1.plot(xv,
                          jac[:, j1a, j1b], label="{:6.1} %h2".format(i),
                          ls="--", color=c)
                aj_1.plot(xv, dh[:, col], label="{:6.1} %h2 FD".format(i),
                          marker="x", markersize=1, color=c,
                          ls="None")


        for col in range(cols):
            if col == 0:
                name = 'ng'
                name = 'co2'
                l = (0, 0)
            elif col == 1:
                name = 'load_actual'
                name = 'co'
                l = (0, 1)
            elif col == 2:
                name = 'eff'
                name = 'nox'
                l = (0, 3)
            elif col == 3:
                name = 'itt'
                name = 'thc'
                l = (0, 4)

            a = ax[l]
            a.set_title(name)

            aj_0 = ax_jac_0[l]
            aj_1 = ax_jac_1[l]

            aj_0.set_title(name)
            aj_1.set_title(name)

            aj_0.get_xaxis().set_ticks([])
            aj_1.get_xaxis().set_ticks([])

            aj_0.set_xlabel("power")
            aj_1.set_xlabel("power")

        f.tight_layout()
        f.suptitle("output", fontsize="x-small")

        f_jac_0.tight_layout()
        f_jac_1.tight_layout()
        f_jac_0.suptitle("df/dipow", fontsize="x-small")
        f_jac_1.suptitle("df/dh2%", fontsize="x-small")

        f.savefig("output_0.png")

        for a in ax.flat:
            a.legend(fontsize="x-small")
        f.savefig("output_0_legend.png")

        f_jac_0.savefig("jac0_0vnum.png")
        f_jac_1.savefig("jac1_0vnum.png")

        for a in ax_jac_0.flat:
            a.legend(fontsize="x-small")
        for a in ax_jac_1.flat:
            a.legend(fontsize="x-small")

        f_jac_0.savefig("jac0_0vnum_legend.png")
        f_jac_1.savefig("jac1_0vnum_legend.png")

