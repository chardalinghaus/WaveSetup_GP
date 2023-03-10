# -*- coding: utf-8 -*-
"""
GpLearn model implementation to identify a mathematical expression that best describes wave setup. 
The GpLearn model parameters can be changed according its documentation (https://gplearn.readthedocs.io).

The Maximum Dissimilarity Algorithm (MDA) from HYWAVES (https://github.com/ripolln/hywaves) was used to select the training data.

@author: Charline Dalinghaus
"""

from MDA.mda import MaxDiss_Simplified_NoThreshold
from MDA.plots.mda import Plot_MDA_Data

import os
import glob
import pandas as pd
import numpy as np 
import time

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sympy import *
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#%% 
## =========================== Part 1: Reading Data ===========================

filepaths = glob.glob(os.path.join("StockdonData", "*.txt"))
dfs = [pd.read_csv(filepath, sep="\t", header=None, names=["r2", "η", "Stt", "Sinc", "Sig", "H0", "Tp", "βf", "D50"]) for filepath in filepaths]
data = pd.concat(dfs, ignore_index=True)
    
data ['D50'] /= 1000 # change D50 mm to m

#%%
## ========================= Part 2: Add New Variables ========================

# wavelength
data ['L0'] = 1.56*(data['Tp']**2)
# same as: data ['L0']= (9.8 * np.power(data.Tp,2)) / (2*math.pi) # import math

# irribaren number
data ['ξ0'] = data['βf']/((data['H0']/data['L0'])**0.5)

#%% 
## ============== Part 3: Maximum Dissimilarity Algorithm (MDA) ===============

# variables to use
vns = ['η', 'H0', 'Tp', 'βf', 'L0', 'ξ0', 'D50']

# subset size and scalar index
n_subset = 150  # subset size ~30% data = TRAINING DATA
ix_scalar = [0, 1, 2, 3, 4, 5, 6]  # n, Hs0, Tp, tanB, L0, Irr, d50

# MDA returns denormalize data - TRAIN and TEST set
train, test  = MaxDiss_Simplified_NoThreshold(data[vns].values[:], n_subset, ix_scalar)
train = pd.DataFrame(data=train, columns=vns)
test = pd.DataFrame(data=test, columns=vns)

# plot classification
Plot_MDA_Data(test, train)
plt.show()
# plt.savefig("MDA/MDA.png", dpi=300)

#%% 
## ============= Part 4: Normalize Data / Train and Testset Data ==============

# define columns and corresponding maximum values for normalization
cols = ['H0', 'Tp', 'βf', 'D50', 'L0', 'ξ0']
maxvalue = {
    'H0': 4.08,
    'Tp': 17,
    'βf': 0.16,
    'D50': 0.002,
    'L0': 450.84000000000003,
    'ξ0': 3.2491285688847187
}

# normalize data dividing by the maximum / separate train and test data
X_train = pd.DataFrame({col: train[col] / maxvalue[col] for col in cols})
X_test = pd.DataFrame({col: test[col] / maxvalue[col] for col in cols})

y_train = train ['η']
y_test = test ['η']

#%% 
## ========================== Part 5: Set Functions ===========================

# def pow_2(x):
#     f = x**2
#     return f
# pow_2 = make_function(function=pow_2,name='pow2',arity=1)

# def pow_3(x):
#     f = x**3
#     return f
# pow_3 = make_function(function=pow_3,name='pow3',arity=1)

function_set = ['add', 'sub', 'mul', 'div']#, 'sqrt', 'neg', pow_2, pow_3]
feature_names = ['H0', 'Tp', 'βf', 'D50', 'L0', 'ξ0']

#%%
## ======================== Part 6: Run GpLearn Model =========================

est_gp = SymbolicRegressor(population_size= 5000, generations=1000,
                            tournament_size=20, stopping_criteria=0.01,
                            const_range=(-5., 5.), init_depth=(2, 6),
                            init_method='half and half', function_set=function_set,
                            metric='mean absolute error', parsimony_coefficient=0.0005,
                            p_crossover=0.7,  p_subtree_mutation=0.1,                                      
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=1, verbose=1,
                            feature_names=feature_names,
                            warm_start=False, low_memory=True,
                            n_jobs=1, random_state=0)

t0 = time.time()
est_gp.fit(X_train, y_train)
print('Time to fit:', time.time() - t0, 'seconds')

###

print('R2:', est_gp.score(X_test,y_test))
print('Equation:', est_gp._program)

converter = {
    'add': lambda x, y : x + y,
    'sub': lambda x, y : x - y,
    'mul': lambda x, y : x*y,
    'div': lambda x, y : x/y,
    'sqrt': lambda x : x**0.5,
    'neg': lambda x : -x,
    'pow2': lambda x : x**2,
    'pow3': lambda x : x**3
}

equation = sympify(str(est_gp._program), locals=converter)
print ('Equation:', equation, '\n')

#%% 
## ========= Part 7: Visualize Measured X Predicted Setup (Test Set) ==========

y_test_predicted = est_gp.predict(X_test)

# least squares polynomial fit 
plt.figure(2)
plt.plot(y_test, y_test_predicted, '*k')
plt.xlim(0, 1.6), plt.ylim(0, 1.6)
plt.xlabel ('Measured Wave Setup (m) (Test Set)'); plt.ylabel ('Predicted Wave Setup (m) (Test Set)')
plt.title('GpLearn Eq. 11')

# best fit line (1:1)
plt.plot([0, 1.6], [0, 1.6], '--r')
plt.show()

#%% 
## ======================= Part 8: Metrics (Test Set) =========================

# coefficient of correlation r2
r2_gplearn_testset = np.corrcoef(y_test, y_test_predicted)[0, 1]**2
print('Coefficient of correlation r2 - GpLearn Eq.11 (Test Set): %.2f' % r2_gplearn_testset)

# coefficient of determination R2
R2_gplearn_testset = r2_score(y_test, y_test_predicted)
print('Coefficient of determination R2 - GpLearn Eq.11 (Test Set): %.2f' % R2_gplearn_testset)

def m_index_agreement(o, p):
    """
	index of agreement
	
	Willmott (1981, 1982) 
	input:
        o: observed
        p: predicted
    output:
        mia: index of agreement
    """
    mia = 1 -(np.sum(np.abs(p-o)))/(np.sum(
    			(np.abs(p-np.mean(o))+np.abs(o-np.mean(o)))))
    return mia

# modified index of agreement
d1_gplearn_testset = m_index_agreement(y_test, y_test_predicted)
print('Modified Index of Agreement - GpLearn Eq.11 (Test Set): %.2f' % d1_gplearn_testset)

# mean absolute error
MAE_gplearn_testset = mean_absolute_error(y_test, y_test_predicted)
print('Mean Absolute Error - GpLearn Eq.11 (Test Set): %.2f' % MAE_gplearn_testset)

# root mean squared error
RMSE_gplearn_testset = mean_squared_error(y_test, y_test_predicted, squared=False)
print('Root Mean Squared Error - GpLearn Eq.11 (Test Set): %.2f' % RMSE_gplearn_testset)

#%% 
## =============== Part 9: Reproducing the Formula (Test Set) =================

# mul(H0, add(add(div(ξ0, add(D50, ξ0)), ξ0), div(ξ0, add(0.197, ξ0))))
# H0*(ξ0 + ξ0/(ξ0 + 0.197) + ξ0/(D50 + ξ0)) 

gplearn_test = X_test['H0']*(X_test['ξ0'] + X_test['ξ0']/(X_test['ξ0'] + 0.197) + X_test['ξ0']/(X_test['D50'] + X_test['ξ0'])) 

# plot test set
plt.figure(3)
plt.plot(gplearn_test, '*k', label = 'Equation (Test Set)'); plt.legend(loc = 'best')
plt.plot(y_test_predicted, '.r', label = 'GpLearn (Test Set)'); plt.legend(loc = 'best')
plt.xlabel ('Sample ID'); plt.ylabel ('Predicted Wave Setup (m) (Test Set)');
plt.show()

#%%
## =============== Part 10: Sensitivity Analysis (All Data Set) ===============

# final equation denormalized
# = H0/4.08*(ξ0/3.2491285688847187 + ξ0/3.2491285688847187/(ξ0/3.2491285688847187 + 0.197) + ξ0/3.2491285688847187/(D50/0.002 + ξ0/3.2491285688847187)) 

def sensitivity_analysis(data, variable):
    # set up plotting parameters
    plt.figure()
    plt.xlabel(f'{variable} (m)')
    plt.ylabel('Predicted Wave Setup (m) - Eq.11')
    plt.title('Sensitivity Analysis')

    # get range of variable values and calculate other variables
    p1 = np.min(data[variable])
    p2 = np.max(data[variable])
    var = np.linspace(p1/2, 2*p2, 100)
    H0 = np.mean(data['H0'])
    ξ0 = np.mean(data['ξ0'])
    D50 = np.mean(data['D50'])

    # calculate formula for sensitivity analysis
    formula_SA = "H0/4.08*(ξ0/3.2491285688847187 + ξ0/3.2491285688847187/(ξ0/3.2491285688847187 + 0.197) + ξ0/3.2491285688847187/(D50/0.002 + ξ0/3.2491285688847187))" 
    formula_SA = formula_SA.replace(variable, 'var')
    formula_SA = eval(formula_SA)
    
    # plot sensitivity analysis
    mydata = var[(p1 <= var) & (var <= p2)]
    mydataFormula = formula_SA[(p1 <= var) & (var <= p2)]
    plt.plot(var, formula_SA, '*-k', label = 'Extrapolated Data')
    plt.plot(mydata, mydataFormula, '*-r', label = 'Measured Data')
    plt.legend(loc = 'best')
    

# plot sensitivity analysis for each variable
sensitivity_analysis(data, 'H0')
sensitivity_analysis(data, 'ξ0')
sensitivity_analysis(data, 'D50')

#%%
## ================ Part 11: Metrics and Plot (All Data Set) ==================

# Final equation simplified (Equation 11)
# = H0/4.08*(ξ0/3.25 + ξ0/(ξ0 + 0.64) + ξ0/(1650*D50 + ξ0)) 

gplearn_eq11 = data ['H0']/4.08*(data ['ξ0']/3.25 + data ['ξ0']/(data ['ξ0'] + 0.64) + data ['ξ0']/(1650*data ['D50'] + data ['ξ0']))

# coefficient of correlation r2
r2_gplearn_alldata = np.corrcoef(data ['η'], gplearn_eq11)[0, 1]**2
print('Coefficient of correlation r2 - GpLearn Eq.11: %.2f' % r2_gplearn_alldata)

# coefficient of determination R2
R2_gplearn_alldata = r2_score(data ['η'], gplearn_eq11)
print('Coefficient of determination R2 - GpLearn Eq.11: %.2f' % R2_gplearn_alldata)

# modified index of agreement
d1_gplearn_alldata = m_index_agreement(data ['η'], gplearn_eq11)
print('Modified Index of Agreement - GpLearn Eq.11: %.2f' % d1_gplearn_alldata)

# mean absolute error
MAE_gplearn_alldata = mean_absolute_error(data ['η'], gplearn_eq11)
print('Mean Absolute Error - GpLearn Eq.11: %.2f' % MAE_gplearn_alldata)

# root mean squared error
RMSE_gplearn_alldata = mean_squared_error(data ['η'], gplearn_eq11, squared=False)
print('Root Mean Squared Error - GpLearn Eq.11: %.2f' % RMSE_gplearn_alldata)


# plot
# least squares polynomial fit 
plt.figure(7)
plt.plot(data ['η'], gplearn_eq11, '*k')
plt.xlim(0, 2.0), plt.ylim(0, 2.0)
plt.xlabel ('Measured Wave Setup (m)'); plt.ylabel ('Predicted Wave Setup (m) - Eq.11')

# best fit line (1:1)
plt.plot([0, 1.6], [0, 1.6], '--r')
plt.show()