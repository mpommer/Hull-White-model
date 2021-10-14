# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:08:56 2021

@author: marce
"""

import numpy as np
from TrinomialTree import treeConstruction, calculateBondPrice


Zerocurve = np.array([[1, 0.03],
                      [2, 0.04],
                      [3, 0.05],
                      [4, 0.06],
                      [5, 0.07],
                      [6, 0.08]])


tree = treeConstruction(Zerocurve, lastDate=5, volatility=0.001, StepsPerYear = 72, a = 0)

cashflow = np.array([[2.0, 1.0]])
                     # [2., 0.0],
                     # [3., 1.0]])
                     
# cashflow = np.array([[1.0, 0.1],
#                       [2., 0.1],
#                       [3., 1.0]])

ExDates = np.array([[0., 0]])

BondPrice = calculateBondPrice(tree, cashflow, ExDates)

print(BondPrice)


print("Ana :", 1/(1 +0.035)**2)
print("Ana :", np.exp(-0.08))

test = tree.TrinomialTreeparameters

eld = test['Yield']
