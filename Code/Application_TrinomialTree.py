# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:08:56 2021

@author: marcel Pommer
"""

import numpy as np
from TrinomialTree import treeConstruction, calculateBondPrice
from tabulate import tabulate

Zerocurve = np.array([[1, 0.03],
                      [2, 0.04],
                      [3, 0.05],
                      [4, 0.06],
                      [5, 0.07],
                      [6, 0.08]])


tree = treeConstruction(Zerocurve, lastDate=5, volatility=0.001, StepsPerYear = 36, a = 0)

paymentTimes = []
cashFlows = []
analyticPrices = []
for index in range(5):
    paymentTimes.append(index+1)
    cashFlows.append(np.array([[index+1, 1.0]]))
    
    analyticPrices.append(np.exp(Zerocurve[::,1][index]*(-(index+1))))

                     

ExDates = np.array([[0., 0.0]])


numericPrices = []
relativeError = []
for index, cashFlow in enumerate(cashFlows):
    numericPrice = calculateBondPrice(tree, cashFlow, ExDates)
    numericPrices.append(numericPrice)
    
    relativeError.append((numericPrice-analyticPrices[index])/analyticPrices[index] * 100)


data = np.array([paymentTimes, analyticPrices, numericPrices, relativeError]).T

colNames = ['Payment Time', 'Analytic Price', 'Numeric Price', 'Relative Error']

print(tabulate(data, headers = colNames))

