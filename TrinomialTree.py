# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:02:58 2021

@author: marce
"""
import numpy as np
from copy import copy


class TrinomialTree:
    
    def __init__(self, StepsPerYear, lastDate, Zerocurve):
        self.StepsPerYear = StepsPerYear
        self.lastDate = lastDate
        self.Zerocurve = Zerocurve
        
        self.NodeTimes, self.TermStructureDates = self.SetUpDates(Zerocurve)
        self.TotalNumberofNodes = self.NodeTimes.shape[0]
        
        
    def SetUpDates(self, Zerocurve):
        self.RateDates = copy(Zerocurve)
        
        index = np.argwhere(Zerocurve[:,0]>self.lastDate)[0][0]
        self.RateDates = self.RateDates[:index, :]
        
        dt = float(1/self.StepsPerYear)

        numberOfSteps = int((self.lastDate)/dt +0.5)
        dates = [0]
        dates += list(np.arange(1, numberOfSteps)*dt)
            
        dates.append(self.lastDate)        # last date needs to be appended
        dates = np.array(dates, dtype = object)
        
        self.NodeTimes = np.zeros((dates.shape[0] +1))
        self.NodeTimes[:-1] = dates
        self.NodeTimes[-1] = 2*self.NodeTimes[-2] - self.NodeTimes[-3]
        
        return self.NodeTimes, self.RateDates
    
    def BuildTree(self, a, volatility):
        self.a = float(a)
        self.volatility = volatility
        
        # initialize the diccionary        
        self.TrinomialTreeparameters = {'t' : np.zeros((self.TotalNumberofNodes)),
                                'volatility' : np.zeros((self.TotalNumberofNodes)),
                                'dR' : np.zeros((self.TotalNumberofNodes)),
                                'dt' : np.zeros((self.TotalNumberofNodes)),
                                'minj' : np.zeros((self.TotalNumberofNodes)).astype(int),
                                'numberOfJNodes' : np.zeros((self.TotalNumberofNodes)).astype(int),
                                'numberOfRates' : np.zeros((self.TotalNumberofNodes)).astype(int),
                                'Alpha' : np.zeros((self.TotalNumberofNodes)),
                                'pu' : np.zeros((self.TotalNumberofNodes), dtype = object),
                                'pd' : np.zeros((self.TotalNumberofNodes), dtype = object),
                                'k' : np.zeros((self.TotalNumberofNodes), dtype = object),
                                'ArrowDebrauPrices' : np.zeros((self.TotalNumberofNodes), dtype = object),
                                'OptionPrice' : np.zeros((self.TotalNumberofNodes), dtype = object),
                                'Yield' : np.zeros((self.TotalNumberofNodes), dtype = object)}
        
        # initialize some parameters
        # times
        self.TrinomialTreeparameters['t'] = self.NodeTimes
        # vola
        self.TrinomialTreeparameters['volatility'] = np.full((1, self.TotalNumberofNodes), self.volatility)[0]
        # delta times
        self.TrinomialTreeparameters['dt'][:-1] = self.TrinomialTreeparameters['t'][1:] - self.TrinomialTreeparameters['t'][:-1]
        # vertical spacing dR = sigma * sqrt(3* \Delta t)
        self.TrinomialTreeparameters['dR'][1:] = self.TrinomialTreeparameters['volatility'][1:] \
            * np.sqrt(3* self.TrinomialTreeparameters['dt'][:-1])
        
        
        # now build the tree following the two steps proposed by hull and white
        # step 1
        self.TreeStructure()
        
        #step 2
        self.TreeAdjustments()
        
        # now build the numeric rates
        self.calculateNumericrates()
        
        
    def TreeStructure(self):
        # initlize parameters at the first node
        numberverticalNodes = 1
        self.TrinomialTreeparameters['k'][0] = np.zeros((numberverticalNodes)).astype(int)
        self.TrinomialTreeparameters['pu'][0] = np.zeros((numberverticalNodes))
        self.TrinomialTreeparameters['pd'][0] = np.zeros((numberverticalNodes))
        self.TrinomialTreeparameters['ArrowDebrauPrices'][0] = np.zeros((numberverticalNodes))
        self.TrinomialTreeparameters['OptionPrice'][0] = np.zeros((numberverticalNodes))
        self.TrinomialTreeparameters['Yield'][0] = np.zeros((numberverticalNodes), dtype=object)
        
        self.TrinomialTreeparameters['numberOfJNodes'][0] = numberverticalNodes
        # minj is zero since we only have one node
        self.TrinomialTreeparameters['minj'][0] = 0
        # we initialize alpha with zero and adjust it in the second step
        self.TrinomialTreeparameters['Alpha'][0] = 0
        
        # now loop over allhorizontal nodes (times)
        for i in range(self.TotalNumberofNodes-1): # only until -1, since we calculate for the next (i+1) period
            # get all necessray parameters (from period before)
            minj = self.TrinomialTreeparameters['minj'][i]
            dt = self.TrinomialTreeparameters['dt'][i]
            dRNext = self.TrinomialTreeparameters['dR'][i+1]
            dR = self.TrinomialTreeparameters['dR'][i]
            Alpha = self.TrinomialTreeparameters['Alpha'][i]
            centralNode = -self.TrinomialTreeparameters['minj'][i]  # central node (vertical) starting to count by the lowest node
            numberOfVerticalNodes = self.TrinomialTreeparameters['numberOfJNodes'][i]
            
            variance = self.TrinomialTreeparameters['volatility'][i+1]**2 * dt
            
            # next we loop over all vertical nodes and calculate the probs for going up and down
            for j in range(centralNode, numberOfVerticalNodes): # because of symmetry same for over and under the central nodes
                r = Alpha + (minj +j)*dR    # basically the first step, the spacing is always j*dR, but counting from 0!!
                
                rNext = r*(1. - self.a*dt)  # from mean of the process
                
                kValuation = int(rNext/dRNext + 0.5) # the next k 
                
                nextK = kValuation
                self.TrinomialTreeparameters['k'][i][j] = nextK
            
                self.TrinomialTreeparameters['pu'][i][j] = 1/6 + 0.5*(self.a**2*(minj +j)*dt**2 - self.a*(minj +j)*dt)
                self.TrinomialTreeparameters['pd'][i][j] = 1/6 + 0.5*(self.a**2*(minj +j)*dt**2 + self.a*(minj +j)*dt)               
                # consistency check
                if self.TrinomialTreeparameters['pu'][i][j]< 0 or self.TrinomialTreeparameters['pd'][i][j] < 0 \
                    or self.TrinomialTreeparameters['pu'][i][j] + self.TrinomialTreeparameters['pd'][i][j] > 1:
                    self.TrinomialTreeparameters['pu'][i][j] = 1
                    self.TrinomialTreeparameters['pd'][i][j] = 0
                    
                # for symmetry applay parameters for lower part
                self.TrinomialTreeparameters['k'][i][2* centralNode - j] = - int(kValuation)
                self.TrinomialTreeparameters['pu'][i][2* centralNode - j] = self.TrinomialTreeparameters['pd'][i][j]
                self.TrinomialTreeparameters['pd'][i][2* centralNode - j] = self.TrinomialTreeparameters['pu'][i][j]
                
            # if we are before the last step we have to initlaize space for the next step
            if i<(self.TotalNumberofNodes - 2):
                numberOfNewNodes = int(2*(kValuation+1) +1)
                self.TrinomialTreeparameters['numberOfJNodes'][i+1] = numberOfNewNodes
                self.TrinomialTreeparameters['minj'][i+1] = -(kValuation +1)
                self.TrinomialTreeparameters['Alpha'][i+1] = 0
                
                # ensure that we have at least one node
                if self.TrinomialTreeparameters['numberOfJNodes'][i+1] < 1:
                    self.TrinomialTreeparameters['numberOfJNodes'][i+1] = 1
                    
                # stoarge for next step
                self.TrinomialTreeparameters['k'][i+1] = np.zeros((numberOfNewNodes)).astype(int)
                self.TrinomialTreeparameters['pu'][i+1] = np.zeros((numberOfNewNodes))
                self.TrinomialTreeparameters['pd'][i+1] = np.zeros((numberOfNewNodes))
                self.TrinomialTreeparameters['ArrowDebrauPrices'][i+1] = np.zeros((numberOfNewNodes))
                self.TrinomialTreeparameters['OptionPrice'][i+1] = np.zeros((numberOfNewNodes))
                self.TrinomialTreeparameters['Yield'][i+1] = np.zeros((numberOfNewNodes), dtype=object)
                
                # finaly we also change the counting from k, so that it is in range[1,2*minj_{i+1}-1]
                for j in range(numberOfVerticalNodes):
                    self.TrinomialTreeparameters['k'][i][j] = self.TrinomialTreeparameters['k'][i][j] - self.TrinomialTreeparameters['minj'][i+1]
                
        
    def TreeAdjustments(self):
        # initialize the arrow debrau price at 1
        self.TrinomialTreeparameters['ArrowDebrauPrices'][0][0] = 1
            
        # calculate alphas according to hull and white
        for i in range(self.TotalNumberofNodes-1):
            shortRate = np.zeros((self.TrinomialTreeparameters['numberOfJNodes'][i]))
            nextT = self.TrinomialTreeparameters['t'][i+1]
            dt1 = self.TrinomialTreeparameters['dt'][i]

            p = np.exp(-nextT * getZero(self.Zerocurve,nextT))
                
            summe = 0
            for j in range(self.TrinomialTreeparameters['numberOfJNodes'][i]):
                summe += self.TrinomialTreeparameters['ArrowDebrauPrices'][i][j]\
                    * np.exp(-1*(self.TrinomialTreeparameters['minj'][i] + j) * self.TrinomialTreeparameters['dR'][i]*dt1)
            alpha = (np.log(summe) - np.log(p))/dt1
                
            self.TrinomialTreeparameters['Alpha'][i] = alpha
                
            # determine arrow debrau prices for the next time step
            if i< self.TotalNumberofNodes-2:
                for j in range(self.TrinomialTreeparameters['numberOfJNodes'][i]):
                    r = self.TrinomialTreeparameters['Alpha'][i] + (self.TrinomialTreeparameters['minj'][i] + j)* self.TrinomialTreeparameters['dR'][i]
                        
                    shortRate[j] = r
                    discountFactor = self.TrinomialTreeparameters['ArrowDebrauPrices'][i][j]*np.exp(-r*dt1)

                    currentK = self.TrinomialTreeparameters['k'][i][j] 
                    pu = self.TrinomialTreeparameters['pu'][i][j]
                    pd = self.TrinomialTreeparameters['pd'][i][j]
                    pm = 1- pu-pd
                        
                    # calculate the arrow debrau prices 
                    self.TrinomialTreeparameters['ArrowDebrauPrices'][i+1][currentK +1] += pu* discountFactor
                    self.TrinomialTreeparameters['ArrowDebrauPrices'][i+1][currentK ] += pm* discountFactor
                    self.TrinomialTreeparameters['ArrowDebrauPrices'][i+1][currentK -1] += pd* discountFactor
        
                # now safe the short rate
                self.TrinomialTreeparameters['ArrowDebrauPrices'][i] = shortRate
        
    
    def calculateNumericrates(self):
        datesOfRate = self.RateDates[:,0][::-1]

        for i in range(self.TotalNumberofNodes-2, -1, -1):
            t = self.TrinomialTreeparameters['t'][i]
            
            # check the number of rates to be calculated
            if t<= datesOfRate[-1]:
                numberOfRates = datesOfRate.shape[0]
            else:
                numberOfRates = np.argwhere(datesOfRate<t)[0][0]
            
            self.TrinomialTreeparameters['numberOfRates'][i] = int(numberOfRates)
            numberOfNodes = self.TrinomialTreeparameters['numberOfJNodes'][i]
            
            for j in range(numberOfNodes):
                self.TrinomialTreeparameters['Yield'][i][j] = np.zeros((numberOfRates, 2))
                self.TrinomialTreeparameters['Yield'][i][j][:,0] = datesOfRate[0:numberOfRates]
                for k in range(numberOfRates):
                    if i == self.TotalNumberofNodes or abs(t-datesOfRate[k])<=(1/365):
                        self.TrinomialTreeparameters['Yield'][i][j][k,1] = 1
                    else:
                        self.TrinomialTreeparameters['Yield'][i][j][k,1] = self.BondValue(i,j,k)
                    
        for i in range(self.TotalNumberofNodes-2, -1, -1):
            t = self.TrinomialTreeparameters['t'][i] 
            numberOfNodes = self.TrinomialTreeparameters['numberOfJNodes'][i]
            
            for j in range(numberOfNodes):
                for k in range(self.TrinomialTreeparameters['numberOfRates'][i]):
                    TimeDelta = self.TrinomialTreeparameters['Yield'][i][j][k,0] - t
                    
                    if TimeDelta >0:
                        self.TrinomialTreeparameters['Yield'][i][j][k,1] = -np.log(self.TrinomialTreeparameters['Yield'][i][j][k,1])/TimeDelta
                    else:
                        self.TrinomialTreeparameters['Yield'][i][j][k,1] = self.TrinomialTreeparameters['ArrowDebrauPrices'][i][j]
                    
                # last reverse the array
                self.TrinomialTreeparameters['Yield'][i][j][:,0] = self.TrinomialTreeparameters['Yield'][i][j][::-1, 0]
                self.TrinomialTreeparameters['Yield'][i][j][:,1] = self.TrinomialTreeparameters['Yield'][i][j][::-1, 1]
        
    def BondValue(self, horizontalNode, verticalNode, Rate):
        i = horizontalNode
        j = verticalNode
        k = self.TrinomialTreeparameters['k'][i][j]
        pu = self.TrinomialTreeparameters['pu'][i][j]
        pd = self.TrinomialTreeparameters['pd'][i][j]
        pm = 1- pd - pu
        
        alpha = self.TrinomialTreeparameters['Alpha'][i] 
        dt1 = self.TrinomialTreeparameters['dt'][i]
        dR = self.TrinomialTreeparameters['dR'][i] 
        verticalPosition = j + self.TrinomialTreeparameters['minj'][i] 
        discountFactor = np.exp(-(alpha+ verticalPosition*dR)*dt1)
        
        result = discountFactor*(pu*self.TrinomialTreeparameters['Yield'][i+1][k+1][Rate,1] +\
           pm*self.TrinomialTreeparameters['Yield'][i+1][k][Rate,1] + pu*self.TrinomialTreeparameters['Yield'][i+1][k-1][Rate,1])
        
        return result
    
    def getStep(self, date):
        if date == 0:
            return 0
        index = np.argwhere(self.NodeTimes>=date)[0][0]
        
        return index
    
    def BondOption(self, Cashflow, ExDates):
        numberOfSteps = self.getStep(ExDates[-1,0])
        
        for i in range(numberOfSteps, -1, -1):
            time = self.TrinomialTreeparameters['t'][i]
            
            # check if ex date
            ExPrice = 0
            isExDate = False
            for j in range(ExDates.shape[0]):
                if time == ExDates[j,0]:
                    ExPrice = ExDates[j,1]
                    isExDate = True
                    break
            # next trhough all j nodes
            for j in range(self.TrinomialTreeparameters['numberOfJNodes'][i]):
                YieldCurve = self.TrinomialTreeparameters['Yield'][i][j]
                InstrinsicValue = 0
                BondValue = 0
                
                if isExDate:
                    BondValue = AnalyticBondValue(time, Cashflow, YieldCurve)
                    
                    InstrinsicValue = BondValue - ExPrice
                    
                    if InstrinsicValue<0:   # beacause call option
                        InstrinsicValue = 0
                
                OptionValue = 0
                if i <numberOfSteps:
                    OptionValue = self.dicountedOptionValue(i,j)
                if InstrinsicValue>OptionValue:
                    OptionValue = InstrinsicValue
                    
                self.TrinomialTreeparameters['OptionPrice'][i][j] = OptionValue
                
        return self.TrinomialTreeparameters['OptionPrice'][0][0]
    
    def dicountedOptionValue(self, i, j):
        k = self.TrinomialTreeparameters['k'][i][j]
        pu = self.TrinomialTreeparameters['pu'][i][j]
        pd = self.TrinomialTreeparameters['pd'][i][j]
        pm = 1- pd - pu
        
        alpha = self.TrinomialTreeparameters['Alpha'][i] 
        dt1 = self.TrinomialTreeparameters['dt'][i]
        dR = self.TrinomialTreeparameters['dR'][i] 
        verticalPosition = j + self.TrinomialTreeparameters['minj'][i] 
        discountFactor = np.exp(-(alpha+ verticalPosition*dR)*dt1)
        
        result = discountFactor*(pu*self.TrinomialTreeparameters['OptionPrice'][i+1][k+1] +\
        pm*self.TrinomialTreeparameters['OptionPrice'][i+1][k] + pu*self.TrinomialTreeparameters['OptionPrice'][i+1][k-1])
        
        return result
           

def getZero(Zerocurve, time):
    if Zerocurve[-1,0]<= time:
        return Zerocurve[-1,1]
    elif Zerocurve[0,0]>= time:
        return Zerocurve[0,1]
    else:
        upperBoundIndex = np.argwhere(Zerocurve[:,0]<=time)[0][0]
        lowerBoundIndex = np.argwhere(Zerocurve[:,0]>=time)[0][0]
        upperBound = Zerocurve[upperBoundIndex,1]
        lowerBound = Zerocurve[lowerBoundIndex,1]
        lowerBoundTime = Zerocurve[lowerBoundIndex,0]
        
        return lowerBound+ (time-lowerBoundTime)*(upperBound-lowerBound)


def AnalyticBondValue(time, cashflow, YieldCurve):  # fixed paments discounted
    sum = 0
    for i, cash in enumerate(cashflow[:,1]):
        period_end = cashflow[i, 0]
        
        if period_end>time:
            sum += cash*np.exp(-getZero(YieldCurve, period_end)*(period_end-time))
            
    return sum
    





def treeConstruction(Zerocurve, lastDate, volatility, StepsPerYear = 72, a = 0):
    Tree = TrinomialTree(StepsPerYear, lastDate, Zerocurve)
    
    Tree.BuildTree(a, volatility)
    
    return Tree

def calculateBondPrice(Tree, cashflow, ExDates):
    # put exdates on nodes
    onlyDates = copy(ExDates[:,0])
    if ExDates !=[]:
        for index, date in enumerate(onlyDates):
            if date not in Tree.NodeTimes:
                difference = lambda nodes : abs(nodes - date)
                exTime = min(Tree.NodeTimes, key = difference)
                ExDates[index, 0] = exTime
                
    BondPrice = Tree.BondOption(cashflow, ExDates)
    
    return BondPrice
        
        