#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 12:32:03 2020

@author: ZZH
"""


import scipy.optimize as optimize
import matplotlib.pyplot as plt
import math
import numpy as np 
from numpy import linalg as LA

Bond1P=[99.85,99.86,99.86,99.86,99.86,99.86,99.86,99.86,99.86,99.86]
Bond2P=[99.26,99.28,99.28,99.27,99.28,99.28,99.28,99.27,99.28,99.3]
Bond3P=[98.89,98.93,98.95,98.94,98.92,98.92,98.88,98.9,98.9,98.93]
Bond4P=[99.27,99.31,99.36,99.34,99.32,99.27,99.25,99.24,99.25,99.28]
Bond5P=[97.57,97.63,97.66,97.64,97.65,97.6,97.61,97.57,97.58,97.61]
Bond6P=[102.53,102.59,102.62,102.59,102.58,102.52,102.52,102.46,102.47,102.51]
Bond7P=[100.31,100.42,100.48,100.45,100.44,100.35,100.31,100.27,100.31,100.38]
Bond8P=[99.48,99.59,99.65,99.61,99.62,99.54,99.53,99.44,99.49,99.56]
Bond9P=[102.52,102.65,102.75,102.58,102.68,102.53,102.47,102.46,102.54,102.64]
Bond10P=[98.72,98.95,99.29,99.11,99.25,98.99,99.03,99.06,98.99,99.1]
Bond11P=[98.24,98.43,98.58,98.48,98.48,98.3,98.25,98.24,98.34,98.47]


#Formula to Calculate YTM of the 1st bond
def calculate_1st_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)  
    return optimize.newton(ytm, 0.001)

#To find the YTM of the 1st bond
def calculate_Bond1_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_1st_ytm(i,1/6,1.5)
        result.append(ytm)       
    return result
  
Bond1ytm=calculate_Bond1_ytms(Bond1P)

#Formula To calculate YTM of the 2nd bond  
def calculate_2nd_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 2nd bond
def calculate_Bond2_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_2nd_ytm(i,2/3,0.75)
        result.append(ytm)       
    return result    

Bond2ytm=calculate_Bond2_ytms(Bond2P)

#Formula To calculate YTM of the 3rd bond  
def calculate_3rd_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365) + \
        Coup/(1+r/365)**((Maturity-1)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 3rd bond
def calculate_Bond3_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_3rd_ytm(i,7/6,0.75)
        result.append(ytm)       
    return result    

Bond3ytm=calculate_Bond3_ytms(Bond3P)

#Formula To calculate YTM of the 4th bond  
def calculate_4th_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365) + \
        Coup/(1+r/365)**((Maturity-1)*365) + Coup/(1+r/365)**((Maturity-3/2)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 4th bond
def calculate_Bond4_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_4th_ytm(i,11/6,1.25)
        result.append(ytm)       
    return result    

Bond4ytm=calculate_Bond4_ytms(Bond4P)


#Formula To calculate YTM of the 5th bond  
def calculate_5th_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365) + \
        Coup/(1+r/365)**((Maturity-1)*365) + Coup/(1+r/365)**((Maturity-3/2)*365) + \
        Coup/(1+r/365)**((Maturity-2)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 5th bond
def calculate_Bond5_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_5th_ytm(i,13/6,0.5)
        result.append(ytm)       
    return result    

Bond5ytm=calculate_Bond5_ytms(Bond5P)

#Formula To calculate YTM of the 6th bond  
def calculate_6th_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365) + \
        Coup/(1+r/365)**((Maturity-1)*365) + Coup/(1+r/365)**((Maturity-3/2)*365) + \
        Coup/(1+r/365)**((Maturity-2)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 6th bond
def calculate_Bond6_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_6th_ytm(i,29/12,2.75)
        result.append(ytm)       
    return result    

Bond6ytm=calculate_Bond6_ytms(Bond6P)

#Formula To calculate YTM of the 7th bond  
def calculate_7th_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365) + \
        Coup/(1+r/365)**((Maturity-1)*365) + Coup/(1+r/365)**((Maturity-3/2)*365) + \
        Coup/(1+r/365)**((Maturity-2)*365) + Coup/(1+r/365)**((Maturity-5/2)*365) + \
        Coup/(1+r/365)**((Maturity-6/2)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 7th bond
def calculate_Bond7_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_7th_ytm(i,19/6,1.75)
        result.append(ytm)       
    return result    

Bond7ytm=calculate_Bond7_ytms(Bond7P)


#Formula To calculate YTM of the 8th bond  
def calculate_8th_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365) + \
        Coup/(1+r/365)**((Maturity-1)*365) + Coup/(1+r/365)**((Maturity-3/2)*365) + \
        Coup/(1+r/365)**((Maturity-2)*365) + Coup/(1+r/365)**((Maturity-5/2)*365) + \
        Coup/(1+r/365)**((Maturity-6/2)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 8th bond
def calculate_Bond8_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_8th_ytm(i,41/12,1.5)
        result.append(ytm)       
    return result    

Bond8ytm=calculate_Bond8_ytms(Bond8P)

#Formula To calculate YTM of the 9th bond  
def calculate_9th_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365) + \
        Coup/(1+r/365)**((Maturity-1)*365) + Coup/(1+r/365)**((Maturity-3/2)*365) + \
        Coup/(1+r/365)**((Maturity-2)*365) + Coup/(1+r/365)**((Maturity-5/2)*365) + \
        Coup/(1+r/365)**((Maturity-6/2)*365)+ Coup/(1+r/365)**((Maturity-7/2)*365) + \
        Coup/(1+r/365)**((Maturity-8/2)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 9th bond
def calculate_Bond9_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_9th_ytm(i,50/12,2.25)
        result.append(ytm)       
    return result    

Bond9ytm=calculate_Bond9_ytms(Bond9P)

#Formula To calculate YTM of the 10th bond  
def calculate_10th_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365) + \
        Coup/(1+r/365)**((Maturity-1)*365) + Coup/(1+r/365)**((Maturity-3/2)*365) + \
        Coup/(1+r/365)**((Maturity-2)*365) + Coup/(1+r/365)**((Maturity-5/2)*365) + \
        Coup/(1+r/365)**((Maturity-6/2)*365)+ Coup/(1+r/365)**((Maturity-7/2)*365) + \
        Coup/(1+r/365)**((Maturity-8/2)*365)+ Coup/(1+r/365)**((Maturity-9/2)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 10th bond
def calculate_Bond10_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_9th_ytm(i,56/12,1.5)
        result.append(ytm)       
    return result    

Bond10ytm=calculate_Bond10_ytms(Bond10P)


#Formula To calculate YTM of the 11th bond  
def calculate_11th_ytm(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    ytm = lambda r: \
        - Price + 100/(1+r/365)**(Maturity*365) + \
        Coup/(1+r/365)**(Maturity*365)+ Coup/(1+r/365)**((Maturity-1/2)*365) + \
        Coup/(1+r/365)**((Maturity-1)*365) + Coup/(1+r/365)**((Maturity-3/2)*365) + \
        Coup/(1+r/365)**((Maturity-2)*365) + Coup/(1+r/365)**((Maturity-5/2)*365) + \
        Coup/(1+r/365)**((Maturity-6/2)*365)+ Coup/(1+r/365)**((Maturity-7/2)*365) + \
        Coup/(1+r/365)**((Maturity-8/2)*365)+ Coup/(1+r/365)**((Maturity-9/2)*365) + \
        Coup/(1+r/365)**((Maturity-10/2)*365)
    return optimize.newton(ytm, 0.001)
    
#To find the YTM of the 11th bond
def calculate_Bond11_ytms (BondPrice):
    result=[]
    for i in BondPrice:
        ytm=calculate_11th_ytm(i,62/12,1.25)
        result.append(ytm)       
    return result    

Bond11ytm=calculate_Bond11_ytms(Bond11P)



ytm1=[Bond1ytm[0],Bond2ytm[0],Bond3ytm[0],Bond4ytm[0],Bond5ytm[0],\
       Bond6ytm[0],Bond7ytm[0],Bond8ytm[0],Bond9ytm[0],Bond10ytm[0],Bond11ytm[0]]
ytm2=[Bond1ytm[1],Bond2ytm[1],Bond3ytm[1],Bond4ytm[1],Bond5ytm[1],\
       Bond6ytm[1],Bond7ytm[1],Bond8ytm[1],Bond9ytm[1],Bond10ytm[1],Bond11ytm[1]]
ytm3=[Bond1ytm[2],Bond2ytm[2],Bond3ytm[2],Bond4ytm[2],Bond5ytm[2],\
       Bond6ytm[2],Bond7ytm[2],Bond8ytm[2],Bond9ytm[2],Bond10ytm[2],Bond11ytm[2]]
ytm4=[Bond1ytm[3],Bond2ytm[3],Bond3ytm[3],Bond4ytm[3],Bond5ytm[3],\
       Bond6ytm[3],Bond7ytm[3],Bond8ytm[3],Bond9ytm[3],Bond10ytm[3],Bond11ytm[3]]
ytm5=[Bond1ytm[4],Bond2ytm[4],Bond3ytm[4],Bond4ytm[4],Bond5ytm[4],\
       Bond6ytm[4],Bond7ytm[4],Bond8ytm[4],Bond9ytm[4],Bond10ytm[4],Bond11ytm[4]]
ytm6=[Bond1ytm[5],Bond2ytm[5],Bond3ytm[5],Bond4ytm[5],Bond5ytm[5],\
       Bond6ytm[5],Bond7ytm[5],Bond8ytm[5],Bond9ytm[5],Bond10ytm[5],Bond11ytm[5]]
ytm7=[Bond1ytm[6],Bond2ytm[6],Bond3ytm[6],Bond4ytm[6],Bond5ytm[6],\
       Bond6ytm[6],Bond7ytm[6],Bond8ytm[6],Bond9ytm[6],Bond10ytm[6],Bond11ytm[6]]
ytm8=[Bond1ytm[7],Bond2ytm[7],Bond3ytm[7],Bond4ytm[7],Bond5ytm[7],\
       Bond6ytm[7],Bond7ytm[7],Bond8ytm[7],Bond9ytm[7],Bond10ytm[7],Bond11ytm[7]]
ytm9=[Bond1ytm[8],Bond2ytm[8],Bond3ytm[8],Bond4ytm[8],Bond5ytm[8],\
       Bond6ytm[8],Bond7ytm[8],Bond8ytm[8],Bond9ytm[8],Bond10ytm[8],Bond11ytm[8]]
ytm10=[Bond1ytm[9],Bond2ytm[9],Bond3ytm[9],Bond4ytm[9],Bond5ytm[9],\
       Bond6ytm[9],Bond7ytm[9],Bond8ytm[9],Bond9ytm[9],Bond10ytm[9],Bond11ytm[9]]

x = [1,2,3,4,5,6,7,8,9,10,11]
positions = (1,2,3,4,5,6,7,8,9,10,11)
labels = ("3/1/2020", "", "3/1/2021","", "3/1/2022", "","3/1/2023", "", "3/1/2024", "","3/1/2025")
plt.xticks(positions, labels)
plt.plot(x,ytm1,label='Jan2')
plt.plot(x,ytm2,label='Jan3')
plt.plot(x,ytm3,label='Jan6')
plt.plot(x,ytm4,label='Jan7')
plt.plot(x,ytm5,label='Jan8')
plt.plot(x,ytm6,label='Jan9')
plt.plot(x,ytm7,label='Jan10')
plt.plot(x,ytm8,label='Jan13')
plt.plot(x,ytm9,label='Jan14')
plt.plot(x,ytm10,label='Jan15')


plt.xlabel('Year')
plt.ylabel('Yield')
plt.title("Yield Curve")

plt.legend()
plt.show()



#Formula to Calculate SpotRate of the 1st bond
def calculate_1st_spot(Price, Maturity, Coupon):
    Price=float(Price)
    Maturity=float(Maturity)
    Coupon=float(Coupon)
    Coup = 0.5*Coupon
    spot = lambda s: \
        - Price + 100*(math.exp(-s*Maturity)) + \
        Coup*(math.exp(-s*Maturity))
    return optimize.newton(spot, 0.001)

#To find the SpotRate of the 1st bond
def calculate_Bond1_spot (BondPrice):
    result=[]
    for i in BondPrice:
        spot=calculate_1st_spot(i,1/6,1.5)
        result.append(spot)       
    return result
  
Bond1Spot=calculate_Bond1_spot(Bond1P)

    
#To find the SpotRate of the 2nd bond
def calculate_Bond2_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=2/3
        Coup=0.75/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond1Spot[i]*(Maturity-0.5)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond2Spot=calculate_Bond2_spot(Bond2P)

#To find the SpotRate of the 3nd bond
def calculate_Bond3_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=7/6
        Coup=0.75/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond2Spot[i]*(Maturity-0.5))) + \
            Coup*(math.exp(-Bond1Spot[i]*(Maturity-1)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond3Spot=calculate_Bond3_spot(Bond3P)

#To find the SpotRate of the 4th bond
def calculate_Bond4_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=11/6
        Coup=1.25/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond3Spot[i]*(Maturity-0.5))) + \
            Coup*(math.exp(-Bond2Spot[i]*(Maturity-1))) + Coup*(math.exp(-Bond1Spot[i]*(Maturity-1.5)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond4Spot=calculate_Bond4_spot(Bond4P)

#To find the SpotRate of the 5th bond
def calculate_Bond5_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=13/6
        Coup=0.5/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond4Spot[i]*(Maturity-0.5))) + \
            Coup*(math.exp(-Bond3Spot[i]*(Maturity-1))) + Coup*(math.exp(-Bond2Spot[i]*(Maturity-1.5))) + \
            Coup*(math.exp(-Bond1Spot[i]*(Maturity-2)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond5Spot=calculate_Bond5_spot(Bond5P)

#To find the SpotRate of the 6th bond
def calculate_Bond6_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=29/12
        Coup=2.75/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond4Spot[i]*(Maturity-0.5))) + \
            Coup*(math.exp(-Bond3Spot[i]*(Maturity-1))) + Coup*(math.exp(-Bond2Spot[i]*(Maturity-1.5))) + \
            Coup*(math.exp(-Bond1Spot[i]*(Maturity-2)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond6Spot=calculate_Bond6_spot(Bond6P)

#To find the SpotRate of the 7th bond
def calculate_Bond7_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=19/6
        Coup=1.75/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond6Spot[i]*(Maturity-0.5))) + \
            Coup*(math.exp(-Bond5Spot[i]*(Maturity-1))) + Coup*(math.exp(-Bond4Spot[i]*(Maturity-1.5))) + \
            Coup*(math.exp(-Bond3Spot[i]*(Maturity-2))) + Coup*(math.exp(-Bond2Spot[i]*(Maturity-2.5))) + \
            Coup*(math.exp(-Bond1Spot[i]*(Maturity-3)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond7Spot=calculate_Bond7_spot(Bond7P)

#To find the SpotRate of the 8th bond
def calculate_Bond8_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=41/12
        Coup=1.5/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond6Spot[i]*(Maturity-0.5))) + \
            Coup*(math.exp(-Bond5Spot[i]*(Maturity-1))) + Coup*(math.exp(-Bond4Spot[i]*(Maturity-1.5))) + \
            Coup*(math.exp(-Bond3Spot[i]*(Maturity-2))) + Coup*(math.exp(-Bond2Spot[i]*(Maturity-2.5))) + \
            Coup*(math.exp(-Bond1Spot[i]*(Maturity-3)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond8Spot=calculate_Bond8_spot(Bond8P)

#To find the SpotRate of the 9th bond
def calculate_Bond9_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=50/12
        Coup=2.25/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond8Spot[i]*(Maturity-0.5))) + \
            Coup*(math.exp(-Bond7Spot[i]*(Maturity-1))) + Coup*(math.exp(-Bond6Spot[i]*(Maturity-1.5))) + \
            Coup*(math.exp(-Bond5Spot[i]*(Maturity-2))) + Coup*(math.exp(-Bond4Spot[i]*(Maturity-2.5))) + \
            Coup*(math.exp(-Bond3Spot[i]*(Maturity-3))) + Coup*(math.exp(-Bond2Spot[i]*(Maturity-3.5))) + \
            Coup*(math.exp(-Bond1Spot[i]*(Maturity-4)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond9Spot=calculate_Bond9_spot(Bond9P)

#To find the SpotRate of the 10th bond
def calculate_Bond10_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=56/12
        Coup=1.5/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond8Spot[i]*(Maturity-0.5))) + \
            Coup*(math.exp(-Bond8Spot[i]*(Maturity-1))) + Coup*(math.exp(-Bond7Spot[i]*(Maturity-1.5))) + \
            Coup*(math.exp(-Bond6Spot[i]*(Maturity-2))) + Coup*(math.exp(-Bond5Spot[i]*(Maturity-2.5))) + \
            Coup*(math.exp(-Bond4Spot[i]*(Maturity-3))) + Coup*(math.exp(-Bond3Spot[i]*(Maturity-3.5))) + \
            Coup*(math.exp(-Bond2Spot[i]*(Maturity-4))) + Coup*(math.exp(-Bond1Spot[i]*(Maturity-4.5)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond10Spot=calculate_Bond10_spot(Bond10P)

#To find the SpotRate of the 11th bond
def calculate_Bond11_spot (BondPrice):
    result=[]
    for i in range(0,10):
        Price=BondPrice[i]
        Maturity=62/12
        Coup=1.25/2
        spot = lambda s: \
            - Price + 100*(math.exp(-s*Maturity)) + \
            Coup*(math.exp(-s*Maturity))+ Coup*(math.exp(-Bond10Spot[i]*(Maturity-0.5))) + \
            Coup*(math.exp(-Bond8Spot[i]*(Maturity-1))) + Coup*(math.exp(-Bond8Spot[i]*(Maturity-1.5))) + \
            Coup*(math.exp(-Bond7Spot[i]*(Maturity-2))) + Coup*(math.exp(-Bond6Spot[i]*(Maturity-2.5))) + \
            Coup*(math.exp(-Bond5Spot[i]*(Maturity-3))) + Coup*(math.exp(-Bond4Spot[i]*(Maturity-3.5))) + \
            Coup*(math.exp(-Bond3Spot[i]*(Maturity-4))) + Coup*(math.exp(-Bond2Spot[i]*(Maturity-4.5))) + \
            Coup*(math.exp(-Bond1Spot[i]*(Maturity-5)))
        result.append(optimize.newton(spot, 0.001))       
    return result    

Bond11Spot=calculate_Bond11_spot(Bond11P)



Spot1=[Bond1Spot[0],Bond2Spot[0],Bond3Spot[0],Bond4Spot[0],Bond5Spot[0],\
       Bond6Spot[0],Bond7Spot[0],Bond8Spot[0],Bond9Spot[0],Bond10Spot[0],Bond11Spot[0]]
Spot2=[Bond1Spot[1],Bond2Spot[1],Bond3Spot[1],Bond4Spot[1],Bond5Spot[1],\
       Bond6Spot[1],Bond7Spot[1],Bond8Spot[1],Bond9Spot[1],Bond10Spot[1],Bond11Spot[1]]
Spot3=[Bond1Spot[2],Bond2Spot[2],Bond3Spot[2],Bond4Spot[2],Bond5Spot[2],\
       Bond6Spot[2],Bond7Spot[2],Bond8Spot[2],Bond9Spot[2],Bond10Spot[2],Bond11Spot[2]]
Spot4=[Bond1Spot[3],Bond2Spot[3],Bond3Spot[3],Bond4Spot[3],Bond5Spot[3],\
       Bond6Spot[3],Bond7Spot[3],Bond8Spot[3],Bond9Spot[3],Bond10Spot[3],Bond11Spot[3]]
Spot5=[Bond1Spot[4],Bond2Spot[4],Bond3Spot[4],Bond4Spot[4],Bond5Spot[4],\
       Bond6Spot[4],Bond7Spot[4],Bond8Spot[4],Bond9Spot[4],Bond10Spot[4],Bond11Spot[4]]
Spot6=[Bond1Spot[5],Bond2Spot[5],Bond3Spot[5],Bond4Spot[5],Bond5Spot[5],\
       Bond6Spot[5],Bond7Spot[5],Bond8Spot[5],Bond9Spot[5],Bond10Spot[5],Bond11Spot[5]]
Spot7=[Bond1Spot[6],Bond2Spot[6],Bond3Spot[6],Bond4Spot[6],Bond5Spot[6],\
       Bond6Spot[6],Bond7Spot[6],Bond8Spot[6],Bond9Spot[6],Bond10Spot[6],Bond11Spot[6]]
Spot8=[Bond1Spot[7],Bond2Spot[7],Bond3Spot[7],Bond4Spot[7],Bond5Spot[7],\
       Bond6Spot[7],Bond7Spot[7],Bond8Spot[7],Bond9Spot[7],Bond10Spot[7],Bond11Spot[7]]
Spot9=[Bond1Spot[8],Bond2Spot[8],Bond3Spot[8],Bond4Spot[8],Bond5Spot[8],\
       Bond6Spot[8],Bond7Spot[8],Bond8Spot[8],Bond9Spot[8],Bond10Spot[8],Bond11Spot[8]]
Spot10=[Bond1Spot[9],Bond2Spot[9],Bond3Spot[9],Bond4Spot[9],Bond5Spot[9],\
       Bond6Spot[9],Bond7Spot[9],Bond8Spot[9],Bond9Spot[9],Bond10Spot[9],Bond11Spot[9]]

#Plot Spot Curve
x = [1,2,3,4,5,6,7,8,9,10,11]
positions = (1,2,3,4,5,6,7,8,9,10,11)
labels = ("3/1/2020", "", "3/1/2021","", "3/1/2022", "","3/1/2023", "", "3/1/2024", "","3/1/2025")
plt.xticks(positions, labels)
plt.plot(x,Spot1,label='Jan2')
plt.plot(x,Spot2,label='Jan3')
plt.plot(x,Spot3,label='Jan6')
plt.plot(x,Spot4,label='Jan7')
plt.plot(x,Spot5,label='Jan8')
plt.plot(x,Spot6,label='Jan9')
plt.plot(x,Spot7,label='Jan10')
plt.plot(x,Spot8,label='Jan13')
plt.plot(x,Spot9,label='Jan14')
plt.plot(x,Spot10,label='Jan15')


plt.xlabel('Date')
plt.ylabel('Spot Rate')
plt.title("Spot Curve")

plt.legend()
plt.show()

#To Calculate first 1-year forward rates 
f1rates=[]
for i in range(0,10):
    ra=Bond5Spot[i]
    rb=Bond3Spot[i]
    f1rate= (1+ra)**2 / (1+rb)  -1
    f1rates.append(f1rate)
    
#To Calculate second 1-year forward rates 
f2rates=[]
for i in range(0,10):
    ra=Bond7Spot[i]
    rb=Bond5Spot[i]
    f2rate= (1+ra)**3 / (1+rb)**2  -1
    f2rates.append(f2rate)
    
#To Calculate third 1-year forward rates 
f3rates=[]
for i in range(0,10):
    ra=Bond9Spot[i]
    rb=Bond7Spot[i]
    f3rate= (1+ra)**4 / (1+rb)**3  -1
    f3rates.append(f3rate)   
  
#To Calculate last 1-year forward rates 
f4rates=[]
for i in range(0,10):
    ra=Bond11Spot[i]
    rb=Bond9Spot[i]
    f4rate= (1+ra)**5 / (1+rb)**4  -1
    f4rates.append(f4rate)   

Forward1=[f1rates[0],f2rates[0],f3rates[0],f4rates[0]]
Forward2=[f1rates[1],f2rates[1],f3rates[1],f4rates[1]]
Forward3=[f1rates[2],f2rates[2],f3rates[2],f4rates[2]]
Forward4=[f1rates[3],f2rates[3],f3rates[3],f4rates[3]]
Forward5=[f1rates[4],f2rates[4],f3rates[4],f4rates[4]]
Forward6=[f1rates[5],f2rates[5],f3rates[5],f4rates[5]]
Forward7=[f1rates[6],f2rates[6],f3rates[6],f4rates[6]]
Forward8=[f1rates[7],f2rates[7],f3rates[7],f4rates[7]]
Forward9=[f1rates[8],f2rates[8],f3rates[8],f4rates[8]]
Forward10=[f1rates[9],f2rates[9],f3rates[9],f4rates[9]]





#Plot Forward Curve
x = [1,2,3,4]
positions = (1,2,3,4)
labels = ("3/1/2022","3/1/2023","3/1/2024","3/1/2025")
plt.xticks(positions, labels)
plt.plot(x,Forward1,label='Jan2')
plt.plot(x,Forward2,label='Jan3')
plt.plot(x,Forward3,label='Jan6')
plt.plot(x,Forward4,label='Jan7')
plt.plot(x,Forward5,label='Jan8')
plt.plot(x,Forward6,label='Jan9')
plt.plot(x,Forward7,label='Jan10')
plt.plot(x,Forward8,label='Jan13')
plt.plot(x,Forward9,label='Jan14')
plt.plot(x,Forward10,label='Jan15')


plt.xlabel('Date')
plt.ylabel('Forward Rate')
plt.title("Forward Curve")

plt.legend()
plt.show()



#Question 5
t1=[Bond3ytm[0],Bond5ytm[0],Bond7ytm[0],Bond9ytm[0],Bond11ytm[0]]
t2=[Bond3ytm[1],Bond5ytm[1],Bond7ytm[1],Bond9ytm[1],Bond11ytm[1]]
t3=[Bond3ytm[2],Bond5ytm[2],Bond7ytm[2],Bond9ytm[2],Bond11ytm[2]]
t4=[Bond3ytm[3],Bond5ytm[3],Bond7ytm[3],Bond9ytm[3],Bond11ytm[3]]
t5=[Bond3ytm[4],Bond5ytm[4],Bond7ytm[4],Bond9ytm[4],Bond11ytm[4]]
t6=[Bond3ytm[5],Bond5ytm[5],Bond7ytm[5],Bond9ytm[5],Bond11ytm[5]]
t7=[Bond3ytm[6],Bond5ytm[6],Bond7ytm[6],Bond9ytm[6],Bond11ytm[6]]
t8=[Bond3ytm[7],Bond5ytm[7],Bond7ytm[7],Bond9ytm[7],Bond11ytm[7]]
t9=[Bond3ytm[8],Bond5ytm[8],Bond7ytm[8],Bond9ytm[8],Bond11ytm[8]]
t10=[Bond3ytm[9],Bond5ytm[9],Bond7ytm[9],Bond9ytm[9],Bond11ytm[9]]

lr1=[math.exp(Bond3ytm[1]/Bond3ytm[0]),math.exp(Bond5ytm[1]/Bond5ytm[0]), \
     math.exp(Bond7ytm[1]/Bond7ytm[0]),math.exp(Bond9ytm[1]/Bond9ytm[0]), \
     math.exp(Bond11ytm[1]/Bond11ytm[0])]
lr2=[math.exp(Bond3ytm[2]/Bond3ytm[1]),math.exp(Bond5ytm[2]/Bond5ytm[1]), \
     math.exp(Bond7ytm[2]/Bond7ytm[1]),math.exp(Bond9ytm[2]/Bond9ytm[1]), \
     math.exp(Bond11ytm[2]/Bond11ytm[1])]
lr3=[math.exp(Bond3ytm[3]/Bond3ytm[2]),math.exp(Bond5ytm[3]/Bond5ytm[2]), \
     math.exp(Bond7ytm[3]/Bond7ytm[2]),math.exp(Bond9ytm[3]/Bond9ytm[2]), \
     math.exp(Bond11ytm[3]/Bond11ytm[2])]
lr4=[math.exp(Bond3ytm[4]/Bond3ytm[3]),math.exp(Bond5ytm[4]/Bond5ytm[3]), \
     math.exp(Bond7ytm[4]/Bond7ytm[3]),math.exp(Bond9ytm[4]/Bond9ytm[3]), \
     math.exp(Bond11ytm[4]/Bond11ytm[3])]
lr5=[math.exp(Bond3ytm[5]/Bond3ytm[4]),math.exp(Bond5ytm[5]/Bond5ytm[4]), \
     math.exp(Bond7ytm[5]/Bond7ytm[4]),math.exp(Bond9ytm[5]/Bond9ytm[4]), \
     math.exp(Bond11ytm[5]/Bond11ytm[4])]
lr6=[math.exp(Bond3ytm[6]/Bond3ytm[5]),math.exp(Bond5ytm[6]/Bond5ytm[5]), \
     math.exp(Bond7ytm[6]/Bond7ytm[5]),math.exp(Bond9ytm[6]/Bond9ytm[5]), \
     math.exp(Bond11ytm[6]/Bond11ytm[5])]
lr7=[math.exp(Bond3ytm[7]/Bond3ytm[6]),math.exp(Bond5ytm[7]/Bond5ytm[6]), \
     math.exp(Bond7ytm[7]/Bond7ytm[6]),math.exp(Bond9ytm[7]/Bond9ytm[6]), \
     math.exp(Bond11ytm[7]/Bond11ytm[6])]
lr8=[math.exp(Bond3ytm[8]/Bond3ytm[7]),math.exp(Bond5ytm[8]/Bond5ytm[7]), \
     math.exp(Bond7ytm[8]/Bond7ytm[7]),math.exp(Bond9ytm[8]/Bond9ytm[7]), \
     math.exp(Bond11ytm[8]/Bond11ytm[7])]
lr9=[math.exp(Bond3ytm[9]/Bond3ytm[8]),math.exp(Bond5ytm[9]/Bond5ytm[8]), \
     math.exp(Bond7ytm[9]/Bond7ytm[8]),math.exp(Bond9ytm[9]/Bond9ytm[8]), \
     math.exp(Bond11ytm[9]/Bond11ytm[8])]

yy1=[Bond3ytm[0],\
     Bond3ytm[1],\
     Bond3ytm[2],\
     Bond3ytm[3],\
     Bond3ytm[4],\
     Bond3ytm[5],\
     Bond3ytm[6],\
     Bond3ytm[7],\
     Bond3ytm[8],\
     Bond3ytm[9]]
    
yy2=[Bond5ytm[0],\
     Bond5ytm[1],\
     Bond5ytm[2],\
     Bond5ytm[3],\
     Bond5ytm[4],\
     Bond5ytm[5],\
     Bond5ytm[6],\
     Bond5ytm[7],\
     Bond5ytm[8],\
     Bond5ytm[9]] 

yy3=[Bond7ytm[0],\
     Bond7ytm[1],\
     Bond7ytm[2],\
     Bond7ytm[3],\
     Bond7ytm[4],\
     Bond7ytm[5],\
     Bond7ytm[6],\
     Bond7ytm[7],\
     Bond7ytm[8],\
     Bond7ytm[9]]

yy4=[Bond9ytm[0],\
     Bond9ytm[1],\
     Bond9ytm[2],\
     Bond9ytm[3],\
     Bond9ytm[4],\
     Bond9ytm[5],\
     Bond9ytm[6],\
     Bond9ytm[7],\
     Bond9ytm[8],\
     Bond9ytm[9]]

yy5=[Bond11ytm[0],\
     Bond11ytm[1],\
     Bond11ytm[2],\
     Bond11ytm[3],\
     Bond11ytm[4],\
     Bond11ytm[5],\
     Bond11ytm[6],\
     Bond11ytm[7],\
     Bond11ytm[8],\
     Bond11ytm[9]]

yy1=[math.log(Bond3ytm[1]/Bond3ytm[0]),\
     math.log(Bond3ytm[2]/Bond3ytm[1]),\
     math.log(Bond3ytm[3]/Bond3ytm[2]),\
     math.log(Bond3ytm[4]/Bond3ytm[3]),\
     math.log(Bond3ytm[5]/Bond3ytm[4]),\
     math.log(Bond3ytm[6]/Bond3ytm[5]),\
     math.log(Bond3ytm[7]/Bond3ytm[6]),\
     math.log(Bond3ytm[8]/Bond3ytm[7]),\
     math.log(Bond3ytm[9]/Bond3ytm[8])]

yy2=[math.log(Bond5ytm[1]/Bond5ytm[0]),\
     math.log(Bond5ytm[2]/Bond5ytm[1]),\
     math.log(Bond5ytm[3]/Bond5ytm[2]),\
     math.log(Bond5ytm[4]/Bond5ytm[3]),\
     math.log(Bond5ytm[5]/Bond5ytm[4]),\
     math.log(Bond5ytm[6]/Bond5ytm[5]),\
     math.log(Bond5ytm[7]/Bond5ytm[6]),\
     math.log(Bond5ytm[8]/Bond5ytm[7]),\
     math.log(Bond5ytm[9]/Bond5ytm[8])]

yy3=[math.log(Bond7ytm[1]/Bond7ytm[0]),\
     math.log(Bond7ytm[2]/Bond7ytm[1]),\
     math.log(Bond7ytm[3]/Bond7ytm[2]),\
     math.log(Bond7ytm[4]/Bond7ytm[3]),\
     math.log(Bond7ytm[5]/Bond7ytm[4]),\
     math.log(Bond7ytm[6]/Bond7ytm[5]),\
     math.log(Bond7ytm[7]/Bond7ytm[6]),\
     math.log(Bond7ytm[8]/Bond7ytm[7]),\
     math.log(Bond7ytm[9]/Bond7ytm[8])]

yy4=[math.log(Bond9ytm[1]/Bond9ytm[0]),\
     math.log(Bond9ytm[2]/Bond9ytm[1]),\
     math.log(Bond9ytm[3]/Bond9ytm[2]),\
     math.log(Bond9ytm[4]/Bond9ytm[3]),\
     math.log(Bond9ytm[5]/Bond9ytm[4]),\
     math.log(Bond9ytm[6]/Bond9ytm[5]),\
     math.log(Bond9ytm[7]/Bond9ytm[6]),\
     math.log(Bond9ytm[8]/Bond9ytm[7]),\
     math.log(Bond9ytm[9]/Bond9ytm[8])]

yy5=[math.log(Bond11ytm[1]/Bond11ytm[0]),\
     math.log(Bond11ytm[2]/Bond11ytm[1]),\
     math.log(Bond11ytm[3]/Bond11ytm[2]),\
     math.log(Bond11ytm[4]/Bond11ytm[3]),\
     math.log(Bond11ytm[5]/Bond11ytm[4]),\
     math.log(Bond11ytm[6]/Bond11ytm[5]),\
     math.log(Bond11ytm[7]/Bond11ytm[6]),\
     math.log(Bond11ytm[8]/Bond11ytm[7]),\
     math.log(Bond11ytm[9]/Bond11ytm[8])]
A=np.cov(yy2,yy3,yy4,yy5)
print(np.cov(yy2,yy3,yy4,yy5))

ff1=[math.log(f1rates[1]/f1rates[0]),\
     math.log(f1rates[2]/f1rates[1]),\
     math.log(f1rates[3]/f1rates[2]),\
     math.log(f1rates[4]/f1rates[3]),\
     math.log(f1rates[5]/f1rates[4]),\
     math.log(f1rates[6]/f1rates[5]),\
     math.log(f1rates[7]/f1rates[6]),\
     math.log(f1rates[8]/f1rates[7]),\
     math.log(f1rates[9]/f1rates[8])]

ff2=[math.log(f2rates[1]/f2rates[0]),\
     math.log(f2rates[2]/f2rates[1]),\
     math.log(f2rates[3]/f2rates[2]),\
     math.log(f2rates[4]/f2rates[3]),\
     math.log(f2rates[5]/f2rates[4]),\
     math.log(f2rates[6]/f2rates[5]),\
     math.log(f2rates[7]/f2rates[6]),\
     math.log(f2rates[8]/f2rates[7]),\
     math.log(f2rates[9]/f2rates[8])]

ff3=[math.log(f3rates[1]/f3rates[0]),\
     math.log(f3rates[2]/f3rates[1]),\
     math.log(f3rates[3]/f3rates[2]),\
     math.log(f3rates[4]/f3rates[3]),\
     math.log(f3rates[5]/f3rates[4]),\
     math.log(f3rates[6]/f3rates[5]),\
     math.log(f3rates[7]/f3rates[6]),\
     math.log(f3rates[8]/f3rates[7]),\
     math.log(f3rates[9]/f3rates[8])]

ff4=[math.log(f4rates[1]/f4rates[0]),\
     math.log(f4rates[2]/f4rates[1]),\
     math.log(f4rates[3]/f4rates[2]),\
     math.log(f4rates[4]/f4rates[3]),\
     math.log(f4rates[5]/f4rates[4]),\
     math.log(f4rates[6]/f4rates[5]),\
     math.log(f4rates[7]/f4rates[6]),\
     math.log(f4rates[8]/f4rates[7]),\
     math.log(f4rates[9]/f4rates[8])]

B=np.cov(ff1,ff2,ff3,ff4)
print(np.cov(ff1,ff2,ff3,ff4))

va1, ve1 = LA.eig(A)
print(va1)
print(ve1)

va2, ve2 = LA.eig(B)
print(va2)
print(ve2)
