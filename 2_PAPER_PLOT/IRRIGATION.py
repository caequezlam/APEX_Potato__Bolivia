# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:38:30 2021

@author: Carlos Quezada
"""

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.stats import linregress
import statsmodels.api as sm
from gekko import GEKKO
import matplotlib.font_manager as font_manager


"-------------------------------------WAYCHA------------------------------"
way="D:/APEX_1501_TORA/11_PAPER_APEX_IRRIGATION/Irrigation_APEX/3_VALIDATION_PATACAMAYA/VALIDATION_WAYCHA/0ut_VALIDATION_WAYCHA.SAD"
dataway = np.loadtxt(way, dtype=str, comments='#', delimiter=None, 
                   skiprows=10)

o="D:/APEX_1501_TORA/NEW_CALIBRATION/PLOT_NEW/Obs_Patacamaya.txt"
"VARIABLES "
Observed = np.loadtxt(o, dtype=str, comments='#', delimiter=None)



date1 = '1984-01-01'
date2 = '1999-12-31'
mydates = pd.date_range(date1, date2).tolist()
mydates=pd.to_datetime(pd.Series(mydates), format='%b_%d')

date11= '1998-10-19'
date22 = '1999-04-30'
mydates2 = pd.date_range(date11, date22).tolist()
mydates2=pd.to_datetime(pd.Series(mydates2), format='%b_%d')

growthperiod=np.array([ Observed[jj][0] for jj in range(0,187)])
DAP=np.array([ Observed[jj][1] for jj in range(0,187)])
DAP= DAP.astype(float)

timegrowth = mdates.datestr2num(growthperiod)
mdates.num2date(timegrowth[0])

"WAYCHA"
TOTWAYC = np.array([ dataway[jj][10] for jj in range(0,len(dataway))])
TOTWAYC = TOTWAYC.astype(float)
TOTWAYC= [element * 100 for element in TOTWAYC]
HUI_WAY= np.array([ dataway[jj][6] for jj in range(0,len(dataway))])
WAYCHA=np.array([ dataway[jj][10] for jj in range(5405,5592)])
WAYCHA = WAYCHA.astype(float)
WAYCHA= [element * 100 for element in WAYCHA]
WAYCHA =np.array(WAYCHA)
STL=np.array([ dataway[jj][11] for jj in range(5405,5592)])
STL = STL.astype(float)
STL= [element * 100 for element in STL]
STL =np.array(STL)
LAI_WAY= np.array([ dataway[jj][7] for jj in range(5405,5592)])
LAI_WAY = LAI_WAY.astype(float)
LAI_WAY=np.array(LAI_WAY)
print(WAYCHA[182])




"-----------------------------IRRIGATION----------------------------"




MSA="D:/APEX_1501_TORA/11_PAPER_APEX_IRRIGATION/Irrigation_APEX/3_VALIDATION_PATACAMAYA/VALIDATION_AJAHUIRI/0ut_VALIDATION_AJAHUIRI.MSA"
Montlysub = np.loadtxt(MSA, dtype=str, comments='#', delimiter=None, 
                   skiprows=9,usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))


IRGA1998=np.array([ Montlysub[1273][jj] for jj in range(13,16)])
IRGA1998 = IRGA1998.astype(float)
IRGA1998 =np.array(IRGA1998)

IRGA1999=np.array([ Montlysub[1363][jj] for jj in range(4,8)])
IRGA1999 = IRGA1999.astype(float)
IRGA1999 =np.array(IRGA1999)

I=np.concatenate((IRGA1998,IRGA1999),axis=0)

Month=np.array(["Oct", "Nov", "Dic", "Jan", "Feb", "Mar", "Apr"])


I=np.vstack((Month,I))

IRDAILY2="D:/APEX_1501_TORA/11_PAPER_APEX_IRRIGATION/Irrigation_APEX/3_VALIDATION_PATACAMAYA/VALIDATION_AJAHUIRI/0ut_VALIDATION_AJAHUIRI.SAD"
D1= np.loadtxt(IRDAILY2, dtype=str, comments='#', delimiter=None, 
                   skiprows=5415, usecols=(2,3,4,5,40))

EVAPO="D:/APEX_1501_TORA/11_PAPER_APEX_IRRIGATION/Irrigation_APEX/3_VALIDATION_PATACAMAYA/VALIDATION_AJAHUIRI/0ut_VALIDATION_AJAHUIRI.SAD"
E= np.loadtxt(IRDAILY2, dtype=str, comments='#', delimiter=None, 
                   skiprows=5415, usecols=(2,3,4,5,41))

ETo=np.array([ E[jj][4] for jj in range(0,187)])
E11 = ETo.astype(float)
ETtotal=sum(E11)
print(ETtotal)

o="D:/APEX_1501_TORA/NEW_CALIBRATION/PLOT_NEW/Obs_Patacamaya.txt"
"VARIABLES "
ObservedIRR = np.loadtxt(o, dtype=str, comments='#', delimiter=None)

growthperiod=np.array([ ObservedIRR[jj][0] for jj in range(0,187)])
timegrowth = mdates.datestr2num(growthperiod)

IRGA=np.array([ D1[jj][4] for jj in range(0,187)])
I11 = IRGA.astype(float)


boundsY = np.linspace(0,60,9,dtype=int)
fig, ax = plt.subplots(figsize=(9,6))

font = {'fontname':'Palatino Linotype','fontsize':19,'style':'normal'}
fontitalic = {'fontname':'Linotype','fontsize':19,'style':'italic'}
font2 = font_manager.FontProperties(family='Palatino Linotype',
                                   weight='bold',
                                   style='normal', size=19)

ax.plot_date(timegrowth, I11, '-',markersize=8,color='k',markeredgecolor='k',linewidth=3,
              markerfacecolor='black')


plt.ylabel('Irrigation [mm]',**font,color='k',size=19)
plt.xlabel("Month",**font,color='k')
plt.yticks(boundsY, boundsY,**font, color='k', size=19)
plt.xticks(**font,color='k',size=19)

ax2=ax.twinx()

plt.plot_date(timegrowth, STL, '-',markersize=8,color='k',markeredgecolor='k',linewidth=3,
              markerfacecolor='black')

boundsY2 = np.linspace(0,1600,6,dtype=int)
plt.ylabel('Dry tuber Yield [g/m2]',**font,color='k',size=19)
plt.xlabel("Month",**font,color='k')
plt.yticks(boundsY2, boundsY2,**font, color='k', size=19)

plt.xticks(**font,color='k',size=19)

ax2.plot


plt.show()



