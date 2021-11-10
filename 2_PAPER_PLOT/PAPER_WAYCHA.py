# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:05:59 2021

@author: Carlos Quezada
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:18:40 2020

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




way="D:/APEX_1501_TORA/11_PAPER_APEX_IRRIGATION/Irrigation_APEX/WAYCHA_PAPER/0ut_PAPER_WAYCHA.SAD"
dataway = np.loadtxt(way, dtype=str, comments='#', delimiter=None, 
                   skiprows=10)

o="D:/APEX_1501_TORA/NEW_CALIBRATION/Validation_PUCHUNI/PUCHINI/PLOT_PUCHUNI/Obs_PUCHUNI.txt"
"VARIABLES "
Observed = np.loadtxt(o, dtype=str, comments='#', delimiter=None)


date1 = '1984-01-01'
date2 = '1999-12-31'
mydates = pd.date_range(date1, date2).tolist()
mydates=pd.to_datetime(pd.Series(mydates), format='%b_%d')

date11= '1998-10-19'
date22 = '1999-05-01'
mydates2 = pd.date_range(date11, date22).tolist()
mydates2=pd.to_datetime(pd.Series(mydates2), format='%b_%d')

growthperiod=np.array([ Observed[jj][0] for jj in range(0,191)])
DAP=np.array([ Observed[jj][1] for jj in range(0,191)])
DAP= DAP.astype(float)

timegrowth = mdates.datestr2num(growthperiod)
mdates.num2date(timegrowth[0])

 
"---------------------------------------WAYCHA-----------------------------"
"WAYCHA"
TOTWAYC = np.array([ dataway[jj][10] for jj in range(0,len(dataway))])
TOTWAYC = TOTWAYC.astype(float)
TOTWAYC= [element * 100 for element in TOTWAYC]
HUI_WAY= np.array([ dataway[jj][6] for jj in range(0,len(dataway))])
WAYCHA=np.array([ dataway[jj][10] for jj in range(5407,5598)])
WAYCHA = WAYCHA.astype(float)
WAYCHA= [element * 100 for element in WAYCHA]
WAYCHA =np.array(WAYCHA)
STL=np.array([ dataway[jj][11] for jj in (5407,5598)])
STL = STL.astype(float)
STL= [element * 100 for element in STL]
STL =np.array(STL)
LAI_WAY= np.array([ dataway[jj][7] for jj in range(5407,5598)])
LAI_WAY = LAI_WAY.astype(float)
LAI_WAY=np.array(LAI_WAY)
print(WAYCHA[182])


WAYCHACSV=np.array([ dataway[jj][10] for jj in range(5407,5598)])
WAYCHACSV= WAYCHACSV.astype(float)
WAYCHACSV= [element * 100 for element in WAYCHACSV]
WAYCHACSV=np.array(WAYCHACSV)



"OBSERVED Waycha"
Obsway1=np.array([ Observed[jj][2] for jj in range(0,191)])
Obsway1= Obsway1.astype(float)
Obsway1= [element * 100 for element in Obsway1]
Obsway1=np.array(Obsway1)

Obsway=np.array([Obsway1[84],Obsway1[112],Obsway1[141],Obsway1[182]])
Obsway= Obsway.astype(float)
Obsway =np.array(Obsway)


"PBIAS - NSE- R2 ---------WAYCHA"
WAYCHAstat=np.array([ dataway[jj][10] for jj in range(5407,5598)])
WAYCHAstat=np.array([WAYCHAstat[84],WAYCHAstat[112],WAYCHAstat[141],WAYCHAstat[182]])
WAYCHAstat = WAYCHAstat.astype(float)
WAYCHAstat= [element * 100 for element in WAYCHAstat]
WAYCHAstat =np.array(WAYCHAstat)
Obswaystat=Obsway


Obswaymean=np.mean(Obswaystat)
WAYmean=np.mean(WAYCHAstat)

Sumway=(Obswaystat-WAYCHAstat)**2
Sumway=sum(Sumway)

Obsrestway=(Obswaystat-Obswaymean)**2
Obsrestway=sum(Obsrestway)
NSE_WAYCHA=1-(Sumway/Obsrestway)
NSE_WAYCHA="{:.4f}".format(NSE_WAYCHA)   


PBIAS_WAYCHA=(Obswaymean-WAYmean)*100/Obswaymean
PBIAS_WAYCHA="{:.2f}".format(PBIAS_WAYCHA)       

numr2=(Obswaystat-Obswaymean)*(WAYCHAstat-WAYmean)
numr2=sum(numr2)
numr2=numr2**2
den1=(Obswaystat-Obswaymean)**2
den1=sum(den1)
den2=(WAYCHAstat-WAYmean)**2
den2=sum(den2)
denr=den1*den2
R2way=numr2/denr
R2way="{:.4f}".format(R2way)  

timeobse=np.array([timegrowth[84],timegrowth[112],timegrowth[141],timegrowth[182]])


Waycsv=[DAP,Obsway1,WAYCHACSV]
Waycsv=np.array(Waycsv)
Waycsv=Waycsv.T

df = pd.DataFrame(Waycsv)
df.to_csv('Validation_Waycha_to_plot_.csv',index=False)

"//////////////////////////////////////////////////////"
x=Obswaystat
y=WAYCHAstat

def rsq(y1,y2):
    yresid=y1-y2
    SSresid=np.sum(yresid**2)
    SStotal=len(y1)*np.var(y1)
    r2=1-SSresid/SStotal
    return r2

def plot(a):
    plt.plot(x,y,"ko",label="Waycha - S. t. andigenum")
 
    xp=np.linspace(0,1800,5)
    
    font_1 = {'fontname':'Palatino Linotype','fontsize':16}
    font_22 = font_manager.FontProperties(family='Palatino Linotype',
                                   
                                   style='italic', size=15)
    slope = str(np.round(a[0],2))
    intercept = str(np.round(a[1],2))
    eqn='LstSQ:y='+slope+'x'+intercept
    plt.plot(xp,a[0]*xp+a[1],'k-')
    plt.ylabel('Simulated [g/m2]',**font_1,color='black')
    plt.xlabel("Observed [g/m2]",**font_1,color='black')
    plt.annotate("PBIAS= "+ str(PBIAS_WAYCHA),(1300,500),**font_1,color='black',size=15)
    plt.annotate("NSE= "+ str(NSE_WAYCHA),(1300,600),**font_1,color='black',size=15)
    plt.annotate("R2= "+ str(R2way),(1300,400),**font_1,color='black',size=15)
    plt.xticks(xp,**font_1,color='k', size=15)
    plt.yticks(xp,**font_1,color='k',size=15)
    plt.grid()
    plt.legend(fontsize=15,prop=font_22,loc='upper left')
    plt.show()
    
    "********* Plot REGRESSION OBS vs Sim   ************"
slope,intercept,r,p_value,std_err=linregress(x,y)
a = [slope,intercept]
print("R2 linear regresion="+str(r**2))
plot(a)

"...........................................PLOT....................................."

"_________WAYCHA - BIOM 1998-1999___________"

boundsY = np.linspace(0,2400,7,dtype=int)
fig, ax = plt.subplots(figsize=(9,6))

font = {'fontname':'Palatino Linotype','fontsize':19,'style':'normal'}
fontitalic = {'fontname':'Linotype','fontsize':19,'style':'italic'}
font2 = font_manager.FontProperties(family='Palatino Linotype',
                                   weight='bold',
                                   style='normal', size=19)

plt.plot_date(timegrowth, WAYCHA, '-',markersize=8,color='k',markeredgecolor='k',linewidth=3,
              markerfacecolor='black')
plt.plot_date(timeobse, Obsway, 'o',markersize=8,color='k',markeredgecolor='k',linewidth=2,
              markerfacecolor='k')

plt.ylabel('Biomass accumulation [g/m2]',**font,color='k')
plt.xlabel("Month",**font,color='k')
plt.yticks(boundsY, boundsY,**font, color='k', size=17)
plt.annotate("Waycha - S.t. andigenum",(729680,1500),**fontitalic,color='k',size=25)
plt.annotate("PBIAS= "+ str(PBIAS_WAYCHA),(729708,550),**font,color='k',size=17)
plt.annotate("NSE= "+ str(NSE_WAYCHA),(729708,700),**font,color='k',size=17)
plt.annotate("R2= "+ str(R2way),(729708,400),**font,color='k',size=17)
plt.xticks(**font,color='k',size=17)

plt.legend(["Simulated","Observed"],fontsize=17,prop=font2,loc='upper right')
plt.show()


"""----------------------------------------WEATHER-----------------------------------------------------------------
----------------------------------------
-----------------------------
------------------------"""
weather="D:/APEX_1501_TORA/NEW_CALIBRATION/PLOT_NEW/TIME_WEATHER.txt"
weatherdate = np.loadtxt(weather, dtype=str, comments='#', delimiter=None)

weatherdate11=np.array([ weatherdate[jj][0] for jj in range(0,191)])
dateweather1= mdates.datestr2num(weatherdate11)
mdates.num2date(dateweather1[0])



dateweather2=np.array([ weatherdate[jj][1] for jj in range(0,191)])

dateweather3= np.array([ weatherdate[jj][2] for jj in range(0,191)])

RAIN=np.array([ dataway[jj][26] for jj in range(5407,5598)])
RAIN = RAIN.astype(float)
RAIN =np.array(RAIN)

TMAX=np.array([ dataway[jj][23] for jj in range(5407,5598)])
TMAX = TMAX.astype(float)
TMAX =np.array(TMAX)
T2=TMIN




TMIN=np.array([ dataway[jj][24] for jj in range(5407,5598)])
TMIN = TMIN.astype(float)
TMIN =np.array(TMIN)
T1=TMIN


RAINMEAN=np.sum(RAIN)
print(RAINMEAN)

TMAXMEAN=np.mean(TMAX)
print(TMAXMEAN)

TMINMEAN=np.mean(TMIN)
print(TMINMEAN)




boundsY = np.linspace(0,42,7,dtype=int)
boundsX = np.linspace(0,191,8,dtype=int)
fig, ax = plt.subplots(figsize=(9,6))

font = {'fontname':'Palatino Linotype','fontsize':19,'style':'normal'}
fontitalic = {'fontname':'Linotype','fontsize':19,'style':'italic'}
font2 = font_manager.FontProperties(family='Palatino Linotype',
                                   weight='bold',
                                   style='normal', size=19)


plt.plot_date(weatherdate11, RAIN, '-',markersize=1,color='k',markeredgecolor='k',linewidth=1,
              markerfacecolor='black')
"""
ax.axvline(weatherdate11[0], color="R",linewidth=2)
ax.text(0.17,26,"Sowing-october", va='center', ha="center", bbox=dict(facecolor="w",alpha=1),fontsize=15,
        transform=ax.get_yaxis_transform())

ax.axvline(weatherdate11[187], color="R",linewidth=2)
ax.text(0.8,26,"Harvesting-April", va='center', ha="center", bbox=dict(facecolor="w",alpha=1),fontsize=15,
        transform=ax.get_yaxis_transform())
"""
plt.ylabel('Precipitation [mm]',**font,color='k',size=19)
plt.xlabel("Days after sowing",**font,color='k')
plt.yticks(boundsY, boundsY,**font, color='k', size=19)
plt.xticks(boundsX, boundsX,**font, color='k', size=19)


plt.show()

boundsY = np.linspace(-10,30,9,dtype=int)
boundsX = np.linspace(0,187,8,dtype=int)
fig, ax = plt.subplots(figsize=(9,6))

font = {'fontname':'Palatino Linotype','fontsize':19,'style':'normal'}
fontitalic = {'fontname':'Linotype','fontsize':19,'style':'italic'}
font2 = font_manager.FontProperties(family='Palatino Linotype',
                                   weight='bold',
                                   style='normal', size=19)


plt.plot_date(weatherdate11, TMAX, '-',markersize=1,color='k',markeredgecolor='k',linewidth=1,
              markerfacecolor='black')
plt.plot_date(weatherdate11, TMIN, '-',markersize=1,color='b',markeredgecolor='b',linewidth=1,
              markerfacecolor='blue')
ax.axhline(TMIN[35], color="k",linewidth=2)

"""
ax.axvline(weatherdate11[0], color="R",linewidth=2)
ax.text(0.17,26,"Sowing-october", va='center', ha="center", bbox=dict(facecolor="w",alpha=1),fontsize=15,
        transform=ax.get_yaxis_transform())

ax.axvline(weatherdate11[187], color="R",linewidth=2)
ax.text(0.8,26,"Harvesting-April", va='center', ha="center", bbox=dict(facecolor="w",alpha=1),fontsize=15,
        transform=ax.get_yaxis_transform())
"""
plt.ylabel('Temperature [oC]',**font,color='k',size=19)
plt.xlabel("Days after sowing",**font,color='k')
plt.yticks(boundsY, boundsY,**font, color='k', size=19)
plt.xticks(boundsX, boundsX,**font, color='k', size=19)
plt.legend(["Tmax","Tmin"],fontsize=19,prop=font2,loc='upper right')

plt.show()







"""
"_________WAYCHA - BIOM 1993-1994___________"

boundsY = np.linspace(0,1200,7,dtype=int)
fig, ax = plt.subplots(figsize=(12,5))
font = {'fontname':'Calibri','fontsize':18}
plt.plot_date(timegrowth, WAYCHA, 'o-',markersize=6,color='b',markeredgecolor='skyblue',linewidth=2,
              markerfacecolor='darkblue')
plt.plot_date(timegrowth, Obsway, 'o-',markersize=4,color='r',markeredgecolor='r',linewidth=2,
              markerfacecolor='r')
ax.axvline(timegrowth[143], color="red",linewidth=2)
ax.text(0.8,190,"DAP 144", va='center', ha="center", bbox=dict(facecolor="w",alpha=1),fontsize=15,
        transform=ax.get_yaxis_transform())
plt.ylabel('WAYCHA biomass [g/m2]',**font,color='k')
plt.xlabel("Month",**font,color='darkblue')
plt.yticks(boundsY, boundsY,fontsize=15,color='k')
plt.xticks(fontsize=15,color='k')
plt.annotate("PBIAS= "+ str(PBIAS_WAYCHA),(727900,500),**font,color='darkblue')
plt.annotate("NSE= "+ str(NSE_WAYCHA),(727900,600),**font,color='darkblue')
plt.annotate("R2= "+ str(R2way),(727900,400),**font,color='darkblue')
plt.title("WAYCHA Biomass growth",fontsize=20,fontstyle="italic",color='k')
plt.legend(["Simulated","Observed"])
plt.show()

"_________WAYCHA - STL 1993-1994___________"

boundsY = np.linspace(0,1200,7,dtype=int)
fig, ax = plt.subplots(figsize=(12,5))
font = {'fontname':'Calibri','fontsize':18}
plt.plot_date(timegrowth, STL, 'o-',markersize=6,color='black',markeredgecolor='skyblue',linewidth=2,
              markerfacecolor='black')
plt.plot_date(timegrowth, Obsway, 'o-',markersize=4,color='r',markeredgecolor='r',linewidth=2,
              markerfacecolor='r')
ax.axvline(timegrowth[143], color="red",linewidth=2)
ax.text(0.8,190,"DAP 144", va='center', ha="center", bbox=dict(facecolor="w",alpha=1),fontsize=15,
        transform=ax.get_yaxis_transform())
plt.ylabel('STL  [g/m2]',**font,color='k')
plt.xlabel("Month",**font,color='darkblue')
plt.yticks(boundsY, boundsY,fontsize=15,color='k')
plt.xticks(fontsize=15,color='k')
plt.title("WAYCHA STANDING LIVE PLANT BIOMASS",fontsize=20,fontstyle="italic",color='k')
plt.legend(["Simulated","Observed"])
plt.show()

"_________WAYCHA - LAI 1993-1994___________"

boundsY = np.linspace(0,5,6,dtype=int)
fig, ax = plt.subplots(figsize=(12,5))
font = {'fontname':'Calibri','fontsize':10}
plt.plot(DAP, LAI_WAY, 'o-',markersize=6,color='pink',markeredgecolor='pink',linewidth=2,
              markerfacecolor='pink')
ax.axvline(DAP[143], color="red",linewidth=2)
ax.text(0.8,4,"DAP 144", va='center', ha="center", bbox=dict(facecolor="w",alpha=1),fontsize=15,
        transform=ax.get_yaxis_transform())
plt.ylabel('WAYCHA LAI [m2/m2]',**font,color='k')
plt.xlabel("Month",**font,color='darkblue')
plt.yticks(boundsY, boundsY,fontsize=15,color='k')
plt.xticks(fontsize=15,color='k')
plt.title("WAYCHA Leaf Area Index",fontsize=20,fontstyle="italic",color='k')
plt.show()


"-----------------16 YEARS SIMULATION----------"

W=WAYCHA[143]


boundsY1 = np.linspace(0,1200,8,dtype=int)
fig, ax = plt.subplots(figsize=(14,8))
plt.plot_date(mydates, TOTWAYC, '-',markersize=20,color="darkblue",markeredgecolor='skyblue',linewidth=2,
markerfacecolor='darkblue')
fig.autofmt_xdate()
ax.set_xlim([datetime.date(1979, 1, 1), datetime.date(1995, 12, 31)])
ax.axhline(W, color="red",linewidth=2)
ax.text(0.51, W+80,"Biom DAP 144 WAYCHA = "+"{:.2f}".format(W), va='center', ha="center", bbox=dict(facecolor="w",alpha=1),fontsize=15,
        transform=ax.get_yaxis_transform())
plt.ylabel('WAYCHA [g/m2]',fontsize=20)
plt.yticks(boundsY1, boundsY1,fontsize=14)
plt.xticks(fontsize=20)
plt.title("WAYCHA Biomass growth",fontsize=22,fontstyle="italic")
plt.show()

"""




