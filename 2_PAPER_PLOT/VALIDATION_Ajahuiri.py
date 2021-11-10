# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:11:01 2020

@author: Carlos Quezada
"""


import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.stats import linregress
import matplotlib.font_manager as font_manager

aja="D:/APEX_1501_TORA/11_PAPER_APEX_IRRIGATION/Irrigation_APEX/3_VALIDATION_PATACAMAYA/VALIDATION_AJAHUIRI/0ut_VALIDATION_AJAHUIRI.SAD"
dataaja = np.loadtxt(aja, dtype=str, comments='#', delimiter=None, 
                   skiprows=9)

o="D:/APEX_1501_TORA/NEW_CALIBRATION/PLOT_NEW/Obs_Patacamaya_2.txt"
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


"---------------------------------------lUKI-----------------------------"
"LUKI"
TOTAJA = np.array([ dataaja[jj][10] for jj in range(0,len(dataaja))])
TOTAJA = TOTAJA.astype(float)
TOTAJA= [element * 100 for element in TOTAJA]
LAI_AJA= np.array([ dataaja[jj][8] for jj in range(0,len(dataaja))])
HUI_AJA= np.array([ dataaja[jj][7] for jj in range(0,len(dataaja))])

AJA=np.array([ dataaja[jj][10] for jj in range(5405,5592)])
AJA = AJA.astype(float)
AJA= [element * 100 for element in AJA]
AJA=np.array(AJA)
print(AJA[182])

STL=np.array([ dataaja[jj][11] for jj in range(5405,5592)])
STL = STL.astype(float)
STL= [element * 100 for element in STL]
STL =np.array(STL)

AJACSV=np.array([ dataaja[jj][10] for jj in range(5405,5592)])
AJACSV=AJACSV.astype(float)
AJACSV=[element * 100 for element in AJACSV]
AJACSV=np.array(AJACSV)


"OBSERVED Luki"
Obsaja1=np.array([ Observed[jj][5] for jj in range(0,187)])
Obsaja1= Obsaja1.astype(float)
Obsaja1= [element * 100 for element in Obsaja1]
Obsaja1=np.array(Obsaja1)

Obsaja=np.array([Obsaja1[84],Obsaja1[112],Obsaja1[141],Obsaja1[182]])
Obsaja= Obsaja.astype(float)
Obsaja =np.array(Obsaja)

" PBIAS - NSE - R2 -----LUKI"
AJAstat=np.array([ dataaja[jj][10] for jj in range(5405,5592)])
AJAstat=np.array([AJAstat[84],AJAstat[112],AJAstat[141],AJAstat[182]])
AJAstat = AJAstat.astype(float)
AJAstat= [element * 100 for element in AJAstat]
AJAstat =np.array(AJAstat)
Obsajastat=Obsaja


Obsajamean=np.mean(Obsajastat)
AJAmean=np.mean(AJAstat)

Sumaja=(Obsajastat-AJAstat)**2
Sumaja=sum(Sumaja)

Obsrestaja=(Obsajastat-Obsajamean)**2
Obsrestaja=sum(Obsrestaja)
NSE_AJA=1-(Sumaja/Obsrestaja)
NSE_AJA="{:.4f}".format(NSE_AJA)   

PBIAS_AJA=(Obsajamean-AJAmean)*100/Obsajamean
PBIAS_AJA="{:.2f}".format(PBIAS_AJA)       

numr2=(Obsajastat-Obsajamean)*(AJAstat-AJAmean)
numr2=sum(numr2)
numr2=numr2**2
den1=(Obsajastat-Obsajamean)**2
den1=sum(den1)
den2=(AJAstat-AJAmean)**2
den2=sum(den2)
denr=den1*den2
R2aja=numr2/denr
R2aja="{:.4f}".format(R2aja)  


timeobse=np.array([timegrowth[84],timegrowth[112],timegrowth[141],timegrowth[182]])


AJAcsv=[DAP,Obsaja1,AJACSV]
AJAcsv=np.array(AJAcsv)
AJAcsv=AJAcsv.T




x=Obsajastat
y=AJAstat

def rsq(y1,y2):
    yresid=y1-y2
    SSresid=np.sum(yresid**2)
    SStotal=len(y1)*np.var(y1)
    r2=1-SSresid/SStotal
    return r2

def plot(a):
    plt.plot(x,y,"ko",label="Ajahuiri - S. ajanhuiri")
 
    xp=np.linspace(0,1800,5)
    font_1 = {'fontname':'Palatino Linotype','fontsize':16}
    font_22 = font_manager.FontProperties(family='Palatino Linotype',
                                   
                                   style='italic', size=15)
    slope = str(np.round(a[0],2))
    intercept = str(np.round(a[1],2))
    eqn='LstSQ:y='+slope+'x'+intercept
    plt.plot(xp,a[0]*xp+a[1],'k-',label=eqn)
    plt.ylabel('Simulated [g/m2]',**font_1,color='black')
    plt.xlabel("Observed [g/m2]",**font_1,color='black')
    plt.annotate("PBIAS= "+ str(PBIAS_AJA),(1300,500),**font_1,color='black',size=15)
    plt.annotate("NSE= "+ str(NSE_AJA),(1300,600),**font_1,color='black',size=15)
    plt.annotate("R2= "+ str(R2aja),(1300,400),**font_1,color='black',size=15)
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

"_________LUKI - BIOM 1998-1999___________"


boundsY = np.linspace(0,2400,7,dtype=int)

fig, ax = plt.subplots(figsize=(9,6))

font = {'fontname':'Palatino Linotype','fontsize':19,'style':'normal'}
fontitalic = {'fontname':'Linotype','fontsize':19,'style':'italic'}
font2 = font_manager.FontProperties(family='Palatino Linotype',
                                   weight='bold',
                                   style='normal', size=19)

plt.plot_date(timegrowth, AJA, '-',markersize=8,color='k',markeredgecolor='k',linewidth=3,
              markerfacecolor='k')
plt.plot_date(timeobse, Obsaja, 'o',markersize=8,color='k',markeredgecolor='k',linewidth=2,
              markerfacecolor='k')

plt.ylabel('Biomass accumulation [g/m2]',**font,color='k')
plt.xlabel("Month",**font,color='k')
plt.yticks(boundsY, boundsY,**font, color='k', size=19)
plt.annotate("Ajahuiri - S. ajanhuiri",(729680,2000),**fontitalic,color='k',size=23)
plt.annotate("PBIAS= "+ str(PBIAS_AJA),(729708,550),**font,color='k',size=19)
plt.annotate("NSE= "+ str(NSE_AJA),(729708,700),**font,color='k',size=19)
plt.annotate("R2= "+ str(R2aja),(729708,400),**font,color='k',size=19)
plt.xticks(**font,color='k',size=19)
plt.legend(["Simulated","Observed"],fontsize=19,prop=font2,loc='upper right')
plt.show()













