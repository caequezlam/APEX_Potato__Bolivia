# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:09:11 2020

@author: Carlos Quezada
"""

import numpy as np
import matplotlib.dates as mdates
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import matplotlib.font_manager as font_manager

alp="D:/APEX_1501_TORA/11_PAPER_APEX_IRRIGATION/Irrigation_APEX/3_VALIDATION_PATACAMAYA/VALIDATION_ALPHA/0ut_VALIDATION_ALPHA.SAD"
dataalph = np.loadtxt(alp, dtype=str, comments='#', delimiter=None, 
                   skiprows=9)

o="D:/APEX_1501_TORA/NEW_CALIBRATION/PLOT_NEW/Obs_Patacamaya.txt"

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

"---------------------------------------ALPHA-----------------------------"
"ALPHA"
TOTALPHA = np.array([ dataalph[jj][10] for jj in range(0,len(dataalph))])
TOTALPHA= TOTALPHA.astype(float)
TOTALPHA= [element * 100 for element in TOTALPHA]
LAI_ALPH= np.array([ dataalph[jj][8] for jj in range(0,len(dataalph))])
HUI_ALPH= np.array([ dataalph[jj][7] for jj in range(0,len(dataalph))])
ALPHA=np.array([ dataalph[jj][10] for jj in range(5405,5592)])
ALPHA = ALPHA.astype(float)
ALPHA= [element * 100 for element in ALPHA]
ALPHA =np.array(ALPHA)
print(ALPHA[182])


STL=np.array([ dataalph[jj][11] for jj in range(5405,5592)])
STL = STL.astype(float)
STL= [element * 100 for element in STL]
STL =np.array(STL)

ALPHACSV=np.array([ dataalph[jj][10] for jj in range(5405,5592)])
ALPHACSV=ALPHACSV.astype(float)
ALPHACSV=[element * 100 for element in ALPHACSV]
ALPHACSV=np.array(ALPHACSV)



"OBSERVED Alpha"
Obsalph1=np.array([ Observed[jj][4] for jj in range(0,187)])
Obsalph1= Obsalph1.astype(float)
Obsalph1= [element * 100 for element in Obsalph1]
Obsalph1=np.array(Obsalph1)



Obsalph=np.array([Obsalph1[84],Obsalph1[112],Obsalph1[141],Obsalph1[182]])
Obsalph= Obsalph.astype(float)
Obsalph =np.array(Obsalph)

" PBIAS - NSE - R2 -------Alpha"
ALPHAstat = np.array([ dataalph[jj][10] for jj in range(5405,5592)])
ALPHAstat=np.array([ALPHAstat[84],ALPHAstat[112],ALPHAstat[141],ALPHAstat[182]])
ALPHAstat = ALPHAstat.astype(float)
ALPHAstat = [element * 100 for element in ALPHAstat]
ALPHAstat = np.array(ALPHAstat)
Obsalphstat =Obsalph

Obsalphmean=np.mean(Obsalphstat)
ALPHmean=np.mean(ALPHAstat)

Sumalph=(Obsalphstat-ALPHAstat)**2
Sumalph=sum(Sumalph)

Obsrestalph=(Obsalphstat-Obsalphmean)**2
Obsrestalph=sum(Obsrestalph)
NSE_ALPHA=1-(Sumalph/Obsrestalph)
NSE_ALPHA="{:.4f}".format(NSE_ALPHA)   

PBIAS_ALPHA=(Obsalphmean-ALPHmean)*100/Obsalphmean
PBIAS_ALPHA="{:.2f}".format(PBIAS_ALPHA)  

numr2=(Obsalphstat-Obsalphmean)*(ALPHAstat-ALPHmean)
numr2=sum(numr2)
numr2=numr2**2
den1=(Obsalphstat-Obsalphmean)**2
den1=sum(den1)
den2=(ALPHAstat-ALPHmean)**2
den2=sum(den2)
denr=den1*den2
R2alph=numr2/denr
R2alph="{:.4f}".format(R2alph) 

timeobse=np.array([timegrowth[84],timegrowth[112],timegrowth[141],timegrowth[182]])


Alphcsv=[DAP,Obsalph1,ALPHACSV]
Alphcsv=np.array(Alphcsv)
Alphcsv=Alphcsv.T

df = pd.DataFrame(Alphcsv)
df.to_csv('Validation_Alpha_to_plot.csv',index=False)

x=Obsalphstat
y=ALPHAstat

def rsq(y1,y2):
    yresid=y1-y2
    SSresid=np.sum(yresid**2)
    SStotal=len(y1)*np.var(y1)
    r2=1-SSresid/SStotal
    return r2

def plot(a):
    plt.plot(x,y,"ko",label="Alpha - S. t. tuberosum")
 
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
    plt.annotate("PBIAS= "+ str(PBIAS_ALPHA),(1300,500),**font_1,color='black',size=15)
    plt.annotate("NSE= "+ str(NSE_ALPHA),(1300,600),**font_1,color='black',size=15)
    plt.annotate("R2= "+ str(R2alph),(1300,400),**font_1,color='black',size=15)
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




"________ALPHA - BIOM 1998-1999___________"


boundsY = np.linspace(0,2400,7,dtype=int)

fig, ax = plt.subplots(figsize=(9,6))

font = {'fontname':'Palatino Linotype','fontsize':19,'style':'normal'}
fontitalic = {'fontname':'Linotype','fontsize':19,'style':'italic'}
font2 = font_manager.FontProperties(family='Palatino Linotype',
                                   weight='bold',
                                   style='normal', size=19)
plt.plot_date(timegrowth, ALPHA, '-',markersize=8,color='k',markeredgecolor='k',linewidth=3,
              markerfacecolor='k')
plt.plot_date(timeobse, Obsalph, 'o',markersize=8,color='k',markeredgecolor='k',linewidth=2,
              markerfacecolor='k')

plt.ylabel('Biomass accumulation [g/m2]',**font,color='k')
plt.xlabel("Month",**font,color='k')
plt.yticks(boundsY, boundsY,**font, color='k', size=19)
plt.annotate("Alpha - S. t. tuberosum",(729680,2000),**fontitalic,color='k',size=23)
plt.annotate("PBIAS= "+ str(PBIAS_ALPHA),(729708,550),**font,color='k',size=19)
plt.annotate("NSE= "+ str(NSE_ALPHA),(729708,700),**font,color='k',size=19)
plt.annotate("R2= "+ str(R2alph),(729708,400),**font,color='k',size=19)
plt.xticks(**font,color='k',size=19)
plt.legend(["Simulated","Observed"],fontsize=19,prop=font2,loc='upper right')
plt.show()


"""
ax.axvline(timegrowth[89], color="red",linewidth=2)------LINEA 

ax.text(0.8,190,"DAP 144", va='center', ha="center", bbox=dict(facecolor="w",alpha=1),fontsize=15,
        transform=ax.get_yaxis_transform())
"""





