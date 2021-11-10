# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:27:16 2021

@author: Carlos Quezada
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:11:01 2020
+
@author: Carlos Quezada
"""


import numpy as np
import matplotlib.dates as mdates
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import matplotlib.font_manager as font_manager



luk="D:/APEX_1501_TORA/11_PAPER_APEX_IRRIGATION/Irrigation_APEX/AJAHUIRI_PAPER/0ut_PAPER_AJAHUIRI.SAD"
dataluk = np.loadtxt(luk, dtype=str, comments='#', delimiter=None, 
                   skiprows=9)

o="D:/APEX_1501_TORA/NEW_CALIBRATION/Validation_PUCHUNI/PUCHINI/PLOT_PUCHUNI/Obs_PUCHUNI.txt"
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

growthperiod=np.array([ Observed[jj][0] for jj in range(0,198)])
DAP=np.array([ Observed[jj][1] for jj in range(0,198)])
DAP= DAP.astype(float)

timegrowth = mdates.datestr2num(growthperiod)
mdates.num2date(timegrowth[0])


"---------------------------------------lUKI-----------------------------"
"LUKI"
TOTLUKI = np.array([ dataluk[jj][10] for jj in range(0,len(dataluk))])
TOTLUKI = TOTLUKI.astype(float)
TOTLUKI= [element * 100 for element in TOTLUKI]
LAI_LUK= np.array([ dataluk[jj][8] for jj in range(0,len(dataluk))])
HUI_LUK= np.array([ dataluk[jj][7] for jj in range(0,len(dataluk))])

LUKI=np.array([ dataluk[jj][10] for jj in range(5407,5605)])
LUKI = LUKI.astype(float)
LUKI= [element * 100 for element in LUKI]
LUKI =np.array(LUKI)
print(LUKI[182])

STL=np.array([ dataluk[jj][11] for jj in range(5407,5605)])
STL = STL.astype(float)
STL= [element * 100 for element in STL]
STL =np.array(STL)

LUKICSV=np.array([ dataluk[jj][10] for jj in range(5407,5605)])
LUKICSV=LUKICSV.astype(float)
LUKICSV=[element * 100 for element in LUKICSV]
LUKICSV=np.array(LUKICSV)


"OBSERVED Luki"
Obsluk1=np.array([ Observed[jj][5] for jj in range(0,198)])
Obsluk1= Obsluk1.astype(float)
Obsluk1= [element * 100 for element in Obsluk1]
Obsluk1=np.array(Obsluk1)

Obsluk=np.array([Obsluk1[84],Obsluk1[112],Obsluk1[141],Obsluk1[182]])
Obsluk= Obsluk.astype(float)
Obsluk =np.array(Obsluk)

" PBIAS - NSE - R2 -----LUKI"
LUKIstat=np.array([ dataluk[jj][10] for jj in range(5407,5605)])
LUKIstat=np.array([LUKIstat[84],LUKIstat[112],LUKIstat[141],LUKIstat[182]])
LUKIstat = LUKIstat.astype(float)
LUKIstat= [element * 100 for element in LUKIstat]
LUKIstat =np.array(LUKIstat)
Obslukstat=Obsluk


Obslukmean=np.mean(Obslukstat)
LUKmean=np.mean(LUKIstat)

Sumluk=(Obslukstat-LUKIstat)**2
Sumluk=sum(Sumluk)

Obsrestluk=(Obslukstat-Obslukmean)**2
Obsrestluk=sum(Obsrestluk)
NSE_LUKI=1-(Sumluk/Obsrestluk)
NSE_LUKI="{:.4f}".format(NSE_LUKI)   

PBIAS_LUKI=(Obslukmean-LUKmean)*100/Obslukmean
PBIAS_LUKI="{:.2f}".format(PBIAS_LUKI)       

numr2=(Obslukstat-Obslukmean)*(LUKIstat-LUKmean)
numr2=sum(numr2)
numr2=numr2**2
den1=(Obslukstat-Obslukmean)**2
den1=sum(den1)
den2=(LUKIstat-LUKmean)**2
den2=sum(den2)
denr=den1*den2
R2luk=numr2/denr
R2luk="{:.4f}".format(R2luk)  


timeobse=np.array([timegrowth[84],timegrowth[112],timegrowth[141],timegrowth[182]])


Lukcsv=[DAP,Obsluk1,LUKICSV]
Lukcsv=np.array(Lukcsv)
Lukcsv=Lukcsv.T

df = pd.DataFrame(Lukcsv)
df.to_csv('Validation_Luki_to_plot.csv',index=False)

x=Obslukstat
y=LUKIstat

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
    plt.plot(xp,a[0]*xp+a[1],'k-')
    plt.ylabel('Simulated [g/m2]',**font_1,color='black')
    plt.xlabel("Observed [g/m2]",**font_1,color='black')
    plt.annotate("PBIAS= "+ str(PBIAS_LUKI),(1300,500),**font_1,color='black',size=15)
    plt.annotate("NSE= "+ str(NSE_LUKI),(1300,600),**font_1,color='black',size=15)
    plt.annotate("R2= "+ str(R2luk),(1300,400),**font_1,color='black',size=15)
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

"_________AJAHUIRI - BIOM 1998-1999___________"


boundsY = np.linspace(0,1800,7,dtype=int)

fig, ax = plt.subplots(figsize=(12,6))
font = {'fontname':'Calibri','fontsize':18}
plt.plot_date(timegrowth, LUKI, 'o-',markersize=8,color='r',markeredgecolor='r',linewidth=2,
              markerfacecolor='r')
plt.plot_date(timeobse, Obsluk, 'o',markersize=8,color='r',markeredgecolor='r',linewidth=2,
              markerfacecolor='r')


plt.ylabel('AJAHUIRI biomass [g/m2]',**font,color='k')
plt.xlabel("Month",**font,color='k')
plt.yticks(boundsY, boundsY,fontsize=15,color='k')
plt.annotate("PBIAS= "+ str(PBIAS_LUKI),(729708,500),**font,color='R',size=20)
plt.annotate("NSE= "+ str(NSE_LUKI),(729708,600),**font,color='R',size=20)
plt.annotate("R2= "+ str(R2luk),(729708,400),**font,color='R',size=20)
plt.xticks(fontsize=15,color='k')
plt.title("Calibration AJAHUIRI PUCHUNI -  Biomass growth",fontsize=20,fontstyle="italic",color='k')
plt.legend(["Simulated","Observed"],fontsize=17)
plt.show()













