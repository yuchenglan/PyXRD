#using numpy module to read data in # much faster than CSV module 
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline #for smooth points
from scipy import signal # remove background of XRD pattern

#loading data  
x, y = np.loadtxt('FO_100.csv', delimiter=' ', unpack=True)
# remove linear backgroud
y_detrended = signal.detrend(y)
#Detrend with a 2d order polynomial
#bg_parameter = np.polyfit(x, y, 1)
#bg = np.polyval(bg_parameter, x)
#y_bg = y - bg
# detrend manually
x1 = (5, 10, 20, 30, 40, 50, 60, 70, 80)
y1 = (280, 660, 1500, 2350, 3100, 3800, 4500, 5090, 5700)
bg_parameter = np.polyfit(x1, y1, 3)
bg = np.polyval(bg_parameter, x)
y_bg = y - bg

#figure size etc
fig = plt.figure(figsize=(3, 1.85), dpi=100)

# use filter to remove background noise
# 1. welch filter
#fs = 10000
#f, int_welch = signal.welch(y_bg, fs, scaling = 'spectrum')
#plt.plot(f, int_welch, color = 'r') # label='y - polynomial bg')
# 2. savgol filter
#smooth curve using savgol filter
yhat = signal.savgol_filter(y_bg, 11, 3) # window size 101, polynomial order 3
#smoothX = np.linspace(15, 70, 2750)
#spl = make_interp_spline(x, y, k=3) #BSpline object
#smoothY = spl(smoothX)
plt.plot(x, yhat + 2000, linewidth = 0.5, linestyle = '-', color = 'b', label = 'BG + smoothness (savgol)') # more peak details reserved
#3. lfilter
n = 10 # the larger n is , the smooth is the curve
b = [1.0/n] * n
a = 1
int_lfilter = signal.lfilter(b, a, y_bg)
plt.plot(x, int_lfilter + 1000, linewidth = 0.5, linestyle = '-', color = 'g', label = 'BG + smoothness (lfilter)') # peak shift right  !_!
# 4. filtfilt, peak shape changed at peak root.  Not good filter (lowpass and highpass) for XRD 
#b2, a2 = signal.butter(3, 0.05, 'lowpass') # or lowpass or highpass
#int_filtfilt = signal.filtfilt(b2, a2, y_bg)
#plt.plot(x, int_filtfilt + 3000, linewidth = 0.5, linestyle = '-', color = 'pink', label = 'BG + smoothness (filtfilt)') # peak shift right  !_!

#legend etc of figure
#orginal data
plt.plot(x, y - 1000, linewidth = 0.5, linestyle = '-', color = 'k', label='raw data')
# detrended data
#plt.plot(x, y_detrended + 50, color = 'r', label='$detrended$')
# backgroud curve
#plt.plot(x, bg, label='polynomial bg')
#data after remove polyfitted background 
plt.plot(x, y_bg, linewidth = 0.5, linestyle = '-', color = 'r', label = 'background removal') # label='y - polynomial bg')
# data after remove polyfitted background and SG fitting
#plt.plot(x, yhat, "k")#, label='filtered')
#plt.plot(smoothX,smoothY, label='Loaded from file!')
plt.xlabel(r'2 $\theta$ ($\degree$)', fontsize=12)
plt.ylabel('Intensity (a. u.)', fontsize=12)
#plt.xlim(np.min(x)*0.9, np.max(x)*1.1)
plt.xlim(32.5, 77.5)
plt.xticks([35, 40, 45, 50, 55, 60, 65, 70, 75])
#plt.minorticks_on()
#plt.ylim(np.min(y)*0.6, np.max(y)*1.2)
plt.ylim(-500, 5500)
plt.tick_params(axis="y", direction="in", labelsize=10)
plt.tick_params(axis="x", direction="in", labelsize=10)
#plt.title('x-ray diffraction')
plt.legend(fontsize=6, frameon=False)
#plt.grid() # show grid

#setup tick and label
plt.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)

#save figure
#plt.savefig('test.png', dpi=100)
#plt.tight_layout()
plt.tight_layout(pad=0.2, w_pad=-0.2, h_pad=-0.2)
plt.savefig('FO_100_diffractiont.pdf', dpi=500)

plt.show()


#using csv module to read data in
#much slower
#import matplotlib.pyplot as plt
#import csv
#
#x = []
#y = []

#with open('TiO2-XRD.txt','r') as csvfile:
#    plots = csv.reader(csvfile, delimiter=',')
#    for row in plots:
#        x.append(int(row[0]))
#        y.append(int(row[1]))
