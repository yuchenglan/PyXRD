# third-party libraries
import numpy as np # for data array, polyfit, ?
import scipy as sp # saugol_filter, Savitzky-Golay filter
from sklearn.cluster import KMeans # ML clustering
import matplotlib.pyplot as plt


# Steps and 
# 1. loading data,
# 2. find background, fit, and remove background,
# 3. machine learn to cluster diffraction peaks and non-peak ranges (BG),
# 4. de-noise of non-peak ranges (BG),
# 5. combine de-noised BG and un-filtered peaks,
# 6. plot, save data and figure

# libraries: loading data <- numpy, np.loadtext
# libraries: find background / fitting <- numpy, np.polyfit
# libraries: machine learn to cluster <- scikit_learn, K-Means
# libraries: de-noise <- scipy, saugol_filter
# libraries: plotting <- matplotlib, pyplot

# load XRD data in cvs to produce array
# syntax: 
theta, intensity = np.loadtxt("PyXRDPlotting_ML.csv", delimiter = ' ', unpack = True) # type of theta, intensity: np.array
#plt.plot(theta, intensity)
#plt.show()


# find background data and fit a polynomial curve using polyfit based on least squares method 
# 2.1. pickup one lowest intensity value in every 5degree (2theta) gap as BG reference
# 2.2. fit the lowest intensity(s) / BG referece using a polynomial curve
# 2.3. removal background based on the fitting curve
# BG of XRD: step: 0.02degree, 1 degree ~ 50 data, 5 degree = 250 data
# BG of XRD: 2theta range: 5 - 80 degree

# 2. 1. picking up background reference data
# syntax: np.min()

# in theta = 5-7.5 degree   
range5 = intensity[0:124] # intensity in 5-7.5 degree 250/2 = 125
lowest5 = np.min(range5) # lowest intensity in the 5 - 10 degree
lowest5index = np.where(range5 == lowest5)[0][0] # find first index position where intensity is equal to lowest5 in 5-10 degree
# theta of lowest5
lowest5theta = theta[0 + lowest5index] # theta value where intensity = lowest5

# in theta = 7.5-10 degree 
range7 = intensity[125:249] # intensity in 7.5-10 degree
lowest7 = np.min(range7) # lowest intensity in the 5 - 10 degree
lowest7index = np.where(range7 == lowest7)[0][0] # find first index position where intensity is equal to lowest5 in 5-10 degree
# theta of lowest5
lowest7theta = theta[125 + lowest7index] # theta value where intensity = lowest5

# in theta = 10-12.5 degree 
range10 = intensity[250:374] # intensity in 10-12.5 degree
lowest10 = np.min(range10) # lowest intensity in 10 - 15 degree
lowest10index = np.where(range10 == lowest10)[0][0] # find first index position where intensity is equal to lowest10 in 10-15 degree
# theta of lowest10
lowest10theta = theta[250 + lowest10index] # theta value where intensity = lowest10

# in theta = 12.5-15 degree 
range12 = intensity[375:499] # intensity in 12.5-15 degree
lowest12 = np.min(range12) # lowest intensity in 10 - 15 degree
lowest12index = np.where(range12 == lowest12)[0][0] # find first index position where intensity is equal to lowest10 in 10-15 degree
# theta of lowest12
lowest12theta = theta[375 + lowest12index] # theta value where intensity = lowest12

# in theta = 15-17.5 degree 
range15 = intensity[500:624] # intensity in 15-17.5 degree
lowest15 = np.min(range15) # lowest intensity in 15 - 17.5 degree
lowest15index = np.where(range15 == lowest15)[0][0] # find first index position where intensity is equal to lowest15 in 15-17.5 degree
# theta of lowest15
lowest15theta = theta[500 + lowest15index] # theta value where intensity = lowest15

# in theta = 17.5-20 degree 
range17 = intensity[625:749] # intensity in 17.5-20 degree
lowest17 = np.min(range17) # lowest intensity in 15 - 20 degree
lowest17index = np.where(range17 == lowest17)[0][0] # find first index position where intensity is equal to lowest15 in 17.5-20 degree
# theta of lowest17
lowest17theta = theta[625 + lowest17index] # theta value where intensity = lowest17
#print(lowest17, lowest17theta)

# in theta = 20-22.5 degree 
range20 = intensity[750:874] # intensity in 20-22.5 degree
lowest20 = np.max(range20) # lowest intensity in 20 - 22.5 degree
lowest20index = np.where(range20 == lowest20)[0][0] # find first index position where intensity is equal to lowest20 in 20-22.5 degree
# theta of lowest20
lowest20theta = theta[750 + lowest20index] # theta value where intensity = lowest20
#print(lowest20, lowest20theta)

# in theta = 22.5-25 degree 
range22 = intensity[875:999] # intensity in 22.5-25 degree
lowest22 = np.min(range22) # lowest intensity in 22.5 - 25 degree
lowest22index = np.where(range22 == lowest22)[0][0] # find first index position where intensity is equal to lowest22 in 22.5-25 degree
# theta of lowest22
lowest22theta = theta[775 + lowest22index] # theta value where intensity = lowest22
#print(lowest22, lowest22theta)

# in theta = 25-27.5 degree 
range25 = intensity[1000:1124] # intensity in 25-30 degree
lowest25 = np.min(range25) # lowest intensity in 25 - 30 degree
lowest25index = np.where(range25 == lowest25)[0][0] # find first index position where intensity is equal to lowest25 in 25-30 degree
# theta of lowest25
lowest25theta = theta[1000 + lowest25index] # theta value where intensity = lowest25

# in theta = 27.5-30 degree 
range27 = intensity[1124:1249] # intensity in 27.5-30 degree
lowest27 = np.min(range27) # lowest intensity in 27.5 - 30 degree
lowest27index = np.where(range27 == lowest27)[0][0] # find first index position where intensity is equal to lowest25 in 25-30 degree
# theta of lowest27
lowest27theta = theta[1125 + lowest27index] # theta value where intensity = lowest25

# in theta = 30-32.5 degree 
range30 = intensity[1250:1374] # intensity in 30-32.5 degree
lowest30 = np.min(range30) # lowest intensity in 30 - 35 degree
lowest30index = np.where(range30 == lowest30)[0][0] # find first index position where intensity is equal to lowest30 in 30-35 degree
# theta of lowest30
lowest30theta = theta[1250 + lowest30index] # theta value where intensity = lowest30

# in theta = 32.50-35 degree 
range32 = intensity[1375:1499] # intensity in 32.5-35 degree
lowest32 = np.min(range32) # lowest intensity in 32.5 - 35 degree
lowest32index = np.where(range32 == lowest32)[0][0] # find first index position where intensity is equal to lowest30 in 30-35 degree
# theta of lowest32
lowest32theta = theta[1375 + lowest32index] # theta value where intensity = lowest30

# in theta = 35-37.5 degree 
range35 = intensity[1500:1624] # intensity in 35-37.5 degree
lowest35 = np.min(range35) # lowest intensity in 35 - 40 degree
lowest35index = np.where(range35 == lowest35)[0][0] # find first index position where intensity is equal to lowest35 in 35-40 degree
# theta of lowest35
lowest35theta = theta[1500 + lowest35index] # theta value where intensity = lowest35

# in theta = 37.5-40 degree 
range37 = intensity[1625:1749] # intensity in 37.5-40 degree
lowest37 = np.min(range37) # lowest intensity in 35 - 40 degree
lowest37index = np.where(range37 == lowest37)[0][0] # find first index position where intensity is equal to lowest35 in 35-40 degree
# theta of lowest37
lowest37theta = theta[1625 + lowest37index] # theta value where intensity = lowest35

# in theta = 40-42.5 degree 
range40 = intensity[1750:1874] # intensity in 40-42.5 degree
lowest40 = np.min(range40) # lowest intensity in 35 - 40 degree
lowest40index = np.where(range40 == lowest40)[0][0] # find first index position where intensity is equal to lowest40 in 40-45 degree
# theta of lowest40
lowest40theta = theta[1750 + lowest40index] # theta value where intensity = lowest40

# in theta = 42.5-45 degree 
range42 = intensity[1875:1999] # intensity in 40-45 degree
lowest42 = np.min(range42) # lowest intensity in 35 - 40 degree
lowest42index = np.where(range42 == lowest42)[0][0] # find first index position where intensity is equal to lowest40 in 40-45 degree
# theta of lowest42
lowest42theta = theta[1875 + lowest42index] # theta value where intensity = lowest40

# in theta = 45-47.5 degree 
range45 = intensity[2000:2124] # intensity in 45-47.5 degree
lowest45 = np.min(range45) # lowest intensity in 45 - 50 degree
lowest45index = np.where(range45 == lowest45)[0][0] # find first index position where intensity is equal to lowest45 in 45-50 degree
# theta of lowest45
lowest45theta = theta[2000 + lowest45index] # theta value where intensity = lowest45

# in theta = 47.5-50 degree 
range47 = intensity[2125:2249] # intensity in 47.5-50 degree
lowest47 = np.min(range47) # lowest intensity in 45 - 50 degree
lowest47index = np.where(range47 == lowest47)[0][0] # find first index position where intensity is equal to lowest45 in 45-50 degree
# theta of lowest47
lowest47theta = theta[2125 + lowest47index] # theta value where intensity = lowest45

# in theta = 50-52.5 degree 
range50 = intensity[2250:2374] # intensity in 50-52.5 degree
lowest50 = np.min(range50) # lowest intensity in 50 - 55 degree
lowest50index = np.where(range50 == lowest50)[0][0] # find first index position where intensity is equal to lowest50 in 50-55 degree
# theta of lowest50
lowest50theta = theta[2250 + lowest50index] # theta value where intensity = lowest50

# in theta = 52.5-55 degree 
range52 = intensity[2375:2499] # intensity in 52.5-55 degree
lowest52 = np.min(range52) # lowest intensity in 50 - 55 degree
lowest52index = np.where(range52 == lowest52)[0][0] # find first index position where intensity is equal to lowest50 in 50-55 degree
# theta of lowest52
lowest52theta = theta[2375 + lowest52index] # theta value where intensity = lowest50

# in theta = 55-57.5 degree 
range55 = intensity[2500:2624] # intensity in 55-57.5 degree
lowest55 = np.min(range55) # lowest intensity in 55 - 60 degree
lowest55index = np.where(range55 == lowest55)[0][0] # find first index position where intensity is equal to lowest55 in 55-60 degree
# theta of lowest55
lowest55theta = theta[2500 + lowest55index] # theta value where intensity = lowest55

# in theta = 57.5-60 degree 
range57 = intensity[2625:2749] # intensity in 57.5-60 degree
lowest57 = np.min(range57) # lowest intensity in 55 - 60 degree
lowest57index = np.where(range57 == lowest57)[0][0] # find first index position where intensity is equal to lowest55 in 55-60 degree
# theta of lowest57
lowest57theta = theta[2625 + lowest57index] # theta value where intensity = lowest55

# in theta = 60-62.5 degree 
range60 = intensity[2750:2874] # intensity in 60-62.5 degree
lowest60 = np.min(range60) # lowest intensity in 60 - 65 degree
lowest60index = np.where(range60 == lowest60)[0][0] # find first index position where intensity is equal to lowest60 in 60-65 degree
# theta of lowest60
lowest60theta = theta[2750 + lowest60index] # theta value where intensity = lowest60

# in theta = 62.5-65 degree 
range62 = intensity[2875:2999] # intensity in 62.5-65 degree
lowest62 = np.min(range62) # lowest intensity in 60 - 65 degree
lowest62index = np.where(range62 == lowest62)[0][0] # find first index position where intensity is equal to lowest60 in 60-65 degree
# theta of lowest62
lowest62theta = theta[2875 + lowest62index] # theta value where intensity = lowest60

# in theta = 65-67.5 degree 
range65 = intensity[3000:3124] # intensity in 65-67.5 degree
lowest65 = np.min(range65) # lowest intensity in 65 - 70 degree
lowest65index = np.where(range65 == lowest65)[0][0] # find first index position where intensity is equal to lowest65 in 65-70 degree
# theta of lowest65
lowest65theta = theta[3000 + lowest65index] # theta value where intensity = lowest65

# in theta = 67.5-70 degree 
range67 = intensity[3125:3249] # intensity in 67.5-70 degree
lowest67 = np.min(range67) # lowest intensity in 65 - 70 degree
lowest67index = np.where(range67 == lowest67)[0][0] # find first index position where intensity is equal to lowest65 in 65-70 degree
# theta of lowest67
lowest67theta = theta[3125 + lowest67index] # theta value where intensity = lowest65

# in theta = 70-72.5 degree 
range70 = intensity[3250:3374] # intensity in 70-72.5 degree
lowest70 = np.min(range70) # lowest intensity in 70 - 75 degree
lowest70index = np.where(range70 == lowest70)[0][0] # find first index position where intensity is equal to lowest70 in 70-75 degree
# theta of lowest70
lowest70theta = theta[3250 + lowest70index] # theta value where intensity = lowest70

# in theta = 72.5-75 degree 
range72 = intensity[3375:3499] # intensity in 72.5-75 degree
lowest72 = np.min(range72) # lowest intensity in 70 - 75 degree
lowest72index = np.where(range72 == lowest72)[0][0] # find first index position where intensity is equal to lowest70 in 70-75 degree
# theta of lowest72
lowest72theta = theta[3375 + lowest72index] # theta value where intensity = lowest70

# in theta = 75-77.5 degree 
range75 = intensity[3500:3624] # intensity in 75-77.5 degree
lowest75 = np.min(range75) # lowest intensity in 75 - 80 degree
lowest75index = np.where(range75 == lowest75)[0][0] # find first index position where intensity is equal to lowest75 in 75-80 degree
# theta of lowest75
lowest75theta = theta[3500 + lowest75index] # theta value where intensity = lowest75

# in theta = 77.5-80 degree 
range77 = intensity[3625:3749] # intensity in 77.5-80 degree
lowest77 = np.min(range77) # lowest intensity in 75 - 80 degree
lowest77index = np.where(range77 == lowest77)[0][0] # find first index position where intensity is equal to lowest75 in 75-80 degree
# theta of lowest77
lowest77theta = theta[3625 + lowest77index] # theta value where intensity = lowest75

# 2.2. fitting BG reference using a polynomial curve
# create BG reference from lowest intensity in various ranges
bgRefIntensity = [lowest5, lowest7, lowest10, lowest12, lowest15, lowest17, lowest20, lowest22, lowest25, lowest27, lowest30, lowest32, lowest35, lowest37, lowest40, lowest42, lowest45, lowest47, lowest50, lowest52, lowest55, lowest57, lowest60, lowest62, lowest65, lowest67, lowest70, lowest72, lowest75, lowest77]
bgRefTheta = [lowest5theta, lowest7theta, lowest10theta, lowest12theta, lowest15theta, lowest17theta, lowest20theta, lowest22theta, lowest25theta, lowest27theta, lowest30theta, lowest32theta, lowest35theta, lowest37theta, lowest40theta, lowest42theta, lowest45theta, lowest47theta, lowest50theta, lowest52theta, lowest55theta, lowest57theta, lowest60theta, lowest62theta, lowest65theta, lowest67theta, lowest70theta, lowest72theta, lowest75theta, lowest77theta]
# fitting 
# syntax: numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
fittingParameter = np.polyfit(bgRefTheta, bgRefIntensity, deg = 3)

# 2.3. removal background to produce new data with BG removal
bgFittingIntensity = fittingParameter[3] + fittingParameter[2] * theta + fittingParameter[1] * theta * theta + fittingParameter[0] * theta * theta * theta
intensityBGRemoval = intensity - bgFittingIntensity
#plt.plot(theta, intensity)
#plt.plot(theta, intensityBGRemoval)
#plt.show()


# ML to find BG and peaks
# scikit-learning, ML K-Means clustering (unsupervised): -> BG cluster and peak cluster
from sklearn.model_selection import train_test_split # split training dataset (75%) and test dataset (25%)
from sklearn.cluster import KMeans # ML the k-means algorithm, naive k-means
# data clear
X = np.vstack((intensityBGRemoval, theta)).T # creat 2D data from 1D intensity (X, feather) and 1D theta (y ~ 0) for ML, to match K-means requrements
# split data 
X_train, X_test = train_test_split(X)
# training
# syntax:
# sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
# the number of d-dimensional vectors (to be clustered), number of clusters, number of iterations needed until convergence
model = KMeans(n_clusters = 2, random_state = 2, max_iter = 500) # 3 clusters: low intensity: noisy BG, med intensity: peak root, high intensity: peak;  # 25% intensity => BG .^^.
# 3-4 clusters should be better, will fix later
model.fit(X_train) # ML training
# prediction = model.predict(X_test)
label = model.fit_predict(X) # label feathers (X) using 0, 1
#print('label: ', label)
# prepare data of BG noise
label_noise = X[label == 0] # noisy BG: will be smoothed
# prepare data of peaks
#label_peakRoot1 = X[label == 0] # peak low root: will be smoothed
#label_peakRoot2 = X[label == 1] # peak high root: not smoothed
# prepare data of peaks
label_peak = X[label == 1] # peak: not smoothed
# plot noise BG and peaks
#plt.scatter(label_noise[:, 1], label_noise[:, 0], marker='o', c = 'r', s = 10, label = 'noise') # plot noise BG
#plt.scatter(label_peakRoot1[:, 1], label_peakRoot1[:, 0], marker='o', c = 'pink', s = 10, label = 'peak') # plot peak
#plt.scatter(label_peakRoot2[:, 1], label_peakRoot2[:, 0], marker='o', c = 'g', s = 10, label = 'peak') # plot peak
#plt.scatter(label_peak[:, 1], label_peak[:, 0], marker='o', c = 'b', s = 10, label = 'peak') # plot peak
#plt.show()


# more optional output
# fit_transform(X, y=None, sample_weight=None)
#score(X, y=None, sample_weight=None)


# smoothing BG noise with savgol_fitter
# syntax
# scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0)
from scipy.signal import savgol_filter
# produce filtered data
label_noise_filteredIntensity = savgol_filter(label_noise[:, 0], 11, 3) # window size 5, polynomial order 3; label_noise[:, 0] is intensity
label_noise_filteredTheta = label_noise[:, 1]
#test below
#plt.scatter(label_noise_filteredTheta, label_noise_filteredIntensity - 200, marker='o', c = 'r', s = 10, label = 'noise') # plot noise BG
#plt.plot(label_noise_filteredTheta, label_noise_filteredIntensity - 200, label = 'noise reducation') # plot noise BG, some peak roots were clustered as BG (should be OK as will be merged with peaks)
#plt.show()

#test below
#plt.scatter(label_peak[:, 1], label_peak[:, 0], marker='o', c = 'g', s = 10, label = 'peak') # plot peak
#plt.scatter(label_noise[:, 1], label_noise[:, 0], marker='o', c = 'r', s = 10, label = 'noise') # plot noise BG, x-intensity, y-2theta
#plt.plot(label_noise_filteredTheta, label_noise_filteredIntensity, label = 'noise reducation') # plot noise BG
#plt.show()


# combining smoothed BG and peaks
label_noise_filtered = np.vstack((label_noise_filteredIntensity, label_noise_filteredTheta)).T # create 2D np.array of filtered BG
label_total = np.vstack((label_peak, label_noise_filtered)) # included un-filtered peaks and filtered BG
# sort label_total
ML_data = label_total[label_total[:, 1].argsort()] # columnIndex = 1 (second column, 2theta)

#test
#print('sorted:', sorted)
#plt.scatter(sorted[:, 1], sorted[:, 0], marker='o', c = 'b', s = 10, label = 'sorted') # plot noise BG, x-intensity, y-2theta
#plt.plot(label_noise_filteredTheta, label_noise_filteredIntensity, label = 'noise reducation') # plot noise BG
#plt.show()


# plotting
# 1. plot raw data
plt.plot(theta, intensity, linewidth = 0.5, linestyle = '-', color = 'k', label='raw data')
# 2. plot data after remove polyfitted background 
plt.plot(theta, intensityBGRemoval, linewidth = 0.5, linestyle = '-', color = 'r', label = 'background removal')
# 3. plot data of BG after remove polyfitted background and SG-filtered noise
#plt.plot(label_noise_filteredTheta, label_noise_filteredIntensity, label = 'filtered noise') # plot noise BG
# 4. plot totall de-noised data with BG removal
#plt.scatter(sorted[:, 1], sorted[:, 0], marker='o', c = 'b', s = 10, label = 'ML smoothing') # plot noise BG, x-intensity, y-2theta
plt.plot(ML_data[:, 1], ML_data[:, 0]-2000,  linewidth = 0.5, linestyle = '-', c = 'b', label = 'ML smoothing') # plot noise BG, x-intensity, y-2theta
# label etc of figures
plt.xlabel(r'2 $\theta$ ($\degree$)', fontsize=12)
plt.ylabel('Intensity (a. u.)', fontsize=12)
plt.xlim(10, 80)
#plt.ylim(0, np.max(intensityBGRemoval)*1.2)
#setup tick and label
#plt.xticks([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
plt.minorticks_on()
plt.tick_params(axis="y", direction="in", labelsize=10)
plt.tick_params(axis="x", direction="in", labelsize=10)
#plt.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
plt.title('X-ray Diffraction after ML Data Cleaning')
plt.legend(fontsize=6, frameon=False)
plt.grid() # show grid
#plt.tight_layout()
#plt.tight_layout(pad=0.2, w_pad=-0.2, h_pad=-0.2)
plt.show()


#saving new data in csv
np.savetxt('PyXRD_ML_New.csv', data, delimiter=',')
# Saving figure
plt.savefig('PyXRD.png', dpi=500)
plt.savefig('PyXRD.pdf', dpi=500)




