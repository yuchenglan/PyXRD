# PyXRD
Analysis of X-ray Dowder Diffraction Patterns

Created and maintained by Yucheng Lan and Kit Sze

Version 0.22.03ML was created and maintained by Yucheng and Grace 

## Versions
1. Version 0.21.01a: Standardize plot of original data, withouth background removal.  Peak intensities were normalized (max: 100).  
2. Version 0.21.01b: background removal with linear (detrend) fitting (disabled).
3. Version 0.21.01c: background removal with polynomial formula fitting.
4. Version 0.22.03: background removal + smoothness with lfilter or savgol filter.  Bothe peaks and BG wwere smoothened. Manually pick up BG reference data.
5. Version 0.22.03ML: auto background pick up, background removal with polynomial formula fitting, ML (K-Means) cluster noisy BG and peaks.  Noisy BG was filtered using savgol filter (peaks kept as raw values).
5. Name was changed in March 2022 to PyXRDPlotting as there is one PyXRD (https://github.com/PyXRD/PyXRD) to computer-model X-ray diffraction (XRD) patterns of disordered lamellar structures.
6. Version 0.03: GUI (coming).
7. Version 1.0 update (coming).

## Other related open-sources
1. https://github.com/dkriegner/xrayutilities: analyze and simulate x-ray diffraction data.  It consists of a Python package and several routines coded in C.
2. https://github.com/PyXRD/PyXRD:  python implementation of the matrix algorithm for computer modeling of X-ray diffraction (XRD) patterns of disordered lamellar structures. 
