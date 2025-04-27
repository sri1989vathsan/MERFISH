# MultiFISH analysis - ongoing work in progress

# This repository has a set of Jupyter Notebooks aimed at analysis dataset generated using a custom multiplexed imaging setup. The data is generated using Nikon microscopies and hte input files are a set of .nd2 files. 

## The script does the following:
## 1. Converts nd2 files to tiff files for futher ease of downstream analysis.
## 2. Maximum intensity projections of the images (as this would be used for further analysis downstream)
## 3. Image Registration to align images generated across multiple cycles
## 4. Spot detection to detect mRNAs across these images
