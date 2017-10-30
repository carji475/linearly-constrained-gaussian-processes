# linearly-constrained-gaussian-processes

This repository contains code for reproducing the experiments in 

"Linearly constrained Gaussian processes" (2017) by Carl Jidling, Niklas Wahlström, Adrian Wills, and Thomas B. Schön.

The code is written in Matlab. The folder "simulation-example" contains the code used for the simulated example in section 5.1. The script "simulation_fieldplots.m" produces the field plots, and the script "simulation_study.m" performs the accuracy study. 

The folder "real-data-experiment" contains the code used for the real-data experiment in section 5.2. The script performing the study is "real_data_study.m", located in the subfolder "study". This folder also contains the data, stored in the file "dataSet14.mat" (see description below).

Note that the scripts do not use a fixed random seed, so the results obtained for different executions will not be identical. 
