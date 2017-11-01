# linearly-constrained-gaussian-processes

This repository contains code for reproducing the experiments in 

"Linearly constrained Gaussian processes" (2017) by Carl Jidling, Niklas Wahlström, Adrian Wills, and Thomas B. Schön.

The code is written in Matlab. The folder "simulation-example" contains the code used for the simulated example in section 5.1. The script "simulation_fieldplots.m" produces the field plots, and the script "simulation_study.m" performs the accuracy study. 

The folder "real-data-experiment" contains the code used for the real-data experiment in section 5.2. The script performing the study is "real_data_study.m", located in the subfolder "study". This folder also contains the data, stored in the file "dataSet14.mat" (see description below).

Note that the scripts do not use a fixed random seed, so the results obtained for different executions will not be identical.

In the folder "real-data-experiment/study" the magnetometer data is presented. It was collected on June 3, 2015 (2015-06-03) in the VICON LAB, University of Linköping, Sweden and has previously been presented in "Modeling and interpolation of the ambient magnetic field by Gaussian processes" IEEE Transactions on Robotics, 2017 by Solin et al.  See more details about the experiment design and the preprocessing in the supplementary material.

The data is stored in a matlab .mat-file and constist of a struct with multiple fields.
time - The time stamp of the collected measurement.
acc -  The accelerometer measurement in x, y, z coordinate, repectively.
gyr -  The gyro measurement in x, y, z coordinates in the global coordinate frame.
mag -  The gyro measurement in x, y, z coordinates in the global coordinate frame.
quat - The orientation of the platform relative to the global coordinate fram described witha 4D unit-norm quaternion.
