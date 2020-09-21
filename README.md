# GTA.
A Graph-Based Temporal Attention Framework for Multi-Sensor Traffic Flow Forecasting
![A Graph-Based Temporal Attention Framework for Multi-Sensor Traffic Flow Forecasting](https://github.com/skzhangPKU/GTA/blob/master/figures/framework.png)

# Predictions
The directory contains the prediction results of GTA  at 15 minutes, 30 minutes, and 60 minutes.

# Dataset
We evaluate GTA using real traffic data collected from all the strategic road network in England. The traffic dataset includes average speed, traffic flow, time period, location, and date of 249 monitoring stations. The sensors are located from site A414 between M1 J7 and A405, which covers several cities that include Manchester, Liverpool, and Blackburn.  Specifically, we use a whole year of traffic data ranging from January 1st, 2014 to December 31st, 2014 for the experiments. The total number of data entries is 8,724,960, the mean value of traffic volume is 466. 

We process the collected traffic data by normalizing them to [0, 1] before we feed them into the algorithm. Because the dataset does not contain road network distance between two monitoring stations, we enhance the data with these topology information based on distances collected from Google Services.

The distribution of monitoring stations studied in our experiment.
![The distribution of monitoring stations](https://github.com/skzhangPKU/GTA/blob/master/figures/ENG-HW.png)

# Code
Please contact me via my email (skzhang@pku.edu.cn)

