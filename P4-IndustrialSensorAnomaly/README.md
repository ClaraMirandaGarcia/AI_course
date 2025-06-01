# Sensor Anomaly Detection in Industrial Machinery
## Overview
This project explores various data science and machine learning techniques for detecting anomalies in sensor data collected from industrial machinery. The primary objective is to identify patterns indicative of unusual behavior or potential equipment malfunctions, thereby enabling predictive maintenance and operational optimization.

The practice progresses through the following stages:

1. Data Exploration and Preprocessing: Understanding the dataset, handling missing values, and scaling features.
2. Dimensionality Reduction: Applying Principal Component Analysis (PCA) to reduce the number of features while retaining essential information for anomaly detection.
3. Anomaly Detection: Implementing and comparing different non-incremental and incremental algorithms.

## Dataset
The project utilizes the sensor.csv dataset, which contains time-series readings from various sensors. The dataset includes a machine_status column indicating the operational state of the machine (NORMAL, BROKEN, RECOVERING).

The preprocessing steps include:
1. Handling duplicate entries.
2. Dropping irrelevant columns (Unnamed: 0, potentially sensor_15 due to missing data).
3. Converting the timestamp column to a datetime format.
4. Imputing missing numerical values with the mean of the respective column.
5. Scaling numerical features using MinMaxScaler.

## Methodology
The project employs both non-incremental and incremental anomaly detection methods.

1. Non-Incremental Methods: These methods process the entire dataset at once to identify anomalies.

* Interquartile Range (IQR): Anomalies are detected based on deviations from the first and third quartiles of the data (specifically on the PCA components).
* K-Means Clustering: Data points are grouped into clusters, and anomalies are identified as points significantly distant from their assigned cluster centroid.
* Isolation Forest: This tree-based algorithm isolates anomalies by randomly partitioning the data space.

2. Incremental Methods:These methods are designed to handle data streams and update their models as new data arrives.

* Incremental K-Means (WIP): An adaptation of the K-Means algorithm that updates cluster centers incrementally.
* Half Space Trees (WIP): An online anomaly detection algorithm that is effective for detecting anomalies in streaming data by recursively partitioning the data space.

## Analysis and Results
The effectiveness of each anomaly detection method is evaluated and visualized. The analysis focuses on:

* Identifying data distributions and patterns through descriptive statistics and correlation heatmaps.
* Visualizing anomaly detection results on the reduced-dimensional PCA space.
* Comparing the performance and characteristics of incremental versus non-incremental approaches.