# AI for Real-World Applications – Practice Series
This repository showcases a collection of four practical projects applying Artificial Intelligence (AI) and Machine Learning (ML) to real-world challenges in diverse domains: automotive, medical imaging, business text analysis, and industrial IoT.

Each practice explores a different problem space, applying data-driven techniques and machine learning models with domain-appropriate tools.

## Practice 1: Driver Maneuver Detection – AI in the Automotive Industry
Detecting and characterizing driving maneuvers from sensor-based data collected during driving sessions.

* Goal: Identify characteristic sequences of atomic actions from vehicle signals for a selected driving maneuver.
* Dataset: Time-series driving logs labeled with different maneuvers.

Methods Used:

1. Preprocessing with sliding windows and signal filtering.
2. Classical ML classifiers for temporal pattern recognition.
3. Tools: Python, NumPy, Pandas, scikit-learn, Matplotlib
4. Exploration of streaming-based learning with River

## Practice 2: Histopathological Image Classification – AI in Healthcare
Deep Learning-based classification of medical images into tissue types and tumor detection.

* Goal: Build both a multiclass and a binary classifier for colorectal histological tissue images.
* Dataset: CRC-VAL-HE-7K, labeled medical image patches.

Model Used:

1. EfficientNetB0 CNN architecture
2. Dropout, Swish activation, Global Average Pooling
3. Categorical and binary cross-entropy loss

Results:

* ~90% training accuracy (binary)
* ROC AUC ≈ 0.65

Tools: Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib

## Practice 3: Text Classification and Topic Modeling – AI in Business
Supervised and unsupervised NLP techniques for business document classification and topic discovery.

* Goal: Automatically classify and explore topics in a labeled text corpus.
* Dataset: BBC News articles (bbc-train, bbc-test)

Models Used:

1. Supervised Deep Learning (MLP with Rectified units)
2. LDA for topic modeling
3. Word2Vec for semantic similarity
4. K-Means clustering on word embeddings

Tools: RapidMiner, Text Processing Extension, Operator Toolbox, Word2Vec plugin

## Practice 4: Sensor Anomaly Detection – AI for Industrial IoT
Detecting anomalies in sensor data from industrial machinery for predictive maintenance.

* Goal: Identify abnormal behavior in time-series sensor readings.
* Dataset: sensor.csv, with operational states: NORMAL, BROKEN, RECOVERING

Pipeline:

1. Data cleaning, imputation, MinMax scaling
2. Dimensionality reduction using PCA

Methods:

1. Non-Incremental:

* Interquartile Range (IQR)
* K-Means Clustering
* Isolation Forest

2. Incremental:

* Streaming K-Means (WIP)
* Half-Space Trees (WIP)

Tools: Python, scikit-learn, River (planned), Matplotlib, Seaborn

# Technologies Used Across Projects
Area	Tools & Libraries
ML/DL	TensorFlow, Keras, scikit-learn, River
Data Processing	NumPy, Pandas, OpenCV, Matplotlib, Seaborn
NLP	RapidMiner, Text Processing Extension, Word2Vec
📷 Imaging	EfficientNetB0, CNN, GlobalPooling, Dropout
🔁 Streaming Models	River (Half-Space Trees, Incremental KMeans – WIP)


