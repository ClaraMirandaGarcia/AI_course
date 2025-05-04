# 🧠 Neural Network from Scratch – MNIST Classifier

## 📄 Project Overview
This project implements a simple neural network using NumPy from scratch to classify the MNIST handwritten digit dataset. It covers core neural network components such as forward propagation, backpropagation, and training loops.

## ⚙️ Features
- Custom neural network class
- Sigmoid and softmax activations
- Manual gradient computation (no external libraries like TensorFlow or PyTorch)
- MNIST dataset loading from local CSV

## 🗃️ Structure
- `ann.py`: Defines the neural network architecture and training process.
- `main.py`: Loads the dataset and trains the model.
- `mnist_train.csv`, `mnist_test.csv`: MNIST data files.
- `results.png`: Sample prediction results plotted after training.

## ▶️ How to Run
```bash
python main.py
```

# 🤖 Artificial Neural Network – Manual Implementation

## 📄 Project Overview
This project is a foundational implementation of an Artificial Neural Network (ANN) to recognize patterns in synthetic and real data. It explores key neural network operations and manual training processes.

## ⚙️ Features
- Neural network built from scratch
- Fully connected layers and sigmoid activation
- Manual training loop using gradient descent
- Visualizations of training and predictions

## 🗃️ Structure
- `ann.py`: Custom ANN implementation
- `main.py`: Entry point to load data and train the network
- `data.csv`: Input dataset for training and evaluation
- `predictions.png`: Output visualizations of classification results

## ▶️ How to Run
```bash
python main.py
```
# 🧪 Perceptron and Activation Functions – AI Lab

## 📄 Project Overview
This project contains exercises focused on understanding AI concepts through hands-on simulations. It includes optimization of activation functions, error computation, and performance evaluation.

## ⚙️ Features
- Perceptron simulation
- Sigmoid and tanh activation function comparison
- Evaluation of classification performance
- Use of external packages like scikit-learn and matplotlib

## 🗃️ Structure
- `Perceptron.ipynb`: Jupyter notebook with interactive code and plots
- `data.csv`: Dataset used for experiments

## ▶️ How to Run
Open the Jupyter notebook:
```bash
jupyter notebook Perceptron.ipynb
```

# 🌐 Sensor Data Prediction with Neural Network

## 📄 Project Overview
This project focuses on training a neural network to predict sensor readings using real-world IoT data. It includes preprocessing, training, and evaluation of a regression model using PyTorch.

## ⚙️ Features
- Use of PyTorch for deep learning
- Multi-layer perceptron for regression
- Normalization and denormalization of data
- Evaluation using MSE and graphical results

## 🗃️ Structure
- `network.py`: Defines the neural network architecture
- `main.py`: Trains and evaluates the model
- `sensor.zip`: Compressed dataset (original CSV >100MB)
- `results.png`: Output of predictions vs ground truth

## ▶️ How to Run
Unzip the dataset first:
```bash
unzip sensor.zip
```

Then run:
```bash
python main.py
```

