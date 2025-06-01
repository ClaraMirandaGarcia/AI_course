🧠 PathoNet: Deep Learning for Histopathology Image Classification
🧬 Overview
    This project applies Deep Learning techniques to the field of digital histopathology, aiming to assist medical diagnosis through automated image classification. Histological image analysis, traditionally performed manually by pathologists, can be enhanced by AI tools to increase speed, accuracy, and objectivity.

The main goals of this practice are:

    1. Multiclass classification of histological tissue images into 9 types.
    2. Binary classification: detecting the presence or absence of tumor tissue.

The work is inspired by the bachelor's thesis:
📖 "Clasificación de imágenes médicas con técnicas de Deep Learning" by Juan José Carballo Pacheco (link)

📂 Dataset
    Source: [CRC-VAL-HE-7K dataset (NCT tissue bank)]

    Includes images of human colorectal tissue samples, categorized into multiple classes.

    Images are split into:

    80% training
    20% validation

    Balanced using data augmentation (rotation, contrast variations).

🏗️ Model Architecture
    We use a Convolutional Neural Network (CNN) based on the EfficientNetB0 architecture:

        ✅ Lightweight and efficient

        🧠 Swish activation function (instead of ReLU)

        ⚙️ Modified for:
            Multiclass classification with softmax output
            Binary classification with a sigmoid output

Key layers and components:

Input: 224x224 RGB images

Preprocessing: Rescaling & normalization

    Convolution + Pooling blocks
    Dropout layers for regularization
    Fully connected dense layers
    Categorical/binary cross-entropy loss

🚀 Training
    Frameworks: TensorFlow/Keras

    Epochs: 40
    Batch size: 64
    Optimizer: Adam (default LR: 0.001)
    No pretrained weights (training from scratch)

    Execution times:

        Multiclass training: ~2h 7m
        Binary training: ~32 minutes

    Hardware: Intel Core i9, RTX 4060 (8GB), 32GB RAM

Results
✅ Accuracy Trends
    Multiclass: Increasing training accuracy, flat validation curve → overfitting risk

    Binary: Stable and high accuracy from early epochs → robust model

🔍 ROC Curve (Binary Model)
    AUC = 0.65
    Indicates the model performs better than random guessing, but has room for improvement.

🧩 Confusion Matrices
    Multiclass: Best performance in tumor class
    Binary: High true positives, some false positives; low false negatives

📦 Requirements
    Python 3.8+

    Libraries:
        tensorflow, keras
        opencv-python
        numpy
        scikit-learn
        matplotlib

Install with:
    pip install -r requirements.txt

📁 File Structure

    ├── Practica2Grupo5.ipynb       # Main implementation notebook
    ├── MemoriaPractica2Grupo5.pdf  # Technical report
    ├── requirements.txt            # Python dependencies
    └── dataset/                    # Unzipped CRC-VAL-HE-7K 
🔬 Conclusions
    1. EfficientNetB0 proves effective in both multiclass and binary classification of histological images.
    2. Binary model shows strong potential for real-world tumor detection support.
    3. Future work includes testing on unseen datasets and clinical validation.