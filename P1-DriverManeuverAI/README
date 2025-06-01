🚗 Artificial Intelligence in the Automotive Industry – Driver Maneuver Detection

📌 Overview
This project is part of a broader effort to enhance road safety and driving experience through Advanced Driver Assistance Systems (ADAS). A key component of ADAS is Driver Behaviour Modelling (DBM), which focuses on detecting and predicting driver actions using:

  1.  Direct observation (e.g., front-facing cameras)
  2.  Indirect observation via vehicle sensors.

This practice involves analyzing time-series sensor data to identify characteristic sequences associated with specific driving maneuvers.

🎯 Objective
Using the provided dataset of driving sessions labeled with various maneuvers, the main tasks are to:

    ✅ Select at least one specific maneuver.
    🔍 Identify the sequence of atomic actions characterizing that maneuver.
    🧹 Preprocess sensor signals to remove redundancy and retain relevant features.
    ⏱ Apply windowing techniques and discretization when appropriate.


🧠 Hierarchical Driver Behaviour Model
    This work is based on a hierarchical representation of driver activity, structured into four decision levels:

    Behavioral – Personal traits, state, and long-term habits.
    Strategic – Route planning and trip-level decisions.
    Tactical – Real-time maneuver-level decisions (e.g., overtaking).
    Operational – Low-level control (steering, braking, acceleration).

Each level influences the lower ones, with Behavioral having the highest control priority.

🛠 Implemented Solution
    The main implementation is available in the notebook Practica1Grupo5.ipynb, and includes:

        1. Maneuver Selection
        A specific driving maneuver (e.g., overtaking) is selected for detailed analysis.

        2. Data Loading & Exploration
        All relevant user files for the chosen maneuver are processed.

    Both maneuver and non-maneuver instances are inspected.

⚙️ 3. Preprocessing
    Sliding window techniques to capture temporal dependencies.

        Noise filtering and signal smoothing.
        Optional discretization of continuous signals.

🧪 4. Modeling
    A machine learning classifier is trained to detect patterns associated with the selected maneuver.
    Different window sizes and signal transformations are evaluated.

📊 5. Evaluation
    Model performance is assessed using metrics such as accuracy, precision, and recall.

📁 Dataset & File Structure
    The dataset is provided in the ManiobrasSimulador.zip archive. To ensure proper execution of the notebook:

    Unzip the archive in the same directory as the notebook.
    The expected file structure is as follows:

./ManiobrasSimulador/
  ├── Driver1/
  │    └── STISIMData_Overtaking.xlsx
  ├── Driver2/
  │    └── STISIMData_Overtaking.xlsx
  ...
  └── Driver5/
       └── STISIMData_Overtaking.xlsx
📦 Requirements
Python 3.8+
Required packages (listed in requirements.txt):

pip install -r requirements.txt
Key libraries:
    numpy
    pandas
    scikit-learn
    matplotlib
    river – for streaming-based incremental learning

📚 References
Goran Andonovski et al. (2020). Detection of driver maneuvers using evolving fuzzy cloud-based system. IEEE Link
Igor Skrjanc et al. (2018). Evolving cloud-based system for the recognition of drivers' actions. Read here

