# Aurora Forecasting Using Solar Wind and Sunspot Data

## Overview

I developed this project during **Aurora Parsec**, a national-level space weather hackathon organized by IIT Dharwad. The objective of the challenge was to build a predictive model capable of forecasting geomagnetic storm activity using historical solar and space weather observations.

The project focuses on predicting the **Dst (Disturbance Storm Time) Index**, an important indicator used to measure geomagnetic disturbances in the Earth's magnetosphere. Accurate prediction of the Dst index can help researchers understand the impact of solar activity on Earth's space environment.

As part of this project, I designed and trained a deep learning model using solar wind and sunspot datasets provided during the hackathon. The solution achieved promising results and helped my team secure **Third Prize in the Aurora Parsec Hackathon**.

---

## Problem Statement

Solar events such as coronal mass ejections and solar wind fluctuations can significantly affect Earth's magnetic field. Predicting these disturbances is an important task in space weather forecasting.

The goal of this project was to develop a machine learning model that could learn patterns from historical solar wind and sunspot measurements and estimate the corresponding Dst index values.

---

## Dataset Description

The model was trained using three datasets:

### 1. Solar Wind Data

Contains measurements related to solar wind conditions, including magnetic field components, density, speed, and temperature.

### 2. Sunspot Data

Contains information about solar activity represented through smoothed sunspot observations.

### 3. Dst Labels

Contains historical Dst index values used as target variables during supervised learning.

---

## Features Used

The following parameters were selected as input features:

* bx_gse
* by_gse
* bz_gse
* theta_gse
* phi_gse
* bx_gsm
* by_gsm
* bz_gsm
* theta_gsm
* phi_gsm
* bt
* density
* speed
* temperature
* source

---

## Methodology

### Data Preprocessing

The following preprocessing steps were performed:

* Loaded solar wind, sunspot, and label datasets.
* Converted timedelta information into numerical day values.
* Merged datasets using common temporal information.
* Handled missing values using mean imputation.
* Standardized numerical features using StandardScaler.
* Split the dataset into training and testing subsets.

### Model Development

A feed-forward neural network was implemented using TensorFlow and Keras.

Architecture:

* Input Layer
* Dense Layer (128 neurons, ReLU activation)
* Dense Layer (64 neurons, ReLU activation)
* Output Layer (1 neuron)

### Training Configuration

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Epochs: 20
* Batch Size: 32
* Validation Split: 20%

---

## Evaluation

The model was evaluated using **Root Mean Squared Error (RMSE)**, which measures the average prediction error between the actual and predicted Dst values.

The trained model demonstrated the ability to learn relationships between solar activity indicators and geomagnetic storm intensity, making it suitable for preliminary space weather forecasting tasks.

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* TensorFlow
* Keras
* Google Colab

---

## Project Structure

```text
aurora/
│
├── Train_on_this/
│   ├── solar_wind.csv
│   ├── sunspots_smooth.csv
│   └── labels(dst).csv
│
├── Dont_Train_on_this/
│   ├── solar_wind.csv
│   └── sunspots_smooth.csv
│
├── trained_model1.h5
├── predictions.csv
├── train_model.py
└── predict.py
```

## How to Run

1. Install the required libraries.

```bash
pip install pandas numpy scikit-learn tensorflow
```

2. Update dataset paths inside the Python scripts.

3. Train the model.

```bash
python train_model.py
```

4. Generate predictions.

```bash
python predict.py
```

5. The generated predictions will be stored in:

```text
predictions.csv
```

---

## Achievement

This project was developed for the **Aurora Parsec Hackathon organized by IIT Dharwad**. Competing against teams from different institutions, our solution was recognized for its effectiveness in predicting geomagnetic storm activity and secured **Third Prize in the competition**.

The experience provided valuable exposure to space weather analytics, machine learning model development, and real-world problem solving using scientific datasets.

---

## Author

**Sidhant Mishra**

Bachelor of Engineering
Chandigarh University

Interests:

* Data Analytics
* Machine Learning
* Space Weather Forecasting
* Artificial Intelligence
