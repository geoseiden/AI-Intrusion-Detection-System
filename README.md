# Optimized AI Intrusion Detection System for Network Security

## Overview

This repository contains an Intrusion Detection System (IDS) built using machine learning techniques, specifically Random Forest Regression, to detect and classify network intrusions. The system is optimized using hyperparameter tuning techniques such as Grid Search and Random Search to achieve better performance. It serves as a tool to enhance network security by identifying and responding to potential threats.

## Features

- **Random Forest Regression:** Utilizes Random Forest Regression, a powerful ensemble learning algorithm, for intrusion detection.
- **Hyperparameter Optimization:** Implements hyperparameter tuning using both Grid Search and Random Search to find the optimal parameters for the Random Forest model.
- **Preprocessing:** Supports preprocessing of data including feature scaling and label encoding to prepare the data for training.
- **Performance Evaluation:** Provides evaluation metrics such as R^2 score to assess the performance of the trained model.
- **Customizable:** Easily customizable and extendable for specific network environments and requirements.

## Getting Started

### Prerequisites

- Python 3.x
- pandas
- scikit-learn
- joblib
- multiprocessing

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/geoseiden/AI-Intrusion-Detection-System.git
    ```

2. Navigate to the project directory:

    ```bash
    cd AI-Intrusion-Detection-System
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Place your training and testing datasets in the `Dataset` directory.
2. Modify the `train_file` and `test_file` variables in the `main` function of ipnyb to point to your dataset files.
3. Run the ipnyb file
4. The system will perform hyperparameter tuning, train the model, and output the test scores for both Grid Search and Random Search.

## Acknowledgements

- This project was inspired by the need for robust intrusion detection mechanisms in modern network environments.
- We acknowledge the contributions of the open-source community and the developers of scikit-learn for providing powerful machine learning tools.

