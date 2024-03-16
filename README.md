# Project Overview

This project utilizes protein 3D structure data and the BINA encoding dataset. It employs a swarm intelligence feature selection algorithm and combines it with machine learning models to construct an efficient predictor for accurately identifying lysine acetylation (Kace) sites in proteins. The project primarily utilizes Python as the programming language and relies on powerful libraries such as NumPy, Pandas, Matplotlib, Scikit-learn, XGBoost, among others, to handle large-scale datasets, perform complex mathematical computations, and implement data visualization.

## Project Configuration

Before using this project, please make sure that the following dependencies are installed on your system:

- **NumPy**: Used for handling large, multi-dimensional arrays and matrices, as well as performing extensive mathematical computations.
- **Pandas**: Provides efficient and user-friendly data structures and data analysis tools, particularly suitable for handling and analyzing input data.
- **Matplotlib**: A plotting library used to generate various static, dynamic, and interactive data visualization graphics.
- **Scikit-learn**: Provides simple yet effective tools for data mining and data analysis, particularly excelling in training and evaluating machine learning models.
- **XGBoost**: An optimized distributed gradient boosting library used for building, training, and predicting decision tree models, particularly excelling with large-scale datasets.

You can install these dependencies by executing the following commands:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost

```

It is recommended to use Python 3.x version to ensure the best compatibility and performance.

## User Guide

**Data Availability**: The datasets generated or analyzed during the current study are available in the PLMD database (http://plmd.biocuckoo.org/). Additionally, the original datasets regarding protein sequences can be found in the DBZ/Datasets/Protein_sequence directory.

**Data Preprocessing**: This project consists of six script files for data preprocessing. The first four scripts handle the raw data from the PLMD database to obtain protein 3D data. The fifth script processes the raw data of protein sequences to obtain BINA encoding data. The last script is used to merge these two datasets.

**Main Module**: This module implements feature selection on the dataset using six different feature selection algorithms (such as ant colony algorithm, genetic algorithm, etc.). It then performs classification using the XGBoost model optimized through grid search. Finally, this module generates a line plot of loss change during the feature selection process and logs the classification performance metrics to a log file.

## Module Introduction

### Data_Processing

This module is primarily used for preprocessing raw protein data (PDB and TXT), and includes the following scripts:

- **draw1.py**: Extract PDB files containing "model_1" from the raw folder.
- **draw2.py**: Find the maximum number of lines in all PDB files to determine the padding length.
- **draw3.py**: Determine the data range for each column and perform mapping encoding.
- **draw4.py**: Extract column data from PDB files, perform padding and concatenation, and generate a CSV file.
- **draw5.py**: Process protein sequence data to obtain BINA encoding, mapping a 21-dimensional vector to a 441-dimensional vector, and generate 2D data.
- **draw6.py**: Concatenate two CSV files (3D and BINA) to obtain 3D+BINA data.

### Datasets/Protein_sequence

This module is used to store the raw dataset of protein sequences.

### Feature Selection and Classification

- **feature_selection**：Stored 54 feature selection algorithms.
- **losses**：Defined a class named jFitnessFunction, which serves as the fitness function primarily used to evaluate the performance of feature selection algorithms.
- **Grid_search_Classifier**：Used for grid search and model training on different classifiers, including KNN, SVM, LR, RF, GB, ERT, XGB, AB, etc.
- **main**:Used for feature selection and classification.


