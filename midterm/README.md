# ML Zoomcamp Midterm Project - November 2021

## Problem Statement
The Great Resignation, A Classification Problem

With all of the talk of the Great Resignation in the news these days due to COVID-19 and general job dissatisfaction, I decided to examine whether a person quits their job or not.

![I Quit](images/i-quit.jpeg)

## Data
The dataset has 14,999 records with 9 independent variables and our dependent variable, whether or not someone quit. It can be found at [Original Dataset](https://github.com/VincentTatan/PythonAnalytics/blob/master/Youtube/dataset/HR_comma_sep.csv)

*    employee satisfaction level
*    last evaluation
*    number of projects
*    average montly hours
*    time spent at the company
*    if there was a work accident
*    was there promotion within the last 5 years
*    job category
*    salary (grouped by low, medium, or high)
*    whether or not the person quit their job


## Classification Models Used
Given the binary nature of someone either quitting or not, we will use the following classification models to test our theory of whether or not someone will quit, given the above fields: 

*   Naive Bayes
*   Logistic Regression
*   Decision Trees
*   Random Forest
*   SVM
*   XGBoost

## Model Evaluation
Also given the binary nature and the tendency for the majority of the samples to be imbalanced towards people not quitting, we will use Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores, rather than accuracy as our evaluation metric.

## Project Structure

 * README.md with
      * Description of the problem
      * Instructions on how to run the project
 * Notebook (notebook.ipynb) with
      * Data preparation and data clearning
      * EDA, feature importance analysis
      * Model selection process and parameter tuning
 * Script train_export_model.py
      * Training the final model
      * Saving it to a file (e.g. pickle)
 * Script predict_quitting_serving.py
      * Loading the model
      * Serving it via a web serice (e.g. with Flask)
 * Pipenv and Pipenv.lock (Pipfile and Pipfile.lock)
      * or equivalents: conda environment file, requirements.txt or pyproject.toml
 * Dockerfile for running the service


## How to Run the Project
* The notebook.ipynb has the data preparation, exploratory analysis and model selection process. Run as you would a normal jupyter notebook.
* The notebook was exported to a script called train_export_model.py
* A standalone script to test the model without a web service:  python3 test_quitting.py
* To run the script with a webservice: python3 test_quitting_localhost.py
