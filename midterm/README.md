# ML Zoomcamp Midterm Project - November 2021

## Problem Statement
The Great Resignation, A Classification Problem

With all of the talk of the Great Resignation in the news these days due to COVID-19 and general job dissatisfaction, I decided to examine whether a person quits their job or not. With employees leaving, it causes hiring and staffing challenges for managers, the work load for existing employees, and for human resources.

## Data
The dataset has 14,999 records with 9 independent variables and our dependent variable, whether or not someone quit.  It can be found at [Original Dataset](https://github.com/VincentTatan/PythonAnalytics/blob/master/Youtube/dataset/HR_comma_sep.csv)

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
 * Script train.py
      * Training the final model
      * Saving it to a file (e.g. pickle)
 * Script predict.py
      * Loading the model
      * Serving it via a web serice (e.g. with Flask)
 * Pipfile and Pipfile.lock
      * or equivalents: conda environment file, requirements.txt or pyproject.toml
 * Dockerfile for running the service


## How I Built the Pipenv on a Mac running BigSur - _steps 2,3,5 are only needed on a Mac_
1. Created a directory called ml-zoomcamp-midterm
2. source ~/.bash_profile  
3. export PATH = "$PATH:/Users/lkirch/.local/bin"
4. conda activate ml-zoomcamp
5. export SYSTEM_VERSION_COMPAT=1  
6. pip install --user pipenv
7. pipenv install requests
8. pipenv install scikit-learn==1.0
9. pipenv install numpy
10. pipenv install flask
11. pipenv install gunicorn
12. pipenv shell

The python code is then run in pipenv.  To start up/activate your pipenv:
```
   pipenv shell
```

## How to Run the Project

* If you just want to see the data wrangling, analysis and model selection process, run **notebook.ipynb** as you would a normal jupyter notebook.

### To export the model and the DictVectorizer to a pickle file called rf_model.bin:
```
   python3 train.py
```

### For testing, a standalone script called test_quitting.py will test the model without a web service:  
```
   python3 test_quitting.py
```

### To test the model with the flask/gunicorn webservice:
```
   python3 predict.py 
   * and while predict.py is running, you can run the notebook **will_this_employee_quit.ipynb**
   pipenv run gunicorn --bind 0.0.0.0:9696 predict:app (_in one terminal window_)
   python3 predict_test.py (_in another terminal window_)
   CTRL+C or CRTL+D to quit (_there might be a more graceful way, please let me know if you know_)
```
   
### To build the docker image in Google Cloud Shell:
   * Follow Ninad Date's very clear instructions found https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-google-cloud-shell-for-docker.md - _thank you Ninad!_
   * I uploaded Pipfile, Pipfile.lock, Dockerfile, rf_model.bin, predict.py request.py
   
   
### To test it in Google Cloud Shell
```
   docker run --rm -d -p 8080:9696 lk-ml-zoomcamp
   docker ps -a (_to be sure it is running_)
   python3 request.py
```

## Enhancements/Next Steps

- [ ] Gather more recent data and perhaps more data.  There may be additional features that affect whether someone quits since the data was collected.  I believe the data was from only one company, so gathering data from multiple companies would also be interesting.
- [ ] Additional graphs comparing the feature importance by type of model.
- [ ] Try additional models and model tuning to see if that improves the final model.
- [ ] Try some sampling methods to see if that improves the final model.
- [ ] More automated modeling experiments, such as GridSearchCV.
- [ ] More test cases and error handling if someone submits the employee request incorrectly.  Currently, it just fails.
- [ ] A more elaborate test web page, so you know heroku is working.