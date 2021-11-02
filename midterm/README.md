# ML Zoomcamp Midterm Project - November 2021

## Problem Statement
The Great Resignation, A Classification Problem

With all of the talk of the Great Resignation in the news these days due to COVID-19 and general job dissatisfaction, I decided to examine whether a person quits their job or not. With employees leaving, it causes hiring and staffing challenges for managers, the work load for existing employees, and for human resources.

![I Quit](https://github.com/lkirch/ml-zoomcamp/blob/main/midterm/images/i-quit.jpeg)

## Data
The dataset has 14,999 records with 9 independent variables and our dependent variable, whether or not someone quit.  Here is the [Original Dataset](https://github.com/VincentTatan/PythonAnalytics/blob/master/Youtube/dataset/HR_comma_sep.csv)

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

*   [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
*   [Decision Trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
*   [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
*   [SVM](https://scikit-learn.org/stable/modules/svm.html)
*   [XGBoost](https://xgboost.readthedocs.io/en/latest/python/index.html)

## Model Evaluation
Also given the binary nature and the tendency for the majority of the samples to be imbalanced towards people not quitting, we will use [Area Under the Receiver Operating Characteristic Curve (ROC AUC)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) from prediction scores, rather than accuracy as our evaluation metric.

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
      * Saving it to a pickle file
 * Pickle file rf_model.bin which contains
      * the model
      * the DictVectorizer to prepare data
 * Script predict.py
      * Loading the model
      * Serving it via a Flask web serice
 * Notebook (will_this_employee_quit.ipynb) to test the Flask web service
 * Script predict_test.py to test the Flask web service
 * Pipfile and Pipfile.lock for pipenv container
 * Dockerfile for running the service in a docker container
 * Script request.py to test the docker container in Google Cloud Shell
 


## How I Built the Pipenv on a Mac running BigSur
_Note: steps 2,3, and 5 are only needed on a Mac_
```
   mkdir ml-zoomcamp-midterm
   source ~/.bash_profile  
   export PATH = "$PATH:/Users/lkirch/.local/bin"
   conda activate ml-zoomcamp
   export SYSTEM_VERSION_COMPAT=1  
   pip install --user pipenv
   pipenv install requests
   pipenv install scikit-learn==1.0
   pipenv install numpy
   pipenv install flask
   pipenv install gunicorn
   pipenv shell
```

The python code is then run in pipenv.  To start up/activate your pipenv:
```
   pipenv shell
```

## How to Run the Project

If you just want to see the data wrangling, analysis and model selection process, run **notebook.ipynb** as you would a normal jupyter notebook.

#### To export the model and the DictVectorizer to a pickle file called rf_model.bin:
```
   python3 train.py
```

#### For testing, a standalone script called test_quitting.py will test the model without a web service:  
```
   python3 test_quitting.py
```

#### To test the model with the flask/gunicorn webservice:
You can run the notebook **will_this_employee_quit.ipynb**. while predict.py is running:
```
   python3 predict.py
```

Or you can run a python script:
   * in one terminal window:
```
   pipenv run gunicorn --bind 0.0.0.0:9696 predict:app 
```
   * in another terminal window:
```   
   python3 predict_test.py 
```

#### To build the docker image in Google Cloud Shell (_you must have a gmail account_):
   * Follow [Ninad Date's very clear instructions](https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-google-cloud-shell-for-docker.md) - _thank you, Ninad!_
   * I uploaded Pipfile, Pipfile.lock, Dockerfile, rf_model.bin, predict.py request.py.
   * created a directory called app-deploy and moved the above files into it.
   * build the docker image
```
   docker build -t ml-zoomcamp-midterm .
```   
   
   
#### To test the web service in Google Cloud Shell:
```
   docker run --rm -d -p 8080:9696 lk-ml-zoomcamp
   docker ps -a
   python3 request.py
```

#### To deploy from Google Cloud Shell to heroku:
  * Again follow [Ninad Date's very clear instructions](https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-heroku.md) - _thank you again, Ninad!_
  * Remove "--bind=0.0.0.0:9696" from the ENTRYPOINT in your Dockerfile
  * Install the heroku cli in google cloud shell: 
```
   curl https://cli-assets.heroku.com/install-ubuntu.sh | sh  
```
  * In Heroku: Account Settings > Applications > Create Authorization > type in heroku-api-gcp and copy the key you receive.
  * In Google Cloud Shell:
```
   export HEROKU_API_KEY="put_your_key_here"
   heroku container:login
   heroku container:push web -a ml-lk-zoo-docker
   heroku container:release web -a ml-lk-zoo-docker
```
  * To see which apps are running:
```
   heroku apps
```
  * To test to see if the web service is running at heroku go to https://ml-lk-zoom-docker.herokuapp.com/test
  * To try test from a python script:
```
   python3 predict_test_heroku.py
```

## Enhancements/Next Steps

- [ ] Gather more recent data and perhaps more data.  There may be additional features that affect whether someone quits since the data was collected.  I believe the data was from only one company, so gathering data from multiple companies would also be interesting.
- [ ] Additional graphs comparing the feature importance by type of model.
- [ ] Try additional models and model tuning to see if that improves the final model.
- [ ] Try some sampling methods to see if that improves the final model.
- [ ] More automated modeling experiments, such as GridSearchCV.
- [ ] More test cases and error handling if someone submits the employee request incorrectly.  Currently, it just fails.
- [ ] A more elaborate test web page, so you know heroku is working.