# Predictive App

An interactive streamlit app that reads CSV dataset and displays exploratory data analysis and uses it to build predictive model using several machine learning algorithms.

## Features

-Displays EDA on numeric and categoric features by displaying histplots and countplots
-Displays correlation matrix on numeric features
-Displays feature and target plot
-Displays feature importance visualization for tree based models
-Predictive model is built using linear regression, logistic regression, Random Forest, Gradient Boosting
-User interface is built using Streamlit


## How to run

Create a virtual enviroment using this command
'''
python3 venv venv
'''
Run the virtual environment
'''
source venv/bin/activate
''''
Install the requirements
'''
pip install -r requirements.txt
'''
Train the model
'''python3 train.py --csv csv --target target --problem problem --algorithm algorithm --out out
'''
csv is the name of the csv file
target is the target
problemn is either classification or regression
algorithm can be RandomForest, 
out is the name of the model 

Run the app
```
streamlit run app.py
```

