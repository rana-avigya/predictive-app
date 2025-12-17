# Predictive App

An interactive streamlit app that reads CSV dataset and displays exploratory data analysis and uses it to build predictive model.

## Features

-Displays EDA on numeric and categorical features by displaying histplots and countplots
-Displays correlation matrix on numeric features
-Displays feature and target plot
-Predictive model is trained by passing arguments on terminal
-Arguments are passed for the csv file, the name of target, problem type and the name of the model file
-User interface is built using Streamlit


## How to run

Download the dataset from this link: https://www.kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset

Save the csv file in project directory

Create a virtual environment using this command
```
python3 venv venv
```
Run the virtual environment
```
source venv/bin/activate
```
Install the requirements
```
pip install -r requirements.txt
```
Train the model
```
python3 train.py --target target --problem problem
```

Run the app
```
streamlit run app.py
```

