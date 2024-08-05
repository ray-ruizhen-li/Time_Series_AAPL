# Created by Rayli
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#to scale the data using z-score
from sklearn.preprocessing import StandardScaler

#to split the dataset
from sklearn.model_selection import train_test_split

#Metrics to evaluate the model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#to ignore warnings
import warnings
warnings.filterwarnings("ignore")
from src.data.make_dataset import load_and_preprocess_data
from src.feature_engineering.build_features import create_dummy_vars
from src.models.train_models import time_series
from src.visulization.visulize import plot_timeseries




#from src.models.predict_model import NN_evaluate_model
#from src.visulization.visulize import loss_curve

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "src/data/raw/AAPL.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    data = create_dummy_vars(df)

    # Train the logistic regression model
    forecast = time_series(data)

    # Plot 
    plot_timeseries(data,forecast)

   