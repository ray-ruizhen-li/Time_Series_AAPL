from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels
from statsmodels.tsa.arima.model import ARIMA


def time_series(df):
    
    decomposed = seasonal_decompose(df['AAPL'])
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    residual = decomposed.resid
    results = adfuller(df['AAPL'])
    print('p-value:', results[1]) # adf, pvalue, usedlag_, nobs_, critical_values_, icbest_
    # 1st order differencing
    v1 = df['AAPL'].diff().dropna()

    # adf test on the new series. if p value < 0.05 the series is stationary
    results1 = adfuller(v1)
    
    # 1,1,1 ARIMA Model
    arima = ARIMA(df.AAPL, order=(1,1,1))
    ar_model = arima.fit()
    print(ar_model.summary())
    
    # Forecast
    forecast = ar_model.get_forecast(2)
    
    
    return forecast

def deep_learning(X,y):
    #Scaling the data
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    
    #splitting the data
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=1,stratify=y)

    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)

    # 1. Create the model using the Sequential API
    model_1 = tf.keras.Sequential([tf.keras.layers.Dense(1) #output layer
                                   ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(), # binary since we are working with 2 clases (0 & 1)
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['accuracy'])

    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=50,verbose=0)
    score = model_1.evaluate(x_train, y_train)
    return score

def deep_learning_layers(X,y):
    # add an extra layer.

    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)
    
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    
    #splitting the data
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=1,stratify=y)

    # set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1), # add an extra layer
    tf.keras.layers.Dense(1) # output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                    metrics=['accuracy'])

    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=50,verbose=0)
    model_1.evaluate(x_train, y_train)
    score = model_1.evaluate(x_train, y_train)
    return score

def find_best_deep_learning_rate(X,y):
    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)
    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)
    
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    
    #splitting the data
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=1,stratify=y)

    # set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1), # add an extra layer
    tf.keras.layers.Dense(1) # output layer
    ])

    # Compile the model
    model_1.compile(loss="binary_crossentropy", # we can use strings here too
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["accuracy"])

    # Create a learning rate scheduler callback
    # traverse a set of learning rate values starting from 1e-3, increasing by 10**(epoch/20) every epoch
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * 0.9**(epoch/3)
    )


    # Fit the model (passing the lr_scheduler callback)
    history = model_1.fit(x_train,
                        y_train,
                        epochs=100,
                        verbose=0,
                        callbacks=[lr_scheduler])
    # Plot the learning rate versus the loss
    lrs = 1e-5 * (10 ** (np.arange(100)/20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs. loss");
    plt.show()
    
def neural_networks(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

   # fit/train the model. Check batch size.
    MLP = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=100, random_state=123)
    MLP.fit(X_train,y_train)
    
    return MLP,X_test_scaled, y_test

# Function to train the model
def train_logistic_regression(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression().fit(X_train_scaled, y_train)
    
    # Save the trained model
    with open('./src/models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, X_test_scaled, y_test

def random_forest(X,y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the random forest model
    rfmodel = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features=None)
    rfmodel.fit(X_train, y_train)
    ypred = rfmodel.predict(X_test)
    return rfmodel, X_test_scaled, y_test

def decision_tree(X,y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
     # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # create an instance of the class
    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=457)
    # train the model
    dtmodel = dt.fit(X_train,y_train)
    # make predictions using the test set
    ytest_pred = dtmodel.predict(X_test)
    # evaluate the model
    test_mae = mean_absolute_error(ytest_pred, y_test)
    print("Decision Tree test error is: ",test_mae)
    # make predictions on train set
    ytrain_pred = dtmodel.predict(X_train)
    # evaluate the model
    train_mae = mean_absolute_error(ytrain_pred, y_train)
    print("Decision Tree Train error is: ",train_mae)
    return dtmodel, X_test_scaled, y_test