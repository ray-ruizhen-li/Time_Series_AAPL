import pandas as pd

# create dummy features
def create_dummy_vars(data):
    #Separating target variable and other variables
    data['Date'] = pd.to_datetime(data['Date'])
    df = data.iloc[:-2,0:2]
    df = df.set_index('Date')

    return df