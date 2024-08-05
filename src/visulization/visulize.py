import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score


# Plot Time series
def plot_timeseries(df,forecast):
    
    # creating a new Dataframe dp with the prediction values.
    ypred = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    Date = pd.Series(['2024-01-01', '2024-02-01'])
    price_actual = pd.Series(['184.40','185.04'])
    price_predicted = pd.Series(ypred.values)
    lower_int = pd.Series(conf_int['lower AAPL'].values)
    upper_int = upper_series = pd.Series(conf_int['upper AAPL'].values)

    dp = pd.DataFrame([Date, price_actual, lower_int, price_predicted, upper_int], index =['Date','price_actual', 'lower_int', 'price_predicted', 'upper_int']).T
    dp = dp.set_index('Date')
    dp.index = pd.to_datetime(dp.index)
    # Plot
    plt.plot(df.AAPL)
    plt.plot(dp.price_predicted, color='orange')
    plt.fill_between(dp.index,
                    lower_int,
                    upper_int,
                    color='k', alpha=.15)


    plt.title('Model Performance')
    plt.legend(['Actual','Prediction'], loc='lower right')
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Price (USD)')
    plt.show()
    return

# Plot cluster method scatter chart
def plot_scatter(df):
    sns.scatterplot(x='Annual_Income', y = 'Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.show()

# Plot elbow chart
def elbow(df, start_range, end_range, *args):
    # try using a for loop
    k = range(start_range,end_range)
    K = []
    WCSS = []
    column = []
    for items in args:
        column.append(items)
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df[column])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)
    # Store the number of clusters and their respective WSS scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
    # Now, plot a Elbow plot
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()

def silhouette(df, start_range, end_range, *args):
    # same as above, calculate sihouetter score for each cluster using a for loop
    k = range(start_range,end_range)
    K = []
    ss = []
    column = []
    for items in args:
        column.append(items)
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[column])
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[column], ypred)
        K.append(i)
        ss.append(sil_score)
    # Store the number of clusters and their respective silhouette scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':ss})
    # Now, plot a Elbow plot
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()
    
def loss_curve(df):
    loss_values = df.loss_curve_

    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()