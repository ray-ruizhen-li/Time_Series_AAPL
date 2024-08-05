ARIMA - Auto Regressive Integrated Moving Average

An ARIMA model is characterized by 3 terms: p, d, q
<br><br>
* **AR (p) Autoregression:** An auto regressive AR(p) term refers to the number of lags of Y to be used as predictors. In other words, the number of terms that are used from the past values in the time series for forecasting the future values. It's the part of the model that captures the relationship between a current observation and its predecessors. A high *p* value can indicate a strong influence of past values on current values.
<br><br>
* **I (d) Integration:** This parameter indicates the number of times the original time series data should be differenced to make it stationary. Stationarity is a critical aspect of time series analysis where the statistical properties of the series (mean, variance) do not change over time. Differencing is a method of transforming a non-stationary time series into a stationary one by subtracting the previous observation from the current observation. The *d* value helps in removing trends and seasonal structures from the time series, making it easier to model.
<br><br>
* **MA (q) Moving Average:**  ‘q’ is the order of the ‘Moving Average’ (MA) term. 'q' refers to the number of past errors (residuals) that are used to improve forecasts. The MA component models the relationship between an observation and a residual error from a moving average model applied to lagged observations. In simpler terms, it accounts for the influence of past prediction errors on the current observation. The errors referred to here are the differences between the actual values and the predictions that a simple moving average model would have made. The choice of *q* has a direct impact on how the ARIMA model predicts future values by considering how past errors influence current predictions. It helps in capturing the autocorrelation in the residuals of the series that is not explained by the autoregressive (AR) part.
<br><br>

### Model Types:

**ARIMA** - Non-seasonal auto regressive integrated moving average

<span style="color:crimson">**ARIMAX** - ARIMA with exogenous variable</span>

**SARIMA** - Seasonal ARIMA

**SARIMAX** - Seasonal ARIMA with exogenous variable