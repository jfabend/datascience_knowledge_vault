Seasonal Autoregressive Integrated Moving Average

Wie [[ARIMA]], but we add the final component: **seasonality S(P, D, Q, s)**, where **s** is simply the season’s length. Furthermore, this component requires the parameters **P** and **Q** which are the same as **p** and **q**, but for the seasonal component. Finally, **D** is the order of seasonal integration representing the number of differences required to remove seasonality from the series

before modelling with SARIMA, we must apply transformations to our time series to remove seasonality and any non-stationary behaviors

