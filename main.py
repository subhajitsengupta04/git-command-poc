import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set visual style for reports
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

# Synthetic Data Generation
def generate_full_dataset():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1100, freq='D')
    time = np.arange(len(dates))
    
    # Target Variable: Revenue
    trend = 0.5 * time + 1000
    seasonal = 150 * np.sin(2 * np.pi * time / 365.25)
    multi_factor = 1 + (time / 1500)
    revenue = (trend + seasonal) * multi_factor + np.random.normal(0, 50, len(dates))
    
    # Auxiliary Variables (Correlated with Revenue)
    marketing = 200 + (0.1 * revenue) + np.random.normal(0, 30, len(dates))
    customers = (revenue / 10).astype(int) + np.random.randint(-20, 20, len(dates))
    transactions = (customers * 0.4).astype(int) + np.random.randint(-5, 5, len(dates))

    df = pd.DataFrame({
        'Date': dates,
        'Revenue': revenue,
        'Transactions': transactions,
        'Marketing_Spend': marketing,
        'Customer_Count': customers
    })
    
    # Inject missing values
    for col in ['Revenue', 'Marketing_Spend']:
        df.loc[np.random.choice(df.index, 15), col] = np.nan
        
    df.to_excel("Data/master-data.xlsx", index=False)
    return df

df = generate_full_dataset()
df.set_index('Date', inplace=True)

# Impute Missing Values
# Using linear interpolation for time-series consistency
df = df.interpolate(method='linear')
print("Missing values handled. Dataset shape:", df.shape)

# Multivariate EDA
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
plt.title('Correlation Heatmap: Financial Feature Relationships')
plt.savefig("Visuals/correlation-heatmap.png")
plt.show()

# Funnel Test to determine model type)
roll_mean = df['Revenue'].rolling(window=30).mean()
roll_std = df['Revenue'].rolling(window=30).std()

plt.figure(figsize=(10, 5))
plt.scatter(roll_mean, roll_std, alpha=0.4, color='darkblue')
plt.title('Funnel Test: Mean vs Standard Deviation')
plt.xlabel('30-Day Moving Average Revenue')
plt.ylabel('30-Day Moving Std Dev')
plt.savefig("Visuals/funnel-test.png")
plt.show()

# Component Decomposition
decomp = seasonal_decompose(df['Revenue'], model='multiplicative', period=365)
fig = decomp.plot()
fig.set_size_inches(12, 8)
plt.savefig("Visuals/component-decomposition.png")
plt.show()

# Stationarity Testing (ADF & KPSS)
def stationarity_report(series, name):
    print(f"\nStats Report for {name}:")
    p_adf = adfuller(series.dropna())[1]
    p_kpss = kpss(series.dropna(), regression='c', nlags='auto')[1]
    print(f"  ADF (Null: Non-Stationary) -> p={p_adf:.4f}")
    print(f"  KPSS (Null: Stationary)    -> p={p_kpss:.4f}")

stationarity_report(df['Revenue'], "Raw Revenue")

# Transformation: Applying Log to stabilize variance and Diff to remove trend
df['Revenue_Stationary'] = np.log(df['Revenue']).diff().dropna()
stationarity_report(df['Revenue_Stationary'], "Log + Differenced Revenue")

# Model Identification (ACF/PACF)
# Spikes in PACF help determine 'p', spikes in ACF help determine 'q'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['Revenue_Stationary'].dropna(), lags=40, ax=ax1)
plot_pacf(df['Revenue_Stationary'].dropna(), lags=40, ax=ax2)
plt.savefig("Visuals/acf-pacf.png")
plt.show()

# SARIMA Modeling and Forecasting
train_size = int(len(df) * 0.9)
train, test = df['Revenue'][:train_size], df['Revenue'][train_size:]

# Fitting the model
# (p,d,q) = (1,1,1) for trend | (P,D,Q,s) = (1,1,1,7) for weekly seasonality
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
fit_results = model.fit(disp=False)

# Diagnostics
# fit_results.plot_diagnostics(figsize=(12, 9))
# plt.show()

# Forecast
forecast_steps = len(test)
pred_obj = fit_results.get_forecast(steps=forecast_steps)
pred_df = pred_obj.summary_frame()

# Accuracy
mape = np.mean(np.abs((test - pred_df['mean']) / test)) * 100
print(f"Final Model MAPE: {mape:.2f}%")

plt.figure(figsize=(14, 7))
plt.plot(df.index[-200:], df['Revenue'][-200:], label='Actual Historical', color='gray', alpha=0.5)
plt.plot(test.index, test, label='Actual Test Data', color='blue')
plt.plot(test.index, pred_df['mean'], label='SARIMA Forecast', color='red', linestyle='--')
plt.fill_between(test.index, pred_df['mean_ci_lower'], pred_df['mean_ci_upper'], color='pink', alpha=0.2)
plt.title('Financial Revenue Forecasting', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.savefig("Visuals/final-forecast.png")
plt.show()