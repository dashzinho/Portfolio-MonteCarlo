import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

########
# DATA #
########

# Tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Timeframe
start = '2010-01-01'
end = '2020-01-01'

# Download and filter data
data = yf.download(tickers, start=start, end=end)['Adj Close']
os.makedirs('data', exist_ok=True)
data.to_csv('data/stock_prices.csv')
data = pd.read_csv('data/stock_prices.csv', index_col='Date', parse_dates=True)
data = data.dropna()
returns = data.pct_change().dropna()  # Calculate daily returns
returns.to_csv('data/daily_returns.csv')

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

##############
# SIMULATION #
##############

# Parameters
num_simulations = 10000
time_horizon = 252  # 1 year of trading days
initial_portfolio_value = 1000000  # $1,000,000

# Simulate portfolio values
portfolio_simulations = np.zeros((num_simulations, time_horizon))

for i in range(num_simulations):
    # Generate random returns
    random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
    # Calculate portfolio value over time
    portfolio_value = initial_portfolio_value * np.cumprod(1 + random_returns @ np.ones(len(tickers)) / len(tickers))
    portfolio_simulations[i, :] = portfolio_value

# Calculate final portfolio values
final_portfolio_values = portfolio_simulations[:, -1]

# Calculate VaR and ES
VaR_95 = np.percentile(final_portfolio_values, 5)
ES_95 = final_portfolio_values[final_portfolio_values <= VaR_95].mean()

# Print the results
print(f"Value at Risk (VaR) 95%: ${initial_portfolio_value - VaR_95:,.2f}")
print(f"Expected Shortfall (ES) 95%: ${initial_portfolio_value - ES_95:,.2f}")

# Plot
plt.figure(figsize=(10, 6))
plt.hist(final_portfolio_values, bins=50, alpha=0.7, label='Final Portfolio Values')
plt.axvline(x=VaR_95, color='r', linestyle='--', label=f'VaR 95%: ${VaR_95:,.2f}')
plt.axvline(x=ES_95, color='g', linestyle='--', label=f'CVaR 95%: ${ES_95:,.2f}')
plt.legend()
plt.title('Monte Carlo Simulation Results')
plt.xlabel('Portfolio Value ($)')
plt.ylabel('Frequency')
plt.show()
