import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # We'll need this later for precise optimization

# --- Configuration ---
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'GS', 'XOM']
start_date = '2018-01-01'
end_date = '2023-12-31'
num_trading_days = 252 # Used for annualizing

# --- 1. Fetch Historical Stock Data ---
print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}...")
try:
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    print("Data fetched successfully!")
except Exception as e:
    print(f"Error fetching data: {e}")
    print("Please check tickers and date range, or your internet connection.")
    exit()

data.dropna(inplace=True)
if data.empty:
    print("No valid data after dropping NaNs. Exiting.")
    exit()

# --- 2. Calculate Daily and Annualized Returns ---
log_returns = np.log(data / data.shift(1)).dropna()
annual_returns = log_returns.mean() * num_trading_days

# --- 3. Calculate Daily and Annualized Volatility (Standard Deviation) ---
daily_volatility = log_returns.std()
annual_volatility = daily_volatility * np.sqrt(num_trading_days)

# --- 4. Calculate Covariance and Correlation Matrix ---
cov_matrix_annual = log_returns.cov() * num_trading_days

print("\n--- Initial Data Processing Complete ---")
print("Proceeding to Monte Carlo Simulation...")

# --- 5. Monte Carlo Simulation for Random Portfolios ---
num_portfolios = 20000 # Number of random portfolios to generate
num_assets = len(tickers)

# Arrays to store results
portfolio_returns = []
portfolio_volatility = []
portfolio_weights = []

# Risk-free rate (can be adjusted) - e.g., current T-bill rate
risk_free_rate = 0.02 # 2%

for portfolio in range(num_portfolios):
    # Generate random weights for assets
    weights = np.random.random(num_assets)
    weights /= np.sum(weights) # Normalize weights to sum to 1
    portfolio_weights.append(weights)

    # Calculate portfolio return
    # R_p = sum(w_i * R_i)
    returns = np.sum(weights * annual_returns)
    portfolio_returns.append(returns)

    # Calculate portfolio volatility (standard deviation)
    # sigma_p = sqrt(w' * Cov_matrix * w)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
    portfolio_volatility.append(volatility)

# Store results in a DataFrame for easier analysis
portfolio_results_df = pd.DataFrame({
    'Return': portfolio_returns,
    'Volatility': portfolio_volatility,
    'Sharpe Ratio': (np.array(portfolio_returns) - risk_free_rate) / np.array(portfolio_volatility)
})

# Add individual stock returns and volatility for plotting reference
for i, ticker in enumerate(tickers):
    portfolio_results_df[f'Weight_{ticker}'] = [w[i] for w in portfolio_weights]

print(f"\nGenerated {num_portfolios} random portfolios.")
print("Portfolio results DataFrame head:")
print(portfolio_results_df.head())

# --- 6. Plotting the Efficient Frontier with Monte Carlo results ---
plt.figure(figsize=(12, 8))
plt.scatter(portfolio_results_df['Volatility'], portfolio_results_df['Return'],
            c=portfolio_results_df['Sharpe Ratio'], cmap='viridis', s=10, alpha=0.6)
plt.colorbar(label='Sharpe Ratio')

# Plot individual stocks
plt.scatter(annual_volatility, annual_returns, s=50, edgecolors='black',
            color='red', marker='o', label='Individual Stocks')
for i, txt in enumerate(tickers):
    plt.annotate(txt, (annual_volatility[i], annual_returns[i]), xytext=(5, -5),
                 textcoords='offset points', fontsize=9)

# Identify the portfolio with the maximum Sharpe Ratio
max_sharpe_portfolio = portfolio_results_df.loc[portfolio_results_df['Sharpe Ratio'].idxmax()]
plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'],
            marker='*', color='green', s=300, label='Max Sharpe Ratio Portfolio (MC)')
plt.annotate('Max Sharpe Ratio Portfolio',
             (max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return']),
             xytext=(max_sharpe_portfolio['Volatility'] + 0.01, max_sharpe_portfolio['Return'] - 0.01),
             textcoords='data', fontsize=10, color='green',
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))


# Identify the portfolio with the minimum volatility
min_volatility_portfolio = portfolio_results_df.loc[portfolio_results_df['Volatility'].idxmin()]
plt.scatter(min_volatility_portfolio['Volatility'], min_volatility_portfolio['Return'],
            marker='X', color='red', s=300, label='Min Volatility Portfolio (MC)')
plt.annotate('Min Volatility Portfolio',
             (min_volatility_portfolio['Volatility'], min_volatility_portfolio['Return']),
             xytext=(min_volatility_portfolio['Volatility'] + 0.01, min_volatility_portfolio['Return'] + 0.01),
             textcoords='data', fontsize=10, color='red',
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))


plt.title('Efficient Frontier with Monte Carlo Simulation')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Return')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(labelspacing=0.8)
plt.show()

print("\n--- Monte Carlo Simulation and Plotting Complete ---")
print("\nMax Sharpe Ratio Portfolio (from Monte Carlo):")
print(max_sharpe_portfolio)
print("\nMin Volatility Portfolio (from Monte Carlo):")
print(min_volatility_portfolio)