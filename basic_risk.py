import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tickers = ['SPY', 'TLT', 'GLD']
data = yf.download(tickers, period = '5y')['Close']
returns = data.pct_change().dropna()
rolling_corr = returns['SPY'].rolling(window=30).corr(returns['TLT'])
plt.figure(figsize=(10, 6))
rolling_corr.plot()
plt.axhline(0, color='red', linestyle='--')
plt.title('Rolling 30-Day Correlation: SPY vs TLT')
plt.ylabel('Correlation Coefficient')
plt.show()
