import pandas as pd
import numpy as np
import yfinance as yf

# Fetch DJIA data
df = yf.download('^DJI', start='2020-01-02', end='2020-12-15')

# Calculate daily returns
df['daily_return'] = df['Adj Close'].pct_change().fillna(0)

# Calculate total assets assuming starting with an initial amount
initial_amount = 100000
df['total assets'] = initial_amount * (1 + df['daily_return']).cumprod()

def evaluate(df):
    start_date = df.index[0]
    end_date = df.index[-1]
    daily_return = df["daily_return"]
    neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
    tr = df["total assets"].values[-1] / df["total assets"].values[0] - 1
    return_rate_list = df["daily_return"].tolist()

    sharpe_ratio = np.mean(return_rate_list) * (252 ** 0.5) / np.std(return_rate_list)
    vol = np.std(return_rate_list)
    mdd = 0
    peak = df["total assets"][0]
    for value in df["total assets"]:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > mdd:
            mdd = dd
    cr = np.sum(daily_return) / mdd
    sor = np.sum(daily_return) / np.std(neg_ret_lst) / np.sqrt(len(daily_return))
    return start_date, end_date, tr, sharpe_ratio, vol, mdd, cr, sor

# Perform analysis
start_date, end_date, tr, sharpe_ratio, vol, mdd, cr, sor = evaluate(df)

# Print results
print(f"Start Date: {start_date}")
print(f"End Date: {end_date}")
print(f"Total Return: {tr:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Volatility: {vol:.4f}")
print(f"Max Drawdown: {mdd:.4f}")
print(f"Calmar Ratio: {cr:.4f}")
print(f"Sortino Ratio: {sor:.4f}")
