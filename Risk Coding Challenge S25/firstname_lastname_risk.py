import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class Portfolio:
    """
        The portfolio class holds a dictionary of every stock in the list of tickers you are given. These are initialized to empty
        dataframes with every metric you need to calculate. You also have an attribute of cash which starts with an initial balance
        of 100,000. 
    """
    def __init__(self, tickers, start_date, end_date, initial_balance=100000):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        
        # Dictionary of empty DataFrames for each ticker. Key-value pairs of ticker-dataframe.
        # Each DataFrame is indexed by date and has columns for risk metrics and return. 
        self.ticker_holdings = {}
        for i in range(len(tickers)):
            newDataFrame = pd.DataFrame(columns=["Price", "Unit Cost", "Volume", "Total Value", "Max DD", "Volatility", "VaR", "Sharpe"], index=pd.Index([], name="Date"))
            newDataFrame.index.name = "Date"
            newDataFrame["Price"] = 0.0
            newDataFrame["Unit Cost"] = 0.0
            newDataFrame["Volume"] = 0.0
            newDataFrame["Total Value"] = 0.0
            newDataFrame["Max DD"] = 0.0
            newDataFrame["Volatility"] = 0.0
            newDataFrame["VaR"] = 0.0
            newDataFrame["Sharpe"] = 0.0

            self.ticker_holdings[tickers[i]] = newDataFrame

        self.cash = initial_balance
        self.cash_history = pd.DataFrame(columns=["Date", "Cash"], index=pd.Index([], name="Date"))
         
        # Create initial weights of equal value for each stock and cash
        self.weights = {stock: 1/(len(tickers) + 1) for stock in ["CASH"] + tickers }

        # Create a dataframe to track the market prices of the S&P 500
        self.market_prices = None


    # TO-DO
    def get_ticker_data(self, filename):
        """ This function populates the price column of each dataframe in the self.ticker_holdings dictionary by extracting the 
        ticker information from stock_data.csv. """
        pass


    def buy(self, ticker, count, date):
        """
        This function accepts the ticker, number of shares, and the date the transaction will be made (pass 'day' from the iterating for loop to get the adj close for that specific day)
        The holdings are all updated automatically
        """
        # Gets the price of the ticker on the date
        price = self.ticker_holdings[ticker]["Price"].loc[date]

        # Finds the previous date
        prev_date = date
        row = 0
        if date != self.ticker_holdings[ticker]["Price"].index[0]:
            row = self.ticker_holdings[self.tickers[0]].index.get_loc(date)
            prev_date = self.ticker_holdings[self.tickers[0]].index[row-1]

        # Updates the portfolio holdings
        self.ticker_holdings[ticker]["Volume"].loc[date] = self.ticker_holdings[ticker]["Volume"].loc[prev_date] + count
        self.ticker_holdings[ticker]["Unit Cost"].iloc[row] =  ((self.ticker_holdings[ticker]["Unit Cost"].loc[prev_date] * self.ticker_holdings[ticker]["Volume"].loc[prev_date]) + (price * count)) / self.ticker_holdings[ticker]["Volume"].loc[date]
        self.ticker_holdings[ticker]["Total Value"].loc[date] = self.ticker_holdings[ticker]["Volume"].loc[date] * self.ticker_holdings[ticker]["Unit Cost"].loc[date]
        self.cash -= (price * count)
        

    def sell(self, ticker, count, date):
        """
        Similar function to buy(), however, processes a sell order
        """
        # Gets the price of the ticker on the date
        price = self.ticker_holdings[ticker]["Price"].loc[date]

        # Find the previous date
        prev_date = date
        if date != self.ticker_holdings[ticker]["Price"].index[0]:
            row = self.ticker_holdings[self.tickers[0]].index.get_loc(date)
            prev_date = self.ticker_holdings[self.tickers[0]].index[row-1]
        
        # Updates portfolio holdings
        self.ticker_holdings[ticker]["Unit Cost"].loc[date] = self.ticker_holdings[ticker]["Unit Cost"].loc[prev_date]
        self.ticker_holdings[ticker]["Volume"].loc[date] = self.ticker_holdings[ticker]["Volume"].loc[prev_date] - count
        self.ticker_holdings[ticker]["Total Value"].loc[date] = self.ticker_holdings[ticker]["Volume"].loc[date] * self.ticker_holdings[ticker]["Unit Cost"].loc[prev_date]
        self.cash += (price * count)

        

    def simulate(self):
        """
        This function process the time series data of adjusted closed prices via for loop to simulate trading days taking place. 
        Data for every stock is initially downloaded using yf.download(), and cleaned into a data frame self.prices for easier processing
        Examples to access specific data are provided
        """

        # Download data for all tickers into self.ticker_holdings
        self.get_ticker_data("stock_data.csv")

        # Download S&P 500 data to use as a benchmark
        sp500 = yf.download("^GSPC", start=self.start_date, end=self.end_date)["Adj Close"]
        self.market_prices = sp500

        # Buy initial holdings (equal value for each stock and cash)
        for stock in self.tickers:
            price_share = self.ticker_holdings[stock]["Price"].iloc[0]
            self.buy(stock, self.initial_balance/(len(self.tickers)+1) // price_share, self.ticker_holdings[stock]["Price"].index[0])

        # Calculate metrics for the initial holdings
        self.calculate_metrics()

        # For loop simulation 
        for day in self.ticker_holdings[self.tickers[0]].index:

            if day != self.ticker_holdings[self.tickers[0]].index[0]:

                # Iterate over each day in the data frame
                self.rebalance_portfolio(day)              # Re-balance portfolio daily
                self.cash_history.loc[day] = self.cash     # Record cash history

        self.display_data() # Display data at the end of the simulation


    # TO-DO
    def calculate_metrics(self):
        """Calculates risk metrics (Max Drawdown, Volatility, VaR, Sharpe Ratio)."""
        window = 30  # Rolling window of 30 days

        for ticker, df in self.ticker_holdings.items():
            df["Daily Return"] = df["Price"].pct_change()

            # Max Drawdown
            rolling_max = df["Price"].rolling(window, min_periods=1).max()
            drawdown = (df["Price"] - rolling_max) / rolling_max
            df["Max DD"] = drawdown.rolling(window, min_periods=1).min()

            # Volatility (Standard Deviation of returns)
            df["Volatility"] = df["Daily Return"].rolling(window).std()

            # Value-at-Risk (VaR - 95% confidence level)
            df["VaR"] = df["Daily Return"].rolling(window).quantile(0.05)

            # Sharpe Ratio (Risk-adjusted return)
            df["Sharpe"] = df["Daily Return"].rolling(window).mean() / df["Volatility"]


    # TO-DO
    def rebalance_portfolio(self, day, risk_metric='Sharpe', target_value=1.5):
        """Rebalances portfolio based on risk metric."""
        ranking = {
            ticker: self.ticker_holdings[ticker][risk_metric].loc[day]
            for ticker in self.tickers
        }
        sorted_tickers = sorted(ranking, key=ranking.get, reverse=True)

        for ticker in sorted_tickers:
            if ranking[ticker] < target_value:
                self.sell(ticker, self.ticker_holdings[ticker]["Volume"].loc[day] // 2, day)


    # TO-DO
    def display_data(self):
        """Displays portfolio performance against the S&P 500 benchmark."""
        portfolio_values = {ticker: df["Total Value"] for ticker, df in self.ticker_holdings.items()}
        portfolio_values["Cash"] = self.cash_history["Cash"]

        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df["Total Portfolio Value"] = portfolio_df.sum(axis=1)

        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df["Total Portfolio Value"], label="Portfolio Value", color="blue")
        plt.plot(self.market_prices.index, self.market_prices, label="S&P 500", color="black", linestyle="dashed")
        plt.title("Portfolio Performance vs S&P 500")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

        print("Final Portfolio Holdings:")
        holdings = {ticker: self.ticker_holdings[ticker]["Volume"].iloc[-1] for ticker in self.tickers}
        print(pd.DataFrame.from_dict(holdings, orient="index", columns=["Shares"]))


port = Portfolio(tickers=["AAPL", "JPM", "GM", "SPY", "XOM", "WMT"], start_date="2023-12-19", end_date="2025-02-03", initial_balance=100000)
port.simulate()
print("\n")