import yfinance as yf

def get_large_cap_ticker_symbols(tickers: [str]):
    high_cap_tickers = []

    for ticker in tickers:
      if ticker.info['marketCap'] > 10000000000:
        high_cap_tickers.append(ticker)
    return tickers

def get_mid_cap_ticker_symbols(tickers: [str]):
    mid_cap_tickers = []

    for ticker in tickers:
      if ticker.info['marketCap'] > 2000000000 and ticker.info['marketCap'] < 10000000000:
        mid_cap_tickers.append(ticker)
    return tickers

def get_small_cap_ticker_symbols(tickers: [str]):
    small_cap_tickers = []

    for ticker in tickers:
      if ticker.info['marketCap'] < 2000000000 and ticker.info['marketCap'] > 250000000:
        small_cap_tickers.append(ticker)
    return tickers