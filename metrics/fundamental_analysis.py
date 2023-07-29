import yfinance as yf
import requests
import csv
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import finnhub

@dataclass
class FundamentalAnalysisMetrics:
    price_earnings_ratio: float
    activity_domain: str
    market_cap: float
    net_revenue: float
    earnings_per_share: float
    projected_earnings_growth: float
    free_cash_flow: float
    return_on_equity: float 
    dividend_payout_ratio: float
    year_over_year_revenue_growth: float
    price_to_free_cash_flow: float
    stock_price: float
    debt_to_equity_ratio: float
    return_on_assets: float
    return_on_investments: float
    revenue_per_employee: float
    price_to_earnings_growth: float
    price_to_book_ratio: float


def get_fundamental_analysis_metrics(ticker_symbol: str) -> Dict[yf.Ticker, FundamentalAnalysisMetrics]:
    ticker = yf.Ticker(ticker_symbol)
    yearly_income_statement, quarterly_income_statement = ticker.income_stmt, ticker.quarterly_income_stmt
    yearly_balance_sheet, quarterly_balance_sheet = ticker.quarterly_balance_sheet, ticker.quarterly_balance_sheet

    return None

def get_ticker_symbols() -> List[str]: 
    bats_symbols_response = requests.get("https://www.cboe.com/us/equities/market_statistics/listed_symbols/csv", timeout=10)
    bats_symbols_file_name = "bats_symbols.csv"
    with open(bats_symbols_file_name, "wb") as bats_symbols_file:
        bats_symbols_file.write(bats_symbols_response.content)
    with open(bats_symbols_file_name, "r", encoding="utf-8") as bats_symbols_file:
        csv_reader = csv.reader(bats_symbols_file, delimiter=',')
        ticker_symbols = [row[0] for idx, row in enumerate(csv_reader) if idx > 0]
    
    return ticker_symbols

def get_ticker_symbols_finnhub() -> List[str]:
    #retrieve tickers from every exchange available
    us_tickers=[]
    finnhub_client = finnhub.Client(api_key="cj2o6rhr01qr0f6fq0k0cj2o6rhr01qr0f6fq0kg")
    us_stocks_dictionary_list = finnhub_client.stock_symbols('US')
    for dicts in us_stocks_dictionary_list:
        us_tickers.append(dicts['symbol'])

    return us_tickers
# print(get_ticker_symbols())
print(get_ticker_symbols_finnhub())
print("AAPL" in get_ticker_symbols_finnhub())
