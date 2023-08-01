from dataclasses import dataclass
from typing import Dict, List

import finnhub
import numpy as np
import pandas as pd
import yfinance as yf

FINNHUB_CLIENT = finnhub.Client(api_key="cj4oprpr01qq6hgdnt60cj4oprpr01qq6hgdnt6g")
EXCHANGES = ['US']
@dataclass
class FundamentalAnalysisMetrics:
    """
        The metrics which are lists represent the values for the last len(list) years
    """
    activity_domain: str
    market_cap: int
    net_revenue: List[float]
    net_income: List[float]
    earnings_per_share: List[float]
    forward_earnings_per_share: float
    price_earnings_ratio: float
    forward_price_earnings_ratio: float
    projected_earnings_growth: float
    forward_projected_earnings_growth: float
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


def get_fundamental_analysis_metrics(ticker_symbol: str) -> Dict[str, str]:
    """Computes the fundamental analysis metrics for a given ticker symbol

    Args:
        ticker_symbol (str): string representing the ticker symbol of a company as traded on the market (e.g. 'AAPL' for "Apple Inc.")

    Returns:
        Dict[str, str]: _description_
    """
    ticker = yf.Ticker(ticker_symbol)
    ticker_info = ticker.info
    yearly_income_statement, yearly_balance_sheet  = ticker.income_stmt, ticker.quarterly_balance_sheet
    financials = ticker.financials
    metrics = {}
    metrics['activity_domain'] = ticker_info['sector']
    metrics['market_cap'] = ticker_info['marketCap']
    # @Mihneaghitu @VladMurgoci TODO : review whether 'Total Revenue' actually refers to net revenue
    metrics['net_revenue'] = financials.loc['Total Revenue'].tolist()
    metrics['net_income'] = financials.loc['Net Income'].tolist()
    # Basic EPS = (Net Income - Preferred Dividends) / Weighted Average Number of Common Shares Outstanding
    metrics['earnings_per_share'] = [ticker_info['trailingEps']] + yearly_income_statement.loc['Basic EPS'].tolist()
    metrics['forward_earnings_per_share'] = ticker_info['forwardEps']
    # P/E ratio = Current Market Price per Share / Earnings Per Share (EPS)
    metrics["price_earnings_ratio"] = ticker_info['trailingPE']
    metrics["forward_price_earnings_ratio"] = ticker_info['forwardPE']
    # @Mihneaghitu @VladMurgoci TODO : check whether we want both the trailing and forward PEG ratios and if they refer to what we think they refer
    # Projected Earnings Growth Rate = ((Future EPS - Current EPS) / Current EPS) * 100
    metrics["projected_earnings_growth"] = metrics['forward_earnings_per_share'] / metrics['earnings_per_share'][0]
    # Forward/Trailing PEG ratio = Forward/Trailing P/E ratio / Projected earnings growth rate
    metrics['forward_projected_earnings_growth'] = ticker_info['pegRatio']
    # Free Cash Flow (FCF) = Operating Cash Flow - Capital Expenditures
    metrics['free_cash_flow'] = ticker_info['freeCashflow']
    # Return on Equity (ROE) = Net Income / Shareholders Equity
    metrics['return_on_equity'] = (financials.loc['Net Income'] / yearly_balance_sheet.loc['Stockholders Equity']).tolist()
    metrics['return_on_equity'].append(ticker_info['returnOnEquity'])
    # Dividend Payout Ratio = (Dividends per Share / Earnings per Share) * 100
    metrics['dividend_payout_ratio'] = ticker_info['payoutRatio']
    # Year over Year Revenue Growth Rate = ((Current Year Revenue - Last Year Revenue) / Last Year Revenue) * 100
    total_revenues = np.array(financials.loc['Total Revenue'].tolist() + [ticker_info['revenue']])
    metrics['year_over_year_revenue_growth'] = ((total_revenues[1:] - total_revenues[:-1]) / total_revenues[:-1]) * 100
    # Price to Free Cash Flow = Stock Price / Free Cash Flow per Share
    metrics['price_to_free_cash_flow'] =  ticker_info['regularMarketPreviousClose'] / (ticker_info['freeCashflow'] / ticker_info['sharesOutstanding'])
    metrics['stock_price'] = ticker_info['regularMarketPreviousClose']
    # Debt to Equity Ratio = Total Liabilities / Shareholders Equity
    metrics['debt_to_equity_ratio'] = (yearly_income_statement.loc['Total Debt'] / yearly_income_statement.loc['Stockholders Equity']).tolist()
    # Return on Assets (ROA) = Net Income / Total Assets
    metrics['return_on_assets'] = (financials.loc['Net Income'] / yearly_income_statement.loc['Total Assets']).tolist()
    # Return on Investments (ROI) = Net Income / Total Investments
    metrics['return_on_investments'] = (financials.loc['Net Income'] / yearly_income_statement.loc['Investments And Advances']).tolist()
    # Revenue per Employee = Total Revenue / Number of Employees
    metrics['revenue_per_employee'] = (financials.loc['Total Revenue'].tolist()[-1] / ticker_info['fullTimeEmployees'])
    # Price to Earnings Growth (PEG) Ratio = P/E ratio / Projected earnings growth rate
    metrics['price_to_earnings_growth'] = ticker_info['pegRatio']
    # Price to Book Ratio = Stock Price / Book Value per Share
    metrics['price_to_book_ratio'] = ticker_info['priceToBook']

    return metrics

def get_ticker_symbols_finnhub() -> List[yf.Ticker]:
    """ Gets all the ticker symbols from the Finnhub API for all the exchanges specified in the EXCHANGES list

    See:
        https://docs.google.com/spreadsheets/d/1I3pBxjfXB056-g_JYf_6o3Rns3BV2kMGG1nCatb91ls/edit#gid=0 for a list of all the exchanges

    Returns:
        List[str]: A list of all the ticker symbols
    """
    ticker_symbols = []
    for exchange in EXCHANGES:
        stocks_dictionary_list = FINNHUB_CLIENT.stock_symbols(exchange=exchange)
        stocks_dictionary_list = [stock_dict for stock_dict in stocks_dictionary_list if stock_dict['type'] == 'Common Stock']
        for dicts in stocks_dictionary_list:
            ticker_symbols.append(dicts['symbol'])

    return ticker_symbols
