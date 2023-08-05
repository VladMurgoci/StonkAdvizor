from dataclasses import dataclass
from typing import Callable, Dict, List, Union

import finnhub
import numpy as np
import pandas as pd
import yfinance as yf
from dacite import from_dict

FINNHUB_CLIENT = finnhub.Client(api_key="cj4oprpr01qq6hgdnt60cj4oprpr01qq6hgdnt6g")
EXCHANGES = ['US']
YFINANCE_METRICS_MAP = {
    'activity_domain': ['sector'],
    'market_cap': ['marketCap'],
    'net_revenue': ['Total Revenue'],
    'net_income': ['Net Income'],
    'earnings_per_share': ['Basic EPS', 'trailingEps'],
    'forward_earnings_per_share': ['forwardEps'],
    'price_earnings_ratio': ['trailingPE'],
    'forward_price_earnings_ratio': ['forwardPE'],
    'projected_earnings_growth': ['Basic EPS', 'forwardEps', 'trailingEps'],
    'free_cash_flow': ['freeCashflow'],
    'return_on_equity': ['Net Income', 'Stockholders Equity', 'returnOnEquity'],
    'dividend_payout_ratio': ['payoutRatio'],
    'year_over_year_revenue_growth': ['Total Revenue', 'totalRevenue'],
    'price_to_free_cash_flow': ['regularMarketPreviousClose', 'freeCashflow', 'sharesOutstanding'],
    'stock_price': ['regularMarketPreviousClose'],
    'debt_to_equity_ratio': ['Total Debt', 'Stockholders Equity'],
    'return_on_assets': ['Net Income', 'Total Assets'],
    'return_on_investments': ['Net Income', 'Investments And Advances'],
    'revenue_per_employee': ['Total Revenue', 'fullTimeEmployees'],
    'price_to_earnings_growth': ['pegRatio'],
    'price_to_book_ratio': ['priceToBook'],
}

@dataclass
class FundamentalAnalysisMetrics:
    """
        The metrics which are lists represent the values for the last len(list) years
    """
    ticker: str
    activity_domain: str
    market_cap: int
    net_revenue: List[float]
    net_income: List[float]
    earnings_per_share: List[float]
    forward_earnings_per_share: float
    price_earnings_ratio: float
    forward_price_earnings_ratio: float
    projected_earnings_growth: float
    free_cash_flow: float
    return_on_equity: List[float]
    dividend_payout_ratio: float
    year_over_year_revenue_growth: List[float]
    price_to_free_cash_flow: float
    stock_price: float
    debt_to_equity_ratio: List[float]
    return_on_assets: List[float]
    return_on_investments: List[float]
    revenue_per_employee: List[float]
    price_to_earnings_growth: float
    price_to_book_ratio: float

def check_metric_exists_and_fill_out(ticker: yf.Ticker,
                                     metrics: Dict[str, str],
                                     metric_name: str,
                                     formula: Callable[[], Union[List[float], float, str]],
                                     alt: Union[List[float], float, str]) -> None:
    """ Checks if the corresponding metrics used in the formula exist in yfinance and if they do, it computes the formula and adds it to the metrics dictionary.
    """
    def property_exists(yf_metric):
        return yf_metric in ticker.info or yf_metric in ticker.income_stmt.index or \
               yf_metric in ticker.financials.index or yf_metric in ticker.balance_sheet.index
    if all(property_exists(yf_metric) for yf_metric in YFINANCE_METRICS_MAP[metric_name]):
        try:
            metrics[metric_name] = formula()
        except Exception as e:
            # print(f"Error while computing {metric_name} for ticker {ticker.ticker} : {e}")
            metrics[metric_name] = alt
    else:
        metrics[metric_name] = alt

def get_fundamental_analysis_metrics(ticker_symbol: str) -> FundamentalAnalysisMetrics:
    """Computes the fundamental analysis metrics for a given ticker symbol

    Args:
        ticker_symbol (str): string representing the ticker symbol of a company as traded on the market (e.g. 'AAPL' for "Apple Inc.")

    Returns:
        Dict[str, str]: _description_
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
    except ValueError:
        return None
    ticker_info = ticker.info
    yearly_income_statement, yearly_balance_sheet  = ticker.income_stmt, ticker.balance_sheet
    financials = ticker.financials
    metrics = {}
    metrics['ticker'] = ticker_symbol
    check_metric_exists_and_fill_out(ticker, metrics, 'activity_domain', lambda: ticker_info['sector'], 'n/a')
    check_metric_exists_and_fill_out(ticker, metrics, 'market_cap', lambda: ticker_info['marketCap'], -1)
    # @Mihneaghitu @VladMurgoci TODO : review whether 'Total Revenue' actually refers to net revenue
    check_metric_exists_and_fill_out(ticker, metrics, 'net_revenue', lambda: financials.loc['Total Revenue'].tolist(), [-1])
    check_metric_exists_and_fill_out(ticker, metrics, 'net_income', lambda: financials.loc['Net Income'].tolist(), [-1])
    # Basic EPS = (Net Income - Preferred Dividends) / Weighted Average Number of Common Shares Outstanding
    check_metric_exists_and_fill_out(ticker, metrics, 'earnings_per_share', 
                                     lambda: [ticker_info['trailingEps']] + yearly_income_statement.loc['Basic EPS'].tolist(), [-1])
    check_metric_exists_and_fill_out(ticker, metrics, 'forward_earnings_per_share', lambda: ticker_info['forwardEps'], -1)
    # P/E ratio = Current Market Price per Share / Earnings Per Share (EPS)
    check_metric_exists_and_fill_out(ticker, metrics, 'price_earnings_ratio', lambda: ticker_info['trailingPE'], -1)
    check_metric_exists_and_fill_out(ticker, metrics, 'forward_price_earnings_ratio', lambda: ticker_info['forwardPE'], -1)
    # @Mihneaghitu @VladMurgoci TODO : check whether we want both the trailing and forward PEG ratios and if they refer to what we think they refer
    # Projected Earnings Growth Rate = ((Future EPS - Current EPS) / Current EPS) * 100
    check_metric_exists_and_fill_out(ticker, metrics, 'projected_earnings_growth', 
                                     lambda: metrics['forward_earnings_per_share'] / metrics['earnings_per_share'][0], -1)
    # Free Cash Flow (FCF) = Operating Cash Flow - Capital Expenditures
    check_metric_exists_and_fill_out(ticker, metrics, 'free_cash_flow', lambda: ticker_info['freeCashflow'], -1)
    # Return on Equity (ROE) = Net Income / Shareholders Equity
    check_metric_exists_and_fill_out(ticker, metrics, 'return_on_equity', 
                                     lambda: (financials.loc['Net Income'] / yearly_balance_sheet.loc['Stockholders Equity']).tolist() + ticker_info['returnOnEquity'], [-1])
    # Dividend Payout Ratio = (Dividends per Share / Earnings per Share) * 100
    check_metric_exists_and_fill_out(ticker, metrics, 'dividend_payout_ratio', lambda: ticker_info['payoutRatio'], -1)
    # Year over Year Revenue Growth Rate = ((Current Year Revenue - Last Year Revenue) / Last Year Revenue) * 100
    def _compute_year_over_year_revenue_growth():
        total_revenues = np.array(financials.loc['Total Revenue'].tolist() + [ticker_info['totalRevenue']])
        return list(((total_revenues[1:] - total_revenues[:-1]) / total_revenues[:-1]) * 100)
    check_metric_exists_and_fill_out(ticker, metrics, 'year_over_year_revenue_growth', _compute_year_over_year_revenue_growth, [-1])
    # Price to Free Cash Flow = Stock Price / Free Cash Flow per Share
    check_metric_exists_and_fill_out(ticker, metrics, 'price_to_free_cash_flow', 
                                     lambda: ticker_info['regularMarketPreviousClose'] / (ticker_info['freeCashflow'] / ticker_info['sharesOutstanding']), -1)
    check_metric_exists_and_fill_out(ticker, metrics, 'stock_price', lambda: ticker_info['regularMarketPreviousClose'], -1)
    # Debt to Equity Ratio = Total Liabilities / Shareholders Equity
    check_metric_exists_and_fill_out(ticker, metrics, 'debt_to_equity_ratio', 
                                     lambda: (yearly_balance_sheet.loc['Total Debt'] / yearly_balance_sheet.loc['Stockholders Equity']).tolist(), [-1])
    # Return on Assets (ROA) = Net Income / Total Assets
    check_metric_exists_and_fill_out(ticker, metrics, 'return_on_assets', 
                                     lambda: (financials.loc['Net Income'] / yearly_balance_sheet.loc['Total Assets']).tolist(), [-1])
    # Return on Investments (ROI) = Net Income / Total Investments
    check_metric_exists_and_fill_out(ticker, metrics, 'return_on_investments', 
                                     lambda: (financials.loc['Net Income'] / yearly_balance_sheet.loc['Investments And Advances']).tolist(), [-1])
    # Revenue per Employee = Total Revenue / Number of Employees
    check_metric_exists_and_fill_out(ticker, metrics, 'revenue_per_employee', 
                                     lambda: (financials.loc['Total Revenue'] / ticker_info['fullTimeEmployees']).tolist(), [-1])
    # Price to Earnings Growth (PEG) Ratio = P/E ratio / Projected earnings growth rate
    check_metric_exists_and_fill_out(ticker, metrics, 'price_to_earnings_growth', lambda: ticker_info['pegRatio'], -1)
    # Price to Book Ratio = Stock Price / Book Value per Share
    check_metric_exists_and_fill_out(ticker, metrics, 'price_to_book_ratio', lambda: ticker_info['priceToBook'], -1)

    return from_dict(data_class=FundamentalAnalysisMetrics, data=metrics)

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
