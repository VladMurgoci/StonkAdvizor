from dataclasses import dataclass
from typing import Callable, Dict, List, Union

import finnhub
import pandas as pd
import yfinance as yf
from dacite import from_dict

FINNHUB_CLIENT = finnhub.Client(api_key="cj4oprpr01qq6hgdnt60cj4oprpr01qq6hgdnt6g")
EXCHANGES = ['US']
TRAILING_YEARS = [2022, 2021, 2020, 2019]
TRAILING_YEARS_TIMESTAMPS = [pd.Timestamp(date) for date in TRAILING_YEARS]
YFINANCE_METRICS_MAP = {
    'activity_domain': ['sector'],
    'market_cap': ['marketCap'],
    'net_revenue_y1': ['Total Revenue'],
    'net_revenue_y2': ['Total Revenue'],
    'net_revenue_y3': ['Total Revenue'],
    'net_revenue_y4': ['Total Revenue'],
    'net_income_y1': ['Net Income'],
    'net_income_y2': ['Net Income'],
    'net_income_y3': ['Net Income'],
    'net_income_y4': ['Net Income'],
    'earnings_per_share_y0': ['trailingEps'],
    'earnings_per_share_y1': ['Basic EPS'],
    'earnings_per_share_y2': ['Basic EPS'],
    'earnings_per_share_y3': ['Basic EPS'],
    'earnings_per_share_y4': ['Basic EPS'],
    'forward_earnings_per_share': ['forwardEps'],
    'price_earnings_ratio': ['trailingPE'],
    'forward_price_earnings_ratio': ['forwardPE'],
    'projected_earnings_growth': ['forwardEps', 'trailingEps'],
    'free_cash_flow': ['freeCashflow'],
    'return_on_equity_y0': ['returnOnEquity'],
    'return_on_equity_y1': ['Net Income', 'Stockholders Equity'],
    'return_on_equity_y2': ['Net Income', 'Stockholders Equity'],
    'return_on_equity_y3': ['Net Income', 'Stockholders Equity'],
    'return_on_equity_y4': ['Net Income', 'Stockholders Equity'],
    'dividend_payout_ratio': ['payoutRatio'],
    'year_over_year_revenue_growth_y0': ['Total Revenue', 'totalRevenue'],
    'year_over_year_revenue_growth_y1': ['Total Revenue'],
    'year_over_year_revenue_growth_y2': ['Total Revenue'],
    'year_over_year_revenue_growth_y3': ['Total Revenue'],
    'price_to_free_cash_flow': ['regularMarketPreviousClose', 'freeCashflow', 'sharesOutstanding'],
    'stock_price': ['regularMarketPreviousClose'],
    'debt_to_equity_ratio_y1': ['Total Debt', 'Stockholders Equity'],
    'debt_to_equity_ratio_y2': ['Total Debt', 'Stockholders Equity'],
    'debt_to_equity_ratio_y3': ['Total Debt', 'Stockholders Equity'],
    'debt_to_equity_ratio_y4': ['Total Debt', 'Stockholders Equity'],
    'return_on_assets_y1': ['Net Income', 'Total Assets'],
    'return_on_assets_y2': ['Net Income', 'Total Assets'],
    'return_on_assets_y3': ['Net Income', 'Total Assets'],
    'return_on_assets_y4': ['Net Income', 'Total Assets'],
    'return_on_investments_y1': ['Net Income', 'Investments And Advances'],
    'return_on_investments_y2': ['Net Income', 'Investments And Advances'],
    'return_on_investments_y3': ['Net Income', 'Investments And Advances'],
    'return_on_investments_y4': ['Net Income', 'Investments And Advances'],
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
    net_revenue_y1: float
    net_revenue_y2: float
    net_revenue_y3: float
    net_revenue_y4: float
    net_income_y1: float
    net_income_y2: float
    net_income_y3: float
    net_income_y4: float
    earnings_per_share_y0: float
    earnings_per_share_y1: float
    earnings_per_share_y2: float
    earnings_per_share_y3: float
    earnings_per_share_y4: float
    forward_earnings_per_share: float
    price_earnings_ratio: float
    forward_price_earnings_ratio: float
    projected_earnings_growth: float
    free_cash_flow: float
    return_on_equity_y0: float
    return_on_equity_y1: float
    return_on_equity_y2: float
    return_on_equity_y3: float
    return_on_equity_y4: float
    dividend_payout_ratio: float
    price_to_free_cash_flow: float
    stock_price: float
    debt_to_equity_ratio_y1: float
    debt_to_equity_ratio_y2: float
    debt_to_equity_ratio_y3: float
    debt_to_equity_ratio_y3: float
    return_on_assets_y1: float
    return_on_assets_y2: float
    return_on_assets_y3: float
    return_on_assets_y4: float
    return_on_investments_y1: float
    return_on_investments_y2: float
    return_on_investments_y3: float
    return_on_investments_y4: float
    price_to_earnings_growth: float
    price_to_book_ratio: float

def check_metric_exists_and_fill_out(ticker: yf.Ticker,
                                     metrics: Dict[str, str],
                                     metric_name: str,
                                     formula: Callable[[], Union[float, str]],
                                     series_idx: int = None,
                                     alt: Union[float, str] = -1) -> None:
    """ Checks if the corresponding metrics used in the formula exist in yfinance and if they do, it computes the formula and adds it to the metrics dictionary.
    """
    def property_exists(yf_metric):
        return yf_metric in ticker.info or yf_metric in ticker.income_stmt.index or \
               yf_metric in ticker.financials.index or yf_metric in ticker.balance_sheet.index
    if all(property_exists(yf_metric) for yf_metric in YFINANCE_METRICS_MAP[metric_name]):
        try:
            value = formula()
            if series_idx is None:
                metrics[metric_name] = value
            elif series_idx is not None and value.index[series_idx].year == TRAILING_YEARS[series_idx]:
                metrics[metric_name] = value[series_idx]
            else:
                metrics[metric_name] = alt
        except Exception as _:
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
    check_metric_exists_and_fill_out(ticker, metrics, 'activity_domain', lambda: ticker_info['sector'], alt='n/a')
    check_metric_exists_and_fill_out(ticker, metrics, 'market_cap', lambda: ticker_info['marketCap'])
    # @Mihneaghitu @VladMurgoci TODO : review whether 'Total Revenue' actually refers to net revenue
    # Basic EPS = (Net Income - Preferred Dividends) / Weighted Average Number of Common Shares Outstanding
    check_metric_exists_and_fill_out(ticker, metrics, 'earnings_per_share_y0', lambda: ticker_info['trailingEps']) 
    check_metric_exists_and_fill_out(ticker, metrics, 'forward_earnings_per_share', lambda: ticker_info['forwardEps'])
    # P/E ratio = Current Market Price per Share / Earnings Per Share (EPS)
    check_metric_exists_and_fill_out(ticker, metrics, 'price_earnings_ratio', lambda: ticker_info['trailingPE'])
    check_metric_exists_and_fill_out(ticker, metrics, 'forward_price_earnings_ratio', lambda: ticker_info['forwardPE'])
    # @Mihneaghitu @VladMurgoci TODO : check whether we want both the trailing and forward PEG ratios and if they refer to what we think they refer
    # Projected Earnings Growth Rate = ((Future EPS - Current EPS) / Current EPS) * 100
    check_metric_exists_and_fill_out(ticker, metrics, 'projected_earnings_growth', lambda: ticker_info['forwardPE'] / ticker_info['trailingEPS'])
    # Free Cash Flow (FCF) = Operating Cash Flow - Capital Expenditures
    check_metric_exists_and_fill_out(ticker, metrics, 'free_cash_flow', lambda: ticker_info['freeCashflow'])
    # Return on Equity (ROE) = Net Income / Shareholders Equity
    check_metric_exists_and_fill_out(ticker, metrics, 'return_on_equity_y0', lambda: ticker_info['returnOnEquity'])
    # Dividend Payout Ratio = (Dividends per Share / Earnings per Share) * 100
    check_metric_exists_and_fill_out(ticker, metrics, 'dividend_payout_ratio', lambda: ticker_info['payoutRatio'])
    # Price to Free Cash Flow = Stock Price / Free Cash Flow per Share
    check_metric_exists_and_fill_out(ticker, metrics, 'price_to_free_cash_flow', 
                                     lambda: ticker_info['regularMarketPreviousClose'] / (ticker_info['freeCashflow'] / ticker_info['sharesOutstanding']))
    check_metric_exists_and_fill_out(ticker, metrics, 'stock_price', lambda: ticker_info['regularMarketPreviousClose'])
    for i in range(4):
        # Debt to Equity Ratio = Total Liabilities / Shareholders Equity
        check_metric_exists_and_fill_out(ticker, metrics, 'debt_to_equity_ratio_y' + str(i + 1),
                                         lambda: yearly_balance_sheet.loc['Total Debt'] / yearly_balance_sheet.loc['Stockholders Equity'], series_idx=i)
        # Return on Assets (ROA) = Net Income / Total Assets
        check_metric_exists_and_fill_out(ticker, metrics, 'return_on_assets_y' + str(i + 1),
                                         lambda: financials.loc['Net Income'] / yearly_balance_sheet.loc['Total Assets'], series_idx=i)
        # Return on Investments (ROI) = Net Income / Total Investments
        check_metric_exists_and_fill_out(ticker, metrics, 'return_on_investments_y' + str(i + 1),
                                         lambda: financials.loc['Net Income'] / yearly_balance_sheet.loc['Investments And Advances'], series_idx=i)
        # Return on Equity (ROE) = Net Income / Shareholders Equity
        check_metric_exists_and_fill_out(ticker, metrics, 'return_on_equity_y' + str(i + 1),
                                         lambda: financials.loc['Net Income'] / yearly_balance_sheet.loc['Stockholders Equity'], series_idx=i)
        # Basic EPS = (Net Income - Preferred Dividends) / Weighted Average Number of Common Shares Outstanding
        check_metric_exists_and_fill_out(ticker, metrics, 'earnings_per_share_y' + str(i + 1), lambda: yearly_income_statement.loc['Basic EPS'], series_idx=i)
        check_metric_exists_and_fill_out(ticker, metrics, 'net_revenue_y' + str(i + 1),
                                         lambda: financials.loc['Total Revenue'], series_idx=i)
        check_metric_exists_and_fill_out(ticker, metrics, 'net_income_y' + str(i + 1),
                                         lambda: financials.loc['Net Income'], series_idx=i)
    # Price to Earnings Growth (PEG) Ratio = P/E ratio / Projected earnings growth rate
    check_metric_exists_and_fill_out(ticker, metrics, 'price_to_earnings_growth', lambda: ticker_info['pegRatio'])
    # Price to Book Ratio = Stock Price / Book Value per Share
    check_metric_exists_and_fill_out(ticker, metrics, 'price_to_book_ratio', lambda: ticker_info['priceToBook'])

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
