from dataclasses import dataclass
from typing import List, Optional

import finnhub
import yfinance as yf

FINNHUB_CLIENT = finnhub.Client(api_key="cj2o6rhr01qr0f6fq0k0cj2o6rhr01qr0f6fq0kg")
EXCHANGES = ['US']

@dataclass
class StockMetrics:
    """
        Metrics for one stock
    """

    ticker_symbol: str
    market_cap: Optional[int] = None
    activity_domain: Optional[str] = None
    net_income_4yrs: Optional[List[float]] = None
    net_income: Optional[float] = None
    basic_eps_4yrs: Optional[List[float]] = None
    basic_eps: Optional[float] = None


    def __init__(self, ticker_symbol: str):
        """
        Initiliazes stock metrics based on yahoo finance query
        :return: StockMetrics
        """
        self.ticker_symbol = ticker_symbol
        ticker = yf.Ticker(ticker_symbol)
        ticker_info = ticker.info
        yearly_income_statement, yearly_balance_sheet  = ticker.income_stmt, ticker.quarterly_balance_sheet
        financials = ticker.financials
        self._initialize_from_ticker_info(ticker_info)
        self._initialize_from_yearly_income_statement(yearly_income_statement)
        self._initialize_from_yearly_balance_sheet(yearly_balance_sheet)
        self.initialize_from_financials(financials)

    def _initialize_from_ticker_info(self, ticker_info):
        if 'marketCap' in ticker_info:
            self.market_cap = ticker_info['marketCap']
        if 'activity_domain' in ticker_info:
            self.activity_domain = ticker_info['sector']

    def _initialize_from_yearly_income_statement(self, yearly_income_statement):
        if 'Basic EPS' in yearly_income_statement.index:
            self.basic_eps_4yrs = yearly_income_statement.loc['Basic EPS'].tolist()
            self.basic_eps = self.basic_eps_4yrs[0]


    def _initialize_from_yearly_balance_sheet(self, yearly_balance_sheet):
        pass

    def initialize_from_financials(self, financials):
        if 'Total Revenue' in financials.index:
            self.net_income_4yrs = financials.loc['Total Revenue'].tolist()
            self.net_income = self.net_income_4yrs[0]

