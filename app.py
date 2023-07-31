from flask import Flask, render_template
from metrics.fundamental_analysis import get_ticker_symbols_finnhub
from metrics.fundamental_analysis import get_fundamental_analysis_metrics
from metrics.utils.stock_metrics import StockMetrics
from flask import jsonify
app = Flask(__name__)


@app.route('/')
def stock_browser():
    # Sample data for testing the layout
    tickers = get_ticker_symbols_finnhub()
    return render_template('stock_browser.html', tickers=tickers)


@app.route('/stock/<ticker_symbol>')
def get_stock_data(ticker_symbol):
    metrics = StockMetrics(ticker_symbol)
    return jsonify(metrics)


@app.route('/tickers')
def get_tickers():
    tickers = get_ticker_symbols_finnhub()
    return tickers[0:10]
    #return tickers


if __name__ == '__main__':
    app.run(debug=True)
