from flask import Flask, render_template
from metrics.fundamental_analysis import get_ticker_symbols_finnhub
from metrics.fundamental_analysis import get_fundamental_analysis_metrics
from metrics.utils.stock_metrics import StockMetrics
from flask import jsonify
import csv
app = Flask(__name__)

tickers = []

@app.route('/')
def stock_browser():
    global tickers
    # Sample data for testing the layout
    tickers = get_ticker_symbols_finnhub()
    return render_template('stock_browser.html', tickers=tickers)


@app.route('/stock/<ticker_symbol>')
def get_stock_data(ticker_symbol):
    metrics = StockMetrics(ticker_symbol)
    return jsonify({
        'ticker_symbol': metrics.ticker_symbol,
        'market_cap': metrics.market_cap,
        'net_income': metrics.net_income,
        'basic_eps': metrics.basic_eps,
        'activity_domain': metrics.activity_domain
    })

@app.route('/stock/search/<search_term>')
def search_stock(search_term):
    global tickers
    filtered_tickers = [ticker for ticker in tickers if search_term.upper() in ticker[0].upper()]
    return jsonify(filtered_tickers)

@app.route('/tickers')
def get_tickers():
    global tickers
    with open('data/tickers.csv', 'r') as file:
        reader = csv.reader(file)
        tickers = list(reader)
    return tickers


if __name__ == '__main__':
    app.run(debug=True)
