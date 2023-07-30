from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def stock_browser():
    # Sample data for testing the layout
    stocks = [
        {'ticker': 'AAPL', 'market_cap': '2.5T', 'pe_ratio': 30.5},
        {'ticker': 'GOOGL', 'market_cap': '1.8T', 'pe_ratio': 40.2},
        # Add more sample stock data here
    ]

    return render_template('stock_browser.html', stocks=stocks)

if __name__ == '__main__':
    app.run(debug=True)
