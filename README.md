# StockSage — Stock Price Prediction Web App

## Folder Structure
```
StockSage/
├── app.py
├── requirements.txt
├── stock_price_model.pkl      (auto-created the first time you run a prediction)
├── static/
│   ├── css/style.css
│   ├── images/                (prediction chart is saved here)
│   └── js/
└── templates/
    ├── base.html, navbar.html
    ├── home.html, about.html, services.html, faq.html, contact.html
    ├── login.html, register.html, dashboard.html
    ├── predict.html, result.html
    └── error.html
```

## How to Run in Anaconda Prompt

1. Open **Anaconda Prompt**.
2. Navigate to the project folder:
   ```
   cd path\to\StockSage
   ```
3. (Recommended) Create and activate a dedicated environment:
   ```
   conda create -n stocksage python=3.11 -y
   conda activate stocksage
   ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the app:
   ```
   python app.py
   ```
6. Open your browser and go to:
   ```
   http://127.0.0.1:5000
   ```

The site will load with the StockSage home page. Register an account, log in,
then go to **Predict**, enter a ticker (e.g. `AAPL`, `TSLA`, `INFY.NS`) and a
number of days, and it will fetch live historical data, train a model, and
show you a forecast chart plus a day-by-day table. Every prediction is saved
to your **Dashboard**.

## Notes
- `stock_price_model.pkl` is regenerated automatically on every prediction
  (trained on whatever ticker you just requested), so it always reflects a
  real, working model rather than a stale/incompatible file.
- Requires an internet connection (to fetch live prices via Yahoo Finance).
- This is an educational demo — not financial advice.
