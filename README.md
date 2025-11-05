# Stock Price Prediction & Analysis Pipeline

> A comprehensive toolkit for stock market analysis and prediction combining deep learning, statistical methods, and sentiment analysis.

## About The Project

This project provides an advanced stock market analysis system that leverages multiple modeling approaches to forecast prices and predict market direction. It seamlessly integrates:

- **Deep Learning Models** (LSTM, GRU, RNN)
- **Statistical Methods** (ARIMA/SARIMA)
- **Tree-Based Models** (RandomForest)
- **Sentiment Analysis** from news sources

### Core Capabilities

- Multi-step price forecasting using historical data
- Ensemble predictions combining multiple models for enhanced accuracy
- Sentiment analysis from news to predict market movements
- Technical indicators and advanced visualizations

**Check the `charts/` folder for visual results**

---

## Available Tasks

| Task                            | Description                                                       |
| ------------------------------- | ----------------------------------------------------------------- |
| **1. Univariate Multistep**     | Predict multiple future prices using only historical close prices |
| **2. Multivariate Single-Step** | Use all OHLCV features to predict tomorrow's close price          |
| **3. Multivariate Multistep**   | Combine multiple features to forecast several days ahead          |
| **4. Model Factory**            | Compare LSTM, GRU, and RNN architectures automatically            |
| **5. Charts & Visualizations**  | Generate candlestick charts and volatility boxplots               |
| **6. 2-Way Ensemble**           | Combine LSTM and ARIMA predictions with optimized weights         |
| **7. Sentiment Analysis**       | Use news sentiment to classify market direction (Up/Down)         |
| **8. 3-Way Ensemble**           | Combine LSTM, ARIMA, and RandomForest for maximum accuracy        |

---

## How It Works

### Data Pipeline

The system automatically handles:

1. Downloads stock price data from Yahoo Finance
2. Cleans and normalizes OHLCV data
3. Handles missing values using forward-fill or interpolation
4. Creates time-windowed sequences for model training
5. Splits data chronologically into train/validation/test sets
6. Scales features to improve model performance

### Model Types

#### Deep Learning Models

- **LSTM** (Long Short-Term Memory) - Best for capturing long-term patterns
- **GRU** (Gated Recurrent Unit) - Lighter than LSTM, similar accuracy
- **RNN** (Simple Recurrent Network) - Fastest but may miss long dependencies

#### Statistical Models

- **ARIMA** - Classical time series forecasting
- **SARIMA** - ARIMA with seasonal patterns

#### Tree-Based Models

- **RandomForest** - Ensemble of decision trees for robust predictions

#### Ensemble Methods

- Automatically finds optimal weights to combine different models
- Uses validation set to tune blend weights
- Typically outperforms individual models

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow statsmodels mplfinance nltk
```

**Optional** (for advanced features):

```bash
pip install transformers torch xgboost
```

---

## How to Run the Project

### Quick Start - Run Everything

```bash
python final.py --all
```

Runs all 8 tasks with default settings (ticker: TSLA, 60-day lookback, 20 epochs).

### Run Specific Tasks

**Single Task Examples:**

```bash
python final.py --univariate          # Task 1: Forecast 5 days ahead
python final.py --mv-single           # Task 2: Next-day prediction
python final.py --ensemble            # Task 6: LSTM + ARIMA
python final.py --sentiment           # Task 7: News sentiment
python final.py --ensemble-3way       # Task 8: 3-model ensemble
```

**Multiple Tasks:**

```bash
python final.py --tasks 1 3 5         # Run tasks 1, 3, and 5
python final.py --ensemble --sentiment --charts
```

### Customize Settings

**Change Stock Ticker:**

```bash
python final.py --all --ticker AAPL
```

**Adjust Training Parameters:**

```bash
python final.py --ensemble --lookback 90 --epochs 30
```

**Configure ARIMA:**

```bash
# Simple ARIMA
python final.py --ensemble --arima-order 5 1 0

# Seasonal ARIMA (monthly patterns)
python final.py --ensemble --seasonal-order 1 1 1 12
```

---

## Task Details

### Task 1: Univariate Multistep Forecasting

Uses only historical close prices to predict 5 days ahead.

```bash
python final.py --univariate
```

**Output Files:**

- `charts/TICKER_uni_step1.png` - Next-day predictions
- `charts/TICKER_uni_step5.png` - 5-day ahead predictions

---

### Task 2: Multivariate Single-Step

Uses all OHLCV features to predict tomorrow's close price.

```bash
python final.py --mv-single
```

**Output Files:**

- `charts/TICKER_mv1_step1.png` - Predictions vs actual

---

### Task 3: Multivariate Multistep

Uses all features to forecast multiple days ahead.

```bash
python final.py --mv-multistep
```

**Output Files:**

- `charts/TICKER_mvk_step1.png` - First step predictions
- `charts/TICKER_mvk_step5.png` - Fifth step predictions

---

### Task 4: Model Factory Experiments

Automatically compares different architectures (LSTM, GRU, BiLSTM).

```bash
python final.py --experiments
```

**Output Files:**

- `experiments_v03.csv` - Performance comparison table

---

### Task 5: Charts & Visualizations

Generates candlestick charts and volatility analysis.

```bash
python final.py --charts
```

**Output Files:**

- `charts/TICKER_candles_1d.png` - Daily candlestick chart
- `charts/TICKER_candles_3d.png` - 3-day aggregated candles
- `boxplots.png` - Moving-window volatility analysis

---

### Task 6: 2-Way Ensemble (LSTM + ARIMA)

Combines deep learning and statistical forecasting with auto-tuned weights.

```bash
python final.py --ensemble
```

**What It Does:**

1. Trains LSTM model on the data
2. Trains ARIMA model on the same data
3. Tests 21 different weight combinations on validation set
4. Applies best weights to test set
5. Generates comparison plots

**Output Files:**

- `ensemble_results/TICKER_weight_search.png` - Weight optimization curve
- `ensemble_results/TICKER_ensemble_vs_actual.png` - Main results
- `ensemble_results/TICKER_ensemble_vs_lstm.png` - Ensemble vs LSTM only
- `ensemble_results/TICKER_ensemble_vs_arima.png` - Ensemble vs ARIMA only
- `ensemble_results/TICKER_all_models_comparison.png` - All models together
- `ensemble_results/TICKER_ensemble_summary.csv` - Performance metrics

**Example Output:**

```
Optimal weight found: w = 0.650
This means: Combined = 65.0% LSTM + 35.0% ARIMA
Validation RMSE with ensemble: $7.8912

Test Set Results:
  LSTM      $8.2534
  ARIMA     $9.4521
  Ensemble  $7.8912  +4.4%
```

---

### Task 7: Sentiment Analysis & Classification

Uses news headlines to predict if tomorrow's price will go up or down.

```bash
python final.py --sentiment
```

**What It Does:**

1. Collects recent news from Yahoo Finance
2. Analyzes sentiment using VADER or FinBERT
3. Aggregates sentiment to daily scores (with recency weighting)
4. Combines sentiment with technical indicators
5. Trains classifier to predict Up/Down movement
6. Compares performance with and without sentiment

**Output Files:**

- `charts/TICKER_vader_logreg_cm_full.png` - Confusion matrix (with sentiment)
- `charts/TICKER_vader_logreg_cm_base.png` - Confusion matrix (baseline)
- `charts/TICKER_sentiment_metrics.json` - Accuracy metrics

**Example Output:**

```
[With Sentiment]  Acc=0.647  Prec=0.658  Rec=0.612  F1=0.634
[Baseline]        Acc=0.591  Prec=0.602  Rec=0.578  F1=0.590
```

---

### Task 8: 3-Way Ensemble (LSTM + ARIMA + RandomForest)

Combines three different model types for maximum accuracy.

```bash
python final.py --ensemble-3way
```

**What It Does:**

1. Trains LSTM (deep learning)
2. Trains ARIMA (statistical)
3. Trains RandomForest (tree-based)
4. Grid searches 121 weight combinations
5. Applies optimal 3-way blend to test set

**Output Files:**

- `charts/TICKER_3way_ensemble.png` - All models comparison

**Example Output:**

```
Individual Model Performance:
  LSTM          $8.2534
  ARIMA         $9.4521
  RandomForest  $7.9103

Top 3 Weight Combinations:
  Rank  LSTM   ARIMA   RF    Val RMSE
  1     50.0%  10.0%  40.0%  $7.6234
  2     60.0%   0.0%  40.0%  $7.6891
  3     45.0%  15.0%  40.0%  $7.7123

3-Way Ensemble  $7.6234  +7.6% vs best single model
```

---

## Output File Structure

After running tasks, you'll find:

```
project/
├── charts/                              # All visualizations
│   ├── TICKER_uni_step1.png            # Univariate predictions
│   ├── TICKER_mv1_step1.png            # Multivariate predictions
│   ├── TICKER_candles_1d.png           # Candlestick chart
│   ├── TICKER_3way_ensemble.png        # 3-way ensemble results
│   └── run_summary.json                # Run configuration summary
│
├── ensemble_results/                    # Ensemble-specific outputs
│   ├── TICKER_weight_search.png        # Weight optimization
│   ├── TICKER_all_models_comparison.png
│   └── TICKER_ensemble_summary.csv     # Metrics table
│
├── data_cache/                          # Downloaded data (reusable)
│   ├── TICKER_2020-01-01_2023-08-01.csv
│   └── TICKER_2020-01-01_2023-08-01_scalers.pkl
│
└── experiments_v03.csv                  # Model architecture comparison
```

---

## Understanding the Metrics

| Metric       | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| **RMSE**     | Root Mean Squared Error - Average prediction error in dollars (lower is better)       |
| **MAE**      | Mean Absolute Error - Average absolute difference between predicted and actual prices |
| **MAPE**     | Mean Absolute Percentage Error - Error as a percentage of the actual price            |
| **Accuracy** | Percentage of correct Up/Down predictions (classification)                            |
| **F1 Score** | Balance between precision and recall for classification                               |

---

## Common Issues & Solutions

### Issue: "VADER lexicon not found"

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Issue: "ARIMA convergence failed"

Try simpler ARIMA order:

```bash
python final.py --ensemble --arima-order 1 1 1
```

### Issue: Out of memory during training

Reduce batch size or epochs:

```bash
python final.py --univariate --epochs 10
```

### Issue: No news data found

This is normal for some tickers. The pipeline will run baseline models without sentiment.

---

## Project Configuration

You can edit default settings in `final.py`:

```python
COMPANY = 'TSLA'              # Default stock ticker
TRAIN_START = '2020-01-01'    # Training start date
TRAIN_END = '2023-08-01'      # Training end date
```

---

## License

This project is provided as-is for educational and research purposes.

---

**Happy Trading!**
