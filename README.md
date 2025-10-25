# 📈 Cryptocurrency LSTM Price Prediction Dashboard

Professional Shiny dashboard for predicting cryptocurrency prices using deep learning (LSTM neural networks).

## ✨ Features

- **10 Cryptocurrencies:** Bitcoin, Ethereum, BNB, XRP, Cardano, Solana, Polkadot, Dogecoin, Avalanche, Polygon
- **Smart Model Caching:** Trained models are saved and reused (predictions in <5 seconds!)
- **Multi-Day Forecasting:** Predict 1 to 30 days ahead
- **17 Technical Indicators:** RSI, MACD, Bollinger Bands, Volume, Volatility, and more
- **Deep LSTM Architecture:** 2-layer LSTM with regularization and bias correction
- **Real-Time Training:** Models train automatically or on-demand
- **Trading Signals:** Buy/Sell/Hold recommendations with confidence levels
- **Interactive Visualizations:** Plotly charts, performance metrics, training history

## 📁 Project Structure

```
crypto-lstm-dashboard/
├── crypto_lstm_functions.R      # Core ML functions (modular)
├── crypto-dashboard-app.R       # Shiny app (UI + server)
├── models/                      # Saved trained models (auto-created)
│   ├── BTC-USD_model/          # Keras model files
│   ├── BTC-USD_metadata.rds    # Model metadata
│   └── ...
├── cache/                       # Temporary cache (auto-created)
└── README.md                    # This file
```

## 🚀 Installation

### 1. Install R (version 4.0+)

Download from: https://cran.r-project.org/

### 2. Install Required Packages

```r
# Core packages
install.packages(c(
  "shiny",
  "shinydashboard",
  "shinyWidgets",
  "shinycssloaders",
  "DT",
  "plotly",
  "dplyr",
  "ggplot2"
))

# Financial data
install.packages(c(
  "quantmod",
  "TTR",
  "Metrics"
))

# Deep learning (TensorFlow + Keras)
install.packages("keras3")

# Then install TensorFlow backend:
library(keras3)
install_keras()
```

**Note:** TensorFlow installation may take 5-10 minutes. It will download ~500MB of files.

### 3. Verify Installation

```r
library(keras3)
library(tensorflow)

# Should print TensorFlow version (e.g., 2.15.0)
tensorflow::tf$version$VERSION
```

## 🎮 Quick Start

### Method 1: Run in RStudio

1. Open RStudio
2. Set working directory to project folder
3. Open `crypto-dashboard-app.R`
4. Click "Run App" button (top right)

### Method 2: Run from R Console

```r
setwd("path/to/crypto-lstm-dashboard")
shiny::runApp("crypto-dashboard-app.R")
```

### Method 3: Run from Terminal

```bash
cd crypto-lstm-dashboard
R -e "shiny::runApp('crypto-dashboard-app.R')"
```

The app will open in your default web browser at `http://127.0.0.1:XXXX`

## 📖 How to Use

### Quick Prediction (Use Cached Model)

1. **Select Cryptocurrency** from dropdown (e.g., Bitcoin)
2. **Choose Forecast Days** (1-30 days)
3. Click **"Quick Prediction"** button
4. **If model exists:** Instant prediction (<5 sec)
5. **If new coin:** Auto-trains model (5-10 min first time)

### Train New Model

1. **Select Cryptocurrency**
2. Optionally: Uncheck "Use cached model" and adjust epochs
3. Click **"Train New Model"** button
4. Wait 5-10 minutes for training
5. Model is saved for future use

### View Results

Navigate through tabs to see:
- **Tomorrow's Forecast:** Price prediction with confidence
- **Performance Metrics:** RMSE, MAE, MAPE, Direction Accuracy
- **Prediction Visualization:** Interactive chart (actual vs predicted)
- **Training History:** Loss curves over epochs
- **Trading Recommendation:** Buy/Sell/Hold signals
- **Data Table:** Detailed predictions with errors

### Model Management

Go to **"Model Training"** tab to:
- View all trained models
- See model performance metrics
- Monitor training logs
- Re-train outdated models

## 🧠 Model Architecture

```
Input: Past 30 days × 17 features

↓

LSTM Layer 1 (96 units)
├── Batch Normalization
└── Dropout (30%)

↓

LSTM Layer 2 (48 units)
├── Batch Normalization
└── Dropout (30%)

↓

Dense Layer 1 (32 units, ReLU, L2 reg)
└── Dropout (25%)

↓

Dense Layer 2 (16 units, ReLU, L2 reg)
└── Dropout (20%)

↓

Output: Tomorrow's % price change
```

**Key Features:**
- **Time Steps:** 30 days lookback
- **Features:** 17 technical indicators (scale-invariant)
- **Target:** Percentage change (not absolute price)
- **Loss Function:** Mean Absolute Error (MAE)
- **Optimizer:** Adam (LR: 0.0001, gradient clipping)
- **Regularization:** Dropout + L2 + Early stopping
- **Bias Correction:** Systematic bias removed post-training

## 📊 Technical Indicators (17 Features)

| Indicator | Description | Type |
|-----------|-------------|------|
| RSI_14 | Relative Strength Index (14-day) | Momentum |
| RSI_7 | Relative Strength Index (7-day) | Momentum |
| Close_to_SMA7 | Price vs 7-day moving average | Trend |
| Close_to_SMA20 | Price vs 20-day moving average | Trend |
| Close_to_SMA50 | Price vs 50-day moving average | Trend |
| SMA_Cross | 7-day vs 20-day MA crossover | Trend |
| ROC_5 | 5-day rate of change | Momentum |
| ROC_10 | 10-day rate of change | Momentum |
| MACD_Hist | MACD histogram | Momentum |
| BB_pctB | Bollinger Bands %B | Volatility |
| BB_width | Bollinger Bands width | Volatility |
| Volatility_7 | 7-day volatility (CV) | Volatility |
| Volatility_20 | 20-day volatility (CV) | Volatility |
| Volume_Ratio | Volume vs 20-day average | Volume |
| Pct_Change_1d | 1-day percentage change | Return |
| Pct_Change_3d | 3-day percentage change | Return |
| HL_Pct | High-Low percentage | Volatility |

## 🎯 Performance Metrics

### Model Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **RMSE** | Root Mean Squared Error (USD) | < 5% of price |
| **MAE** | Mean Absolute Error (USD) | < 3% of price |
| **MAPE** | Mean Absolute Percentage Error | < 7% |
| **Direction Accuracy** | % correct up/down predictions | > 60% |

### Confidence Levels

- **HIGH:** Direction accuracy > 70% → Reliable signals
- **MODERATE:** Direction accuracy 60-70% → Use caution
- **LOW:** Direction accuracy < 60% → High uncertainty

### Trading Signals

| Signal | Condition | Action |
|--------|-----------|--------|
| **STRONG BUY** | +3%+ predicted, HIGH confidence | Enter long position |
| **BUY** | +1.5%+ predicted, MODERATE confidence | Small long position |
| **HOLD** | ±0.5% predicted | Maintain positions |
| **SELL** | -1.5%- predicted, MODERATE confidence | Reduce exposure |
| **STRONG SELL** | -3%- predicted, HIGH confidence | Exit long / Short |

## 💾 Model Caching System

### How It Works

1. **First Run (New Coin):**
   - Downloads historical data
   - Trains LSTM model (5-10 min)
   - Saves model to `models/COIN_model/`
   - Saves metadata to `models/COIN_metadata.rds`

2. **Subsequent Runs:**
   - Loads saved model (<1 sec)
   - Downloads fresh data
   - Makes instant prediction (<5 sec)
   - **300-600x faster than re-training!**

### Cache Management

```r
# List all trained models
list.files("models", pattern = "_metadata.rds")

# Delete a specific model (force retrain)
unlink("models/BTC-USD_model", recursive = TRUE)
file.remove("models/BTC-USD_metadata.rds")

# Clear all models
unlink("models", recursive = TRUE)
```

## ⚙️ Configuration Options

### In the App Sidebar:

- **Use cached model:** Toggle model caching (default: ON)
- **Training Epochs:** Max epochs when training (default: 150)
- **Forecast Days:** Days to predict ahead (default: 1, max: 30)

### In Code (Advanced):

Edit `crypto_lstm_functions.R`:

```r
# Change time steps (lookback period)
prepare_lstm_data(..., time_steps = 30)  # Default: 30

# Change test size
prepare_lstm_data(..., test_size = 30)  # Default: 30

# Modify model architecture
build_lstm_model() {
  # Change units: 96 → 128, 48 → 64
  layer_lstm(units = 128, ...)
}

# Adjust learning rate
compile(..., optimizer = optimizer_adam(learning_rate = 0.0001))
```

## 🔧 Troubleshooting

### Issue: TensorFlow not installed

**Solution:**
```r
install.packages("keras3")
library(keras3)
install_keras()
```

### Issue: "Failed to download data"

**Causes:**
- No internet connection
- Yahoo Finance API down
- Invalid ticker symbol

**Solution:**
- Check internet connection
- Try different cryptocurrency
- Wait and retry later

### Issue: Model training stuck/slow

**Causes:**
- No GPU available (CPU training is slow)
- Insufficient RAM
- Too many epochs

**Solutions:**
- Reduce epochs to 50-100
- Close other applications
- Upgrade RAM (min 8GB recommended)
- Install GPU support (optional, advanced)

### Issue: Prediction errors are NA

**Causes:**
- Not enough data
- Feature calculation failed
- Model didn't converge

**Solutions:**
- Use crypto with longer history (e.g., Bitcoin)
- Check training log for errors
- Increase epochs or adjust learning rate

### Issue: App won't load

**Solution:**
```r
# Check all packages installed
required_packages <- c("shiny", "shinydashboard", "keras3", "quantmod", "TTR", "plotly", "DT")
missing <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing) > 0) {
  install.packages(missing)
}
```

## 📚 Understanding the Output

### Value Boxes (Top)

- **Current Price:** Latest closing price
- **Tomorrow's Price:** Model prediction
- **Expected Change:** Predicted % change
- **Signal:** Trading recommendation

### Forecast Details

- **Date:** Tomorrow's date
- **Prices:** Current vs predicted
- **Signal:** Buy/Sell/Hold
- **Confidence:** HIGH/MODERATE/LOW
- **Bias Correction:** Systematic bias removed

### Performance Metrics

- **RMSE:** Average error magnitude
- **MAE:** Average absolute error
- **MAPE:** Average percentage error
- **Direction Accuracy:** Hit rate for up/down

### Prediction Visualization

- **Blue line:** Actual prices (test set)
- **Red dashed line:** Model predictions
- **X-axis:** Date
- **Y-axis:** Price in USD

### Training History

- **Blue line:** Training loss
- **Red line:** Validation loss
- **Y-axis:** Log scale MAE
- **Early stopping:** When validation stops improving

### Trading Recommendation

- **Signal:** Current trading action
- **Advice:** Contextual recommendation
- **Risk Note:** Confidence level explanation
- **Disclaimer:** Legal warning

## ⚠️ Important Disclaimers

### This is NOT Financial Advice

- This tool is for **educational purposes only**
- Cryptocurrency trading is **highly risky**
- Past performance does **not** guarantee future results
- **Always do your own research** (DYOR)
- **Never invest more than you can afford to lose**

### Model Limitations

- Predictions are probabilistic, not certain
- Black swan events cannot be predicted
- Market manipulation affects accuracy
- News/sentiment not included (only technical)
- Extreme volatility reduces reliability

### Legal

- Not a licensed financial advisor
- Not responsible for trading losses
- Use at your own risk
- Consult professional advisor before trading

## 🤝 Contributing

Improvements welcome! Areas to enhance:

- [ ] Add more cryptocurrencies
- [ ] Include sentiment analysis (Twitter, Reddit)
- [ ] Multi-step forecasting (predict 7 days at once)
- [ ] Ensemble methods (combine multiple models)
- [ ] Attention mechanisms (Transformer)
- [ ] Portfolio optimization
- [ ] Backtesting module
- [ ] Alert system (email/SMS when signal changes)

## 📄 License

MIT License - Free to use, modify, and distribute.

## 🙏 Acknowledgments

- **Data:** Yahoo Finance (quantmod)
- **Deep Learning:** Keras3 + TensorFlow
- **Technical Analysis:** TTR package
- **UI Framework:** Shiny + shinydashboard

## 📞 Support

For issues or questions:
1. Check this README
2. Review error messages in R console
3. Check training log in app
4. Verify all packages installed

---

**Built with ❤️ using R, Shiny, and Keras**

*Last updated: October 2025 | Version 2.0*
