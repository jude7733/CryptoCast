# ============================================

library(keras3)
library(quantmod)
library(TTR)
library(dplyr)
library(ggplot2)
library(Metrics)

# Set seeds
set_random_seed(123)
tensorflow::set_random_seed(123)

# ============================================
# 1. DATA LOADING AND PREPARATION

#' Download cryptocurrency data from Yahoo Finance
#' @param symbol Crypto symbol (e.g., "BTC-USD")
#' @param from_date Start date (default "2019-01-01")
#' @return data.frame with OHLCV data
get_crypto_data <- function(symbol, from_date = "2019-01-01") {
  tryCatch({
    raw_data <- getSymbols(symbol, src = "yahoo", from = from_date, auto.assign = FALSE)
    
    df <- data.frame(
      Date = index(raw_data),
      Open = as.numeric(raw_data[,1]),
      High = as.numeric(raw_data[,2]),
      Low = as.numeric(raw_data[,3]),
      Close = as.numeric(raw_data[,4]),
      Volume = as.numeric(raw_data[,5])
    )
    
    return(na.omit(df))
    
  }, error = function(e) {
    message("Error downloading data: ", e$message)
    return(NULL)
  })
}

# ============================================
# 2. FEATURE ENGINEERING

#' Add technical indicators to crypto data
#' @param df Data frame with OHLCV columns
#' @return Data frame with added features
add_technical_features <- function(df) {
  
  # Basic technical indicators
  df$RSI_14 <- RSI(df$Close, n = 14)
  df$RSI_7 <- RSI(df$Close, n = 7)
  
  df$SMA_7 <- SMA(df$Close, n = 7)
  df$SMA_20 <- SMA(df$Close, n = 20)
  df$SMA_50 <- SMA(df$Close, n = 50)
  df$EMA_12 <- EMA(df$Close, n = 12)
  df$EMA_26 <- EMA(df$Close, n = 26)
  
  # Momentum indicators
  df$ROC_5 <- ROC(df$Close, n = 5, type = "discrete") * 100
  df$ROC_10 <- ROC(df$Close, n = 10, type = "discrete") * 100
  
  # Price position features (scale-invariant)
  df$Close_to_SMA7 <- (df$Close / df$SMA_7 - 1) * 100
  df$Close_to_SMA20 <- (df$Close / df$SMA_20 - 1) * 100
  df$Close_to_SMA50 <- (df$Close / df$SMA_50 - 1) * 100
  df$SMA_Cross <- (df$SMA_7 / df$SMA_20 - 1) * 100
  
  # MACD
  macd <- MACD(df$Close, nFast = 12, nSlow = 26, nSig = 9)
  df$MACD <- macd[, "macd"]
  df$MACD_Signal <- macd[, "signal"]
  df$MACD_Hist <- df$MACD - df$MACD_Signal
  
  # Bollinger Bands
  bb <- BBands(df$Close, n = 20, sd = 2)
  df$BB_pctB <- (df$Close - bb[,"dn"]) / (bb[,"up"] - bb[,"dn"])
  df$BB_width <- (bb[,"up"] - bb[,"dn"]) / df$Close
  
  # Volatility
  df$Volatility_7 <- rollapply(df$Close, width = 7, 
                                FUN = function(x) sd(x) / mean(x),
                                align = "right", fill = NA)
  df$Volatility_20 <- rollapply(df$Close, width = 20,
                                 FUN = function(x) sd(x) / mean(x),
                                 align = "right", fill = NA)
  
  # Volume
  df$Volume_SMA <- SMA(df$Volume, n = 20)
  df$Volume_Ratio <- df$Volume / df$Volume_SMA
  
  # Price changes
  df$Pct_Change_1d <- c(0, diff(df$Close) / df$Close[-length(df$Close)] * 100)
  df$Pct_Change_3d <- c(rep(0, 3), 
                        (df$Close[4:length(df$Close)] - df$Close[1:(length(df$Close)-3)]) /
                          df$Close[1:(length(df$Close)-3)] * 100)
  
  df$HL_Pct <- (df$High - df$Low) / df$Close * 100
  
  # Target
  df$Tomorrow_Close <- c(df$Close[-1], NA)
  df$Target_Pct_Change <- ((df$Tomorrow_Close - df$Close) / df$Close) * 100
  
  df <- na.omit(df)
  return(df)
}

# ============================================
# 3. DATA PREPROCESSING

#' Prepare data for LSTM training
#' @param df Data frame with features
#' @param feature_cols Vector of feature column names
#' @param time_steps Lookback period (default 30)
#' @return List with train/test data and scaling params
prepare_lstm_data <- function(df, feature_cols, time_steps = 30, test_size = 30) {
  
  feature_data <- as.matrix(df[, feature_cols])
  target_data <- df$Target_Pct_Change
  
  # Calculate scaling parameters
  scaling_params <- data.frame(
    feature = feature_cols,
    mean = apply(feature_data, 2, mean, na.rm = TRUE),
    sd = apply(feature_data, 2, sd, na.rm = TRUE)
  )
  
  # Standardize features
  scaled_features <- matrix(0, nrow = nrow(feature_data), ncol = ncol(feature_data))
  for (i in 1:ncol(feature_data)) {
    scaled_features[, i] <- (feature_data[, i] - scaling_params$mean[i]) / 
      (scaling_params$sd[i] + 1e-8)
  }
  
  # Clip extreme values
  scaled_features[scaled_features > 4] <- 4
  scaled_features[scaled_features < -4] <- -4
  scaled_features[is.na(scaled_features)] <- 0
  scaled_features[is.infinite(scaled_features)] <- 0
  
  # Normalize target
  target_mean <- mean(target_data, na.rm = TRUE)
  target_sd <- sd(target_data, na.rm = TRUE)
  scaled_target <- (target_data - target_mean) / target_sd
  
  # Create sequences
  X <- list()
  y <- list()
  
  for (i in time_steps:(length(scaled_target))) {
    X[[length(X) + 1]] <- scaled_features[(i - time_steps + 1):i, ]
    y[[length(y) + 1]] <- scaled_target[i]
  }
  
  X_full <- array(unlist(X), dim = c(length(X), time_steps, ncol(scaled_features)))
  y_full <- array(unlist(y), dim = c(length(y), 1))
  
  # Train-test split
  train_size <- dim(X_full)[1] - test_size
  
  X_train <- X_full[1:train_size, , ]
  y_train <- y_full[1:train_size, ]
  X_test <- X_full[(train_size + 1):dim(X_full)[1], , ]
  y_test <- y_full[(train_size + 1):dim(X_full)[1], ]
  
  return(list(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test,
    scaling_params = scaling_params,
    target_mean = target_mean,
    target_sd = target_sd,
    scaled_features = scaled_features,
    train_size = train_size,
    time_steps = time_steps
  ))
}

# ============================================
# 4. MODEL BUILDING

#' Build LSTM model architecture
#' @param time_steps Lookback period
#' @param n_features Number of input features
#' @return Compiled Keras model
build_lstm_model <- function(time_steps, n_features) {
  
  model <- keras_model_sequential(name = "Crypto_LSTM") %>%
    layer_lstm(units = 96, return_sequences = TRUE, recurrent_dropout = 0.2,
               input_shape = c(time_steps, n_features)) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.3) %>%
    layer_lstm(units = 48, return_sequences = FALSE, recurrent_dropout = 0.2) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 32, activation = 'relu', kernel_regularizer = regularizer_l2(0.01)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_dense(units = 16, activation = 'relu', kernel_regularizer = regularizer_l2(0.01)) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = 'linear')
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.0001, clipnorm = 1.0),
    loss = 'mean_absolute_error',
    metrics = list('mean_squared_error')
  )
  
  return(model)
}

# ============================================
# 5. MODEL TRAINING

#' Train LSTM model
#' @param model Keras model
#' @param X_train Training features
#' @param y_train Training targets
#' @param epochs Maximum epochs (default 150)
#' @param batch_size Batch size (default 32)
#' @return Training history
train_lstm_model <- function(model, X_train, y_train, epochs = 150, batch_size = 32) {
  
  early_stop <- callback_early_stopping(
    monitor = 'val_loss',
    patience = 30,
    restore_best_weights = TRUE,
    verbose = 0
  )
  
  reduce_lr <- callback_reduce_lr_on_plateau(
    monitor = 'val_loss',
    factor = 0.5,
    patience = 12,
    min_lr = 0.000001,
    verbose = 0
  )
  
  history <- model %>% fit(
    X_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.2,
    callbacks = list(early_stop, reduce_lr),
    verbose = 0
  )
  
  return(history)
}

# ============================================
# 6. PREDICTION AND EVALUATION

#' Calculate directional accuracy
calculate_direction_accuracy <- function(predicted, actual, current) {
  if (is.null(predicted) || is.null(actual) || is.null(current)) return(50)
  if (length(predicted) < 3 || length(actual) < 3 || length(current) < 3) return(50)
  
  valid_mask <- !is.na(predicted) & !is.na(actual) & !is.na(current)
  if (sum(valid_mask, na.rm = TRUE) < 3) return(50)
  
  predicted <- predicted[valid_mask]
  actual <- actual[valid_mask]
  current <- current[valid_mask]
  
  pred_change <- predicted - current
  actual_change <- actual - current
  
  threshold <- 0.005 * median(current, na.rm = TRUE)
  
  pred_direction <- ifelse(abs(pred_change) < threshold, 0, ifelse(pred_change > 0, 1, -1))
  actual_direction <- ifelse(abs(actual_change) < threshold, 0, ifelse(actual_change > 0, 1, -1))
  
  valid_indices <- pred_direction != 0
  num_valid <- sum(valid_indices, na.rm = TRUE)
  
  if (is.na(num_valid) || num_valid == 0) return(50)
  
  correct <- sum(pred_direction[valid_indices] == actual_direction[valid_indices], na.rm = TRUE)
  if (is.na(correct)) return(50)
  
  accuracy <- (correct / num_valid) * 100
  if (is.na(accuracy) || !is.finite(accuracy)) return(50)
  
  return(accuracy)
}

#' Evaluate model performance
#' @param model Trained Keras model
#' @param data_prep Prepared data list
#' @param crypto_df Original crypto data frame
#' @return List with metrics and predictions
evaluate_model <- function(model, data_prep, crypto_df) {
  
  # Unpack data
  X_train <- data_prep$X_train
  y_train <- data_prep$y_train
  X_test <- data_prep$X_test
  y_test <- data_prep$y_test
  target_mean <- data_prep$target_mean
  target_sd <- data_prep$target_sd
  train_size <- data_prep$train_size
  time_steps <- data_prep$time_steps
  
  # Predict
  y_pred_scaled <- model %>% predict(X_test, verbose = 0)
  y_pred_pct <- as.vector(y_pred_scaled * target_sd + target_mean)
  
  # Bias correction
  train_pred_scaled <- model %>% predict(X_train, verbose = 0)
  train_pred_pct <- as.vector(train_pred_scaled * target_sd + target_mean)
  train_actual_pct <- as.vector(y_train * target_sd + target_mean)
  systematic_bias <- mean(train_pred_pct - train_actual_pct, na.rm = TRUE)
  
  y_pred_pct_corrected <- y_pred_pct - systematic_bias
  
  # Get prices
  test_size <- dim(X_test)[1]
  test_start_idx <- train_size + time_steps + 1
  test_end_idx <- min(test_start_idx + test_size - 1, nrow(crypto_df))
  
  test_dates <- crypto_df$Date[test_start_idx:test_end_idx]
  test_prices_today <- crypto_df$Close[test_start_idx:test_end_idx]
  test_prices_tomorrow_actual <- crypto_df$Close[(test_start_idx + 1):(test_end_idx + 1)]
  
  # Ensure equal lengths
  min_len <- min(length(test_prices_today), length(test_prices_tomorrow_actual), length(y_pred_pct_corrected))
  test_prices_today <- test_prices_today[1:min_len]
  test_prices_tomorrow_actual <- test_prices_tomorrow_actual[1:min_len]
  y_pred_pct_corrected <- y_pred_pct_corrected[1:min_len]
  test_dates <- test_dates[1:min_len]
  
  # Calculate predicted prices
  test_prices_tomorrow_pred <- test_prices_today * (1 + y_pred_pct_corrected / 100)
  
  # Apply constraints
  for (i in 1:length(test_prices_tomorrow_pred)) {
    if (!is.na(test_prices_today[i]) && !is.na(test_prices_tomorrow_pred[i])) {
      test_prices_tomorrow_pred[i] <- max(test_prices_today[i] * 0.90, 
                                          min(test_prices_today[i] * 1.10, test_prices_tomorrow_pred[i]))
    }
  }
  
  # Metrics
  valid_indices <- !is.na(test_prices_tomorrow_actual) & !is.na(test_prices_tomorrow_pred)
  
  if (sum(valid_indices) > 3) {
    test_rmse <- sqrt(mean((test_prices_tomorrow_actual[valid_indices] - 
                             test_prices_tomorrow_pred[valid_indices])^2))
    test_mae <- mean(abs(test_prices_tomorrow_actual[valid_indices] - 
                          test_prices_tomorrow_pred[valid_indices]))
    test_mape <- mean(abs((test_prices_tomorrow_actual[valid_indices] - 
                            test_prices_tomorrow_pred[valid_indices]) / 
                           test_prices_tomorrow_actual[valid_indices])) * 100
    
    direction_accuracy <- calculate_direction_accuracy(
      test_prices_tomorrow_pred[valid_indices],
      test_prices_tomorrow_actual[valid_indices],
      test_prices_today[valid_indices]
    )
  } else {
    test_rmse <- test_mae <- test_mape <- NA
    direction_accuracy <- 50
  }
  
  return(list(
    test_rmse = test_rmse,
    test_mae = test_mae,
    test_mape = test_mape,
    direction_accuracy = direction_accuracy,
    test_dates = test_dates,
    test_actual = test_prices_tomorrow_actual,
    test_pred = test_prices_tomorrow_pred,
    systematic_bias = systematic_bias
  ))
}

#' Predict tomorrow's price
#' @param model Trained Keras model
#' @param data_prep Prepared data list
#' @param crypto_df Original crypto data frame
#' @param evaluation Evaluation results
#' @return List with tomorrow's prediction
predict_tomorrow <- function(model, data_prep, crypto_df, evaluation) {
  
  scaled_features <- data_prep$scaled_features
  time_steps <- data_prep$time_steps
  target_mean <- data_prep$target_mean
  target_sd <- data_prep$target_sd
  systematic_bias <- evaluation$systematic_bias
  direction_accuracy <- evaluation$direction_accuracy
  test_mae <- evaluation$test_mae
  
  # Get latest sequence
  latest_features <- tail(scaled_features, time_steps)
  latest_sequence <- array(latest_features, dim = c(1, time_steps, ncol(scaled_features)))
  
  latest_date <- tail(crypto_df$Date, 1)
  current_price <- tail(crypto_df$Close, 1)
  
  # Predict
  tomorrow_pct_scaled <- model %>% predict(latest_sequence, verbose = 0)
  tomorrow_pct_raw <- as.numeric(tomorrow_pct_scaled) * target_sd + target_mean
  tomorrow_pct_corrected <- tomorrow_pct_raw - systematic_bias
  
  tomorrow_price_raw <- current_price * (1 + tomorrow_pct_corrected / 100)
  tomorrow_price <- max(current_price * 0.90, min(current_price * 1.10, tomorrow_price_raw))
  
  price_change <- tomorrow_price - current_price
  pct_change <- (price_change / current_price) * 100
  
  # Confidence
  confidence <- if (is.na(direction_accuracy)) {
    "LOW"
  } else if (direction_accuracy > 70) {
    "HIGH"
  } else if (direction_accuracy > 60) {
    "MODERATE"
  } else {
    "LOW"
  }
  
  # Signal
  signal <- if (abs(pct_change) < 0.5) {
    "HOLD"
  } else if (pct_change > 0) {
    if (pct_change > 3 && confidence == "HIGH") "STRONG BUY" else if (pct_change > 1.5) "BUY" else "WEAK BUY"
  } else {
    if (pct_change < -3 && confidence == "HIGH") "STRONG SELL" else if (pct_change < -1.5) "SELL" else "WEAK SELL"
  }
  
  # Bounds
  lower_bound <- max(tomorrow_price - test_mae, current_price * 0.92)
  upper_bound <- min(tomorrow_price + test_mae, current_price * 1.08)
  
  return(list(
    latest_date = latest_date,
    current_price = current_price,
    tomorrow_price = tomorrow_price,
    price_change = price_change,
    pct_change = pct_change,
    confidence = confidence,
    signal = signal,
    lower_bound = lower_bound,
    upper_bound = upper_bound
  ))
}

# ============================================
# 7. SAVE/LOAD FUNCTIONS

#' Save trained model and metadata
#' @param model Keras model
#' @param data_prep Prepared data
#' @param evaluation Evaluation results
#' @param history Training history
#' @param symbol Crypto symbol
#' @param output_dir Output directory
save_crypto_model <- function(model, data_prep, evaluation, history, symbol, output_dir = "models") {
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  model_path <- file.path(output_dir, paste0(symbol, "_model.keras"))
  save_model(model, model_path, overwrite = TRUE)
  
  # Save metadata
  metadata <- list(
    symbol = symbol,
    trained_date = Sys.time(),
    scaling_params = data_prep$scaling_params,
    target_mean = data_prep$target_mean,
    target_sd = data_prep$target_sd,
    time_steps = data_prep$time_steps,
    systematic_bias = evaluation$systematic_bias,
    test_rmse = evaluation$test_rmse,
    test_mae = evaluation$test_mae,
    test_mape = evaluation$test_mape,
    direction_accuracy = evaluation$direction_accuracy,
    history = history
  )
  
  metadata_path <- file.path(output_dir, paste0(symbol, "_metadata.rds"))
  saveRDS(metadata, metadata_path)
  
  message("✓ Model saved to: ", model_path)
  message("✓ Metadata saved to: ", metadata_path)
}

#' Load trained model and metadata
#' @param symbol Crypto symbol
#' @param model_dir Model directory
#' @return List with model and metadata
load_crypto_model <- function(symbol, model_dir = "models") {
  
  model_path <- file.path(model_dir, paste0(symbol, "_model.keras"))
  metadata_path <- file.path(model_dir, paste0(symbol, "_metadata.rds"))
  
  if (!file(model_path) || !file.exists(metadata_path)) {
    return(NULL)
  }
  
  tryCatch({
    model <- load_model(model_path)
    metadata <- readRDS(metadata_path)
    
    list(model = model, metadata = metadata)
    
  }, error = function(e) {
    message("Error loading model: ", e$message)
    return(NULL)
  })
}


# ============================================
# 8. FEATURE COLUMNS (GLOBAL)

get_feature_columns <- function() {
  c("RSI_14", "RSI_7", "Close_to_SMA7", "Close_to_SMA20", "Close_to_SMA50",
    "SMA_Cross", "ROC_5", "ROC_10", "MACD_Hist", "BB_pctB", "BB_width",
    "Volatility_7", "Volatility_20", "Volume_Ratio", "Pct_Change_1d",
    "Pct_Change_3d", "HL_Pct")
}

message("✓ Crypto LSTM functions loaded successfully!")

