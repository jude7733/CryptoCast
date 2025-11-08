# ============================================
# CONTINUAL LEARNING LSTM FUNCTIONS
# ============================================

library(keras3)
library(tensorflow)
library(quantmod)
library(TTR)
library(dplyr)
library(ggplot2)
library(Metrics)

set_random_seed(123)
tensorflow::set_random_seed(123)

# ============================================
# 1. DATA LOADING
# ============================================

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
# ============================================

add_technical_features <- function(df) {
  
  df$RSI_14 <- RSI(df$Close, n = 14)
  df$RSI_7 <- RSI(df$Close, n = 7)
  
  df$SMA_7 <- SMA(df$Close, n = 7)
  df$SMA_20 <- SMA(df$Close, n = 20)
  df$SMA_50 <- SMA(df$Close, n = 50)
  df$EMA_12 <- EMA(df$Close, n = 12)
  df$EMA_26 <- EMA(df$Close, n = 26)
  
  df$ROC_5 <- ROC(df$Close, n = 5, type = "discrete") * 100
  df$ROC_10 <- ROC(df$Close, n = 10, type = "discrete") * 100
  
  df$Close_to_SMA7 <- (df$Close / df$SMA_7 - 1) * 100
  df$Close_to_SMA20 <- (df$Close / df$SMA_20 - 1) * 100
  df$Close_to_SMA50 <- (df$Close / df$SMA_50 - 1) * 100
  df$SMA_Cross <- (df$SMA_7 / df$SMA_20 - 1) * 100
  
  macd <- MACD(df$Close, nFast = 12, nSlow = 26, nSig = 9)
  df$MACD <- macd[, "macd"]
  df$MACD_Signal <- macd[, "signal"]
  df$MACD_Hist <- df$MACD - df$MACD_Signal
  
  bb <- BBands(df$Close, n = 20, sd = 2)
  df$BB_pctB <- (df$Close - bb[,"dn"]) / (bb[,"up"] - bb[,"dn"])
  df$BB_width <- (bb[,"up"] - bb[,"dn"]) / df$Close
  
  df$Volatility_7 <- rollapply(df$Close, width = 7, 
                               FUN = function(x) sd(x) / mean(x),
                               align = "right", fill = NA)
  df$Volatility_20 <- rollapply(df$Close, width = 20,
                                FUN = function(x) sd(x) / mean(x),
                                align = "right", fill = NA)
  
  df$Volume_SMA <- SMA(df$Volume, n = 20)
  df$Volume_Ratio <- df$Volume / df$Volume_SMA
  
  df$Pct_Change_1d <- c(0, diff(df$Close) / df$Close[-length(df$Close)] * 100)
  df$Pct_Change_3d <- c(rep(0, 3), 
                        (df$Close[4:length(df$Close)] - df$Close[1:(length(df$Close)-3)]) /
                          df$Close[1:(length(df$Close)-3)] * 100)
  
  df$HL_Pct <- (df$High - df$Low) / df$Close * 100
  
  df$Tomorrow_Close <- c(df$Close[-1], NA)
  df$Target_Pct_Change <- ((df$Tomorrow_Close - df$Close) / df$Close) * 100
  
  df <- na.omit(df)
  return(df)
}

# ============================================
# 3. DATA PREPROCESSING
# ============================================

prepare_lstm_data <- function(df, feature_cols, time_steps = 30, test_size = 30) {
  
  feature_data <- as.matrix(df[, feature_cols])
  target_data <- df$Target_Pct_Change
  
  scaling_params <- data.frame(
    feature = feature_cols,
    mean = apply(feature_data, 2, mean, na.rm = TRUE),
    sd = apply(feature_data, 2, sd, na.rm = TRUE)
  )
  
  scaled_features <- matrix(0, nrow = nrow(feature_data), ncol = ncol(feature_data))
  for (i in 1:ncol(feature_data)) {
    scaled_features[, i] <- (feature_data[, i] - scaling_params$mean[i]) / 
      (scaling_params$sd[i] + 1e-8)
  }
  
  scaled_features[scaled_features > 4] <- 4
  scaled_features[scaled_features < -4] <- -4
  scaled_features[is.na(scaled_features)] <- 0
  scaled_features[is.infinite(scaled_features)] <- 0
  
  target_mean <- mean(target_data, na.rm = TRUE)
  target_sd <- sd(target_data, na.rm = TRUE)
  scaled_target <- (target_data - target_mean) / target_sd
  
  X <- list()
  y <- list()
  
  for (i in time_steps:(length(scaled_target))) {
    X[[length(X) + 1]] <- scaled_features[(i - time_steps + 1):i, ]
    y[[length(y) + 1]] <- scaled_target[i]
  }
  
  X_full <- array(unlist(X), dim = c(length(X), time_steps, ncol(scaled_features)))
  y_full <- array(unlist(y), dim = c(length(y), 1))
  
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
# ============================================

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
# 5. CONTINUAL LEARNING - Train with fewer epochs
# ============================================

train_or_continue_model <- function(model = NULL, X_train, y_train, 
                                    epochs = 50, batch_size = 32, 
                                    is_continual = FALSE) {
  
  # Adjust epochs for continual learning
  if (is_continual) {
    epochs <- min(epochs, 30)  # Fewer epochs for fine-tuning
    message("🔄 Continual learning mode: ", epochs, " epochs")
  } else {
    message("🆕 Training from scratch: ", epochs, " epochs")
  }
  
  early_stop <- callback_early_stopping(
    monitor = 'val_loss',
    patience = if (is_continual) 10 else 30,
    restore_best_weights = TRUE,
    verbose = 0
  )
  
  reduce_lr <- callback_reduce_lr_on_plateau(
    monitor = 'val_loss',
    factor = 0.5,
    patience = if (is_continual) 5 else 12,
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
# 6. MULTI-DAY FORECASTING
# ============================================

#' Predict multiple days ahead
#' @param model Trained model
#' @param scaled_features Scaled feature matrix
#' @param time_steps Lookback period
#' @param n_days Number of days to forecast
#' @param target_mean Target mean
#' @param target_sd Target std dev
#' @param systematic_bias Bias correction
#' @param current_price Current price
#' @return Vector of predicted prices
predict_multi_day <- function(model, scaled_features, time_steps, n_days,
                              target_mean, target_sd, systematic_bias, current_price) {
  
  predictions <- numeric(n_days)
  current_seq <- tail(scaled_features, time_steps)
  last_price <- current_price
  
  for (day in 1:n_days) {
    # Predict next day
    seq_array <- array(current_seq, dim = c(1, time_steps, ncol(scaled_features)))
    pred_scaled <- model %>% predict(seq_array, verbose = 0)
    pred_pct <- as.numeric(pred_scaled) * target_sd + target_mean - systematic_bias
    
    # Calculate price
    pred_price <- last_price * (1 + pred_pct / 100)
    pred_price <- max(last_price * 0.90, min(last_price * 1.10, pred_price))
    
    predictions[day] <- pred_price
    
    if (day < n_days) {
      # Shift sequence
      current_seq <- rbind(current_seq[-1, ], current_seq[time_steps, ])
      last_price <- pred_price
    }
  }
  
  return(predictions)
}

# ============================================
# 7. EVALUATION
# ============================================

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

evaluate_model <- function(model, data_prep, crypto_df) {
  
  X_train <- data_prep$X_train
  y_train <- data_prep$y_train
  X_test <- data_prep$X_test
  y_test <- data_prep$y_test
  target_mean <- data_prep$target_mean
  target_sd <- data_prep$target_sd
  train_size <- data_prep$train_size
  time_steps <- data_prep$time_steps
  
  y_pred_scaled <- model %>% predict(X_test, verbose = 0)
  y_pred_pct <- as.vector(y_pred_scaled * target_sd + target_mean)
  
  train_pred_scaled <- model %>% predict(X_train, verbose = 0)
  train_pred_pct <- as.vector(train_pred_scaled * target_sd + target_mean)
  train_actual_pct <- as.vector(y_train * target_sd + target_mean)
  systematic_bias <- mean(train_pred_pct - train_actual_pct, na.rm = TRUE)
  
  y_pred_pct_corrected <- y_pred_pct - systematic_bias
  
  test_size <- dim(X_test)[1]
  test_start_idx <- train_size + time_steps + 1
  test_end_idx <- min(test_start_idx + test_size - 1, nrow(crypto_df))
  
  test_dates <- crypto_df$Date[test_start_idx:test_end_idx]
  test_prices_today <- crypto_df$Close[test_start_idx:test_end_idx]
  test_prices_tomorrow_actual <- crypto_df$Close[(test_start_idx + 1):(test_end_idx + 1)]
  
  min_len <- min(length(test_prices_today), length(test_prices_tomorrow_actual), length(y_pred_pct_corrected))
  test_prices_today <- test_prices_today[1:min_len]
  test_prices_tomorrow_actual <- test_prices_tomorrow_actual[1:min_len]
  y_pred_pct_corrected <- y_pred_pct_corrected[1:min_len]
  test_dates <- test_dates[1:min_len]
  
  test_prices_tomorrow_pred <- test_prices_today * (1 + y_pred_pct_corrected / 100)
  
  for (i in 1:length(test_prices_tomorrow_pred)) {
    if (!is.na(test_prices_today[i]) && !is.na(test_prices_tomorrow_pred[i])) {
      test_prices_tomorrow_pred[i] <- max(test_prices_today[i] * 0.90, 
                                          min(test_prices_today[i] * 1.10, test_prices_tomorrow_pred[i]))
    }
  }
  
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

# ============================================
# 8. SAVE/LOAD
# ============================================

save_crypto_model <- function(model, data_prep, evaluation, history, symbol, output_dir = "models") {
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  model_path <- file.path(output_dir, paste0(symbol, "_model.keras"))
  save_model(model, model_path, overwrite = TRUE)
  
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
    history = history,
    total_epochs_trained = length(history$metrics$loss)
  )
  
  metadata_path <- file.path(output_dir, paste0(symbol, "_metadata.rds"))
  saveRDS(metadata, metadata_path)
  
  message("✓ Model saved: ", model_path)
  message("✓ Metadata saved: ", metadata_path)
}

load_crypto_model <- function(symbol, model_dir = "models") {
  
  model_path <- file.path(model_dir, paste0(symbol, "_model.keras"))
  metadata_path <- file.path(model_dir, paste0(symbol, "_metadata.rds"))
  
  if (!file.exists(model_path) || !file.exists(metadata_path)) {
    return(NULL)
  }
  
  tryCatch({
    model <- load_model(model_path)
    metadata <- readRDS(metadata_path)
    
    message("✓ Loaded model for ", symbol)
    message("  Last trained: ", format(metadata$trained_date, "%Y-%m-%d %H:%M"))
    
    list(model = model, metadata = metadata)
    
  }, error = function(e) {
    message("Error loading model: ", e$message)
    return(NULL)
  })
}

# ============================================
# 9. UTILITY
# ============================================

get_feature_columns <- function() {
  c("RSI_14", "RSI_7", "Close_to_SMA7", "Close_to_SMA20", "Close_to_SMA50",
    "SMA_Cross", "ROC_5", "ROC_10", "MACD_Hist", "BB_pctB", "BB_width",
    "Volatility_7", "Volatility_20", "Volume_Ratio", "Pct_Change_1d",
    "Pct_Change_3d", "HL_Pct")
}

message("✓ Continual Learning LSTM functions loaded!")
