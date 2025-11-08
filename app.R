# ============================================
# CONTINUAL LEARNING CRYPTO DASHBOARD
# With multi-day forecasting and live price chart
# ============================================

library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(shinycssloaders)
library(DT)
library(plotly)

source("lstm_functions.R")

# ============================================
# CONFIGURATION
# ============================================

crypto_list <- c(
  "Bitcoin" = "BTC-USD",
  "Ethereum" = "ETH-USD",
  "Binance Coin" = "BNB-USD",
  "Ripple" = "XRP-USD",
  "Cardano" = "ADA-USD",
  "Solana" = "SOL-USD",
  "Polkadot" = "DOT-USD",
  "Dogecoin" = "DOGE-USD",
  "Avalanche" = "AVAX-USD",
  "Polygon" = "MATIC-USD"
)

MODEL_DIR <- "models"

if (!dir.exists(MODEL_DIR)) dir.create(MODEL_DIR, recursive = TRUE)

# ============================================
# UI
# ============================================

ui <- dashboardPage(
  skin = "blue",
  
  dashboardHeader(title = "Crypto Continual Learning", titleWidth = 320),
  
  dashboardSidebar(
    width = 320,
    sidebarMenu(
      menuItem("Live Price & Forecast", tabName = "prediction", icon = icon("chart-line")),
      menuItem("Model Info", tabName = "training", icon = icon("cogs")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    ),
    
    br(),
    
    div(style = "padding: 15px;",
        
        selectInput("crypto_select", 
                    "📊 Select Cryptocurrency:",
                    choices = crypto_list,
                    selected = "BTC-USD"),
        
        hr(),
        
        sliderInput("forecast_days",
                    "🔮 Forecast Horizon (Days):",
                    min = 1,
                    max = 30,
                    value = 7,
                    step = 1),
        
        sliderInput("epochs",
                    "⚡ Training Epochs:",
                    min = 20,
                    max = 100,
                    value = 50,
                    step = 10),
        
        hr(),
        
        actionButton("train_predict_btn", 
                     "Train & Predict",
                     icon = icon("rocket"),
                     class = "btn-success btn-block",
                     style = "margin-top: 10px; font-size: 16px; padding: 10px;"),
        
        hr(),
        
        p(strong("📍 Status:"), style = "margin-bottom: 5px;"),
        verbatimTextOutput("status_text", placeholder = TRUE),
        
        hr(),
        
        uiOutput("model_info_sidebar")
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper { background-color: #ecf0f5; }
        .box { border-top: 3px solid #3c8dbc; }
        .value-box { font-size: 22px; font-weight: bold; }
        .btn-success { background-color: #28a745; border-color: #28a745; }
        .btn-success:hover { background-color: #218838; }
      "))
    ),
    
    tabItems(
      
      # ============================================
      # PREDICTION TAB
      # ============================================
      
      tabItem(
        tabName = "prediction",
        
        # Live Price Chart (First)
        fluidRow(
          box(
            title = "📈 Live Historical Price Chart",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            withSpinner(plotlyOutput("live_price_plot", height = "450px"))
          )
        ),
        
        # Value Boxes
        fluidRow(
          valueBoxOutput("current_price_box", width = 4),
          valueBoxOutput("forecast_price_box", width = 4),
          valueBoxOutput("change_box", width = 4)
        ),
        
        # Forecast Details and Metrics
        fluidRow(
          box(
            title = "🔮 Multi-Day Forecast",
            status = "primary",
            solidHeader = TRUE,
            width = 6,
            withSpinner(uiOutput("forecast_details"))
          ),
          
          box(
            title = "📊 Model Performance",
            status = "info",
            solidHeader = TRUE,
            width = 6,
            withSpinner(uiOutput("performance_metrics"))
          )
        ),
        
        # Forecast Chart
        fluidRow(
          box(
            title = "📉 Forecast Visualization (Next N Days)",
            status = "success",
            solidHeader = TRUE,
            width = 12,
            withSpinner(plotlyOutput("forecast_plot", height = "400px"))
          )
        ),
        
        # Test Predictions and Trading Advice
        fluidRow(
          box(
            title = "✅ Test Set Predictions",
            status = "primary",
            solidHeader = TRUE,
            width = 7,
            withSpinner(plotlyOutput("test_plot", height = "350px"))
          ),
          
          box(
            title = "💡 Trading Recommendation",
            status = "warning",
            solidHeader = TRUE,
            width = 5,
            withSpinner(uiOutput("trading_advice"))
          )
        ),
        
        # Training History
        fluidRow(
          box(
            title = "📚 Training History",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            withSpinner(plotlyOutput("training_plot", height = "300px"))
          )
        )
      ),
      
      # ============================================
      # TRAINING TAB
      # ============================================
      
      tabItem(
        tabName = "training",
        
        fluidRow(
          box(
            title = "💾 Saved Models",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            withSpinner(DTOutput("models_table"))
          )
        ),
        
        fluidRow(
          box(
            title = "📝 Training Log",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            verbatimTextOutput("training_log")
          )
        )
      ),
      
      # ============================================
      # ABOUT TAB
      # ============================================
      
      tabItem(
        tabName = "about",
        box(
          title = "About This Application",
          status = "primary",
          solidHeader = TRUE,
          width = 12,
          HTML("
            <h3>🚀 Continual Learning Cryptocurrency Forecaster</h3>
            <p>Advanced LSTM-based cryptocurrency price forecasting with <strong>continual learning</strong>.</p>
            
            <h4>✨ Key Features:</h4>
            <ul>
              <li><strong>Continual Learning:</strong> Loads existing model and continues training with new data</li>
              <li><strong>Multi-Day Forecasting:</strong> Predict 1-30 days into the future</li>
              <li><strong>Live Price Chart:</strong> View complete historical data before forecasting</li>
              <li><strong>10 Cryptocurrencies:</strong> Bitcoin, Ethereum, and 8 other major coins</li>
              <li><strong>17 Technical Indicators:</strong> RSI, MACD, Bollinger Bands, volatility, volume, etc.</li>
              <li><strong>Faster Training:</strong> Continual learning requires fewer epochs (~30-50 vs 150)</li>
            </ul>
            
            <h4>🔄 How Continual Learning Works:</h4>
            <ol>
              <li><strong>First Time:</strong> Builds new model and trains from scratch (~10 min)</li>
              <li><strong>Subsequent Runs:</strong> Loads saved model and continues training with new data (~2-3 min)</li>
              <li><strong>Auto-Detection:</strong> System automatically detects if a model exists</li>
            </ol>
            
            <h4>📖 How to Use:</h4>
            <ol>
              <li>Select cryptocurrency from dropdown</li>
              <li>Choose forecast horizon (1-30 days)</li>
              <li>Adjust training epochs (30-50 recommended)</li>
              <li>Click <strong>'Train & Predict'</strong></li>
              <li>View live price chart, forecast, and trading signals</li>
            </ol>
            
            <h4>⏱️ Training Time:</h4>
            <ul>
              <li><strong>First Training:</strong> 8-12 minutes</li>
              <li><strong>Continual Learning:</strong> 2-4 minutes</li>
              <li><strong>Time Savings:</strong> ~70% faster with continual learning!</li>
            </ul>
            
            <h4>⚠️ Disclaimer:</h4>
            <div class='alert alert-danger'>
              <strong>NOT FINANCIAL ADVICE!</strong> For educational purposes only.
            </div>
            
            <hr>
            <p><em>Built with R Shiny, Keras3, TensorFlow, and quantmod</em></p>
            <p><small>Version 3.0 - Continual Learning Edition | November 2025</small></p>
          ")
        )
      )
    )
  )
)

# ============================================
# SERVER (FIXED - No infinite loops!)
# ============================================

server <- function(input, output, session) {
  
  # Reactive values
  prediction_results <- reactiveVal(NULL)
  training_log_text <- reactiveVal("Ready to train.\n")
  live_data <- reactiveVal(NULL)
  
  # Append log
  append_log <- function(text) {
    current_log <- training_log_text()
    training_log_text(paste0(current_log, text, "\n"))
  }
  
  
  observeEvent(input$crypto_select, {
    symbol <- input$crypto_select
    crypto_name <- names(crypto_list)[crypto_list == symbol]
    
    # Show loading message
    append_log(paste0("📊 Loading data for ", crypto_name, "..."))
    
    tryCatch({
      data <- get_crypto_data(symbol)
      
      if (!is.null(data)) {
        live_data(data)
        append_log(paste0("✓ Loaded ", nrow(data), " days of data"))
      } else {
        append_log("✗ Failed to load data")
      }
    }, error = function(e) {
      append_log(paste0("ERROR: ", e$message))
    })
  }, ignoreInit = FALSE)
  
  # Status text
  output$status_text <- renderText({
    results <- prediction_results()
    symbol <- input$crypto_select
    model_path <- file.path(MODEL_DIR, paste0(symbol, "_model.keras"))
    
    if (is.null(results)) {
      if (file.exists(model_path)) {
        "✅ Model exists\n🔄 Ready for continual learning"
      } else {
        "🆕 No model found\n⚡ Will train from scratch"
      }
    } else {
      paste0("✅ Last trained:\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
    }
  })
  
  # Model info sidebar
  output$model_info_sidebar <- renderUI({
    symbol <- input$crypto_select
    model_path <- file.path(MODEL_DIR, paste0(symbol, "_model.keras"))
    metadata_path <- file.path(MODEL_DIR, paste0(symbol, "_metadata.rds"))
    
    if (file.exists(model_path) && file.exists(metadata_path)) {
      tryCatch({
        meta <- readRDS(metadata_path)
        
        tags$div(
          class = "alert alert-success",
          style = "padding: 10px; font-size: 13px;",
          icon("check-circle"),
          strong(" Saved Model Found"),
          br(),
          "Trained: ", format(meta$trained_date, "%Y-%m-%d"),
          br(),
          "MAPE: ", round(meta$test_mape, 2), "%",
          br(),
          "Dir. Acc: ", round(meta$direction_accuracy, 1), "%"
        )
      }, error = function(e) {
        tags$div(
          class = "alert alert-warning",
          style = "padding: 10px; font-size: 13px;",
          icon("exclamation-triangle"),
          "Model metadata corrupted"
        )
      })
    } else {
      tags$div(
        class = "alert alert-info",
        style = "padding: 10px; font-size: 13px;",
        icon("info-circle"),
        strong(" No Model Yet"),
        br(),
        "Will train from scratch"
      )
    }
  })
  
  # ============================================
  # TRAIN & PREDICT (MAIN LOGIC)
  # ============================================
  
  observeEvent(input$train_predict_btn, {
    
    symbol <- input$crypto_select
    crypto_name <- names(crypto_list)[crypto_list == symbol]
    n_days <- input$forecast_days
    epochs <- input$epochs
    
    append_log(paste0("\n========== TRAINING: ", crypto_name, " =========="))
    append_log(paste0("Forecast: ", n_days, " days | Epochs: ", epochs))
    append_log(paste0("Time: ", Sys.time()))
    
    progress <- Progress$new(session, min = 0, max = 1)
    on.exit(progress$close())
    
    # Check if model exists
    model_path <- file.path(MODEL_DIR, paste0(symbol, "_model.keras"))
    model_exists <- file.exists(model_path)
    
    is_continual <- model_exists
    
    if (is_continual) {
      append_log("🔄 CONTINUAL LEARNING MODE")
      append_log("  Loading existing model...")
    } else {
      append_log("🆕 TRAINING FROM SCRATCH")
    }
    
    # Download data
    progress$set(value = 0.05, message = "Downloading data...")
    append_log("Downloading latest data...")
    
    crypto_df <- get_crypto_data(symbol)
    
    if (is.null(crypto_df)) {
      showNotification("Failed to download data!", type = "error")
      append_log("ERROR: Failed to download data")
      return()
    }
    
    append_log(paste0("✓ Downloaded ", nrow(crypto_df), " days"))
    
    # Add features
    progress$set(value = 0.1, message = "Calculating indicators...")
    append_log("Calculating 17 technical indicators...")
    
    crypto_df <- add_technical_features(crypto_df)
    append_log(paste0("✓ Clean data: ", nrow(crypto_df), " rows"))
    
    # Prepare data
    progress$set(value = 0.15, message = "Preparing sequences...")
    append_log("Preparing LSTM sequences...")
    
    feature_cols <- get_feature_columns()
    data_prep <- prepare_lstm_data(crypto_df, feature_cols, time_steps = 30, test_size = 30)
    
    append_log(paste0("✓ Train samples: ", dim(data_prep$X_train)[1]))
    append_log(paste0("✓ Test samples: ", dim(data_prep$X_test)[1]))
    
    # Build or load model
    progress$set(value = 0.2, message = "Loading/building model...")
    
    if (is_continual) {
      append_log("Loading saved model...")
      loaded <- load_crypto_model(symbol, MODEL_DIR)
      
      if (!is.null(loaded)) {
        model <- loaded$model
        old_meta <- loaded$metadata
        append_log(paste0("✓ Model loaded (last trained: ", format(old_meta$trained_date, "%Y-%m-%d"), ")"))
      } else {
        append_log("✗ Failed to load model, training from scratch")
        model <- build_lstm_model(data_prep$time_steps, length(feature_cols))
        is_continual <- FALSE
      }
    } else {
      append_log("Building new LSTM model...")
      model <- build_lstm_model(data_prep$time_steps, length(feature_cols))
      append_log("✓ Model built: 2 LSTM + dense layers")
    }
    
    # Train model
    if (is_continual) {
      progress$set(value = 0.25, message = "Continual learning (2-4 min)...")
      append_log(paste0("Fine-tuning with ", epochs, " epochs..."))
      append_log("(Estimated: 2-4 minutes)")
    } else {
      progress$set(value = 0.25, message = "Training from scratch (8-12 min)...")
      append_log(paste0("Training ", epochs, " epochs..."))
      append_log("(Estimated: 8-12 minutes)")
    }
    
    start_time <- Sys.time()
    
    history <- train_or_continue_model(
      model = model,
      X_train = data_prep$X_train,
      y_train = data_prep$y_train,
      epochs = epochs,
      batch_size = 32,
      is_continual = is_continual
    )
    
    end_time <- Sys.time()
    training_time <- difftime(end_time, start_time, units = "mins")
    
    append_log(paste0("✓ Training complete in ", round(training_time, 2), " min"))
    append_log(paste0("  Epochs: ", length(history$metrics$loss)))
    append_log(paste0("  Final loss: ", round(tail(history$metrics$loss, 1), 6)))
    
    # Evaluate
    progress$set(value = 0.9, message = "Evaluating...")
    append_log("Evaluating performance...")
    
    evaluation <- evaluate_model(model, data_prep, crypto_df)
    
    append_log("✓ Metrics:")
    append_log(paste0("  RMSE: $", round(evaluation$test_rmse, 2)))
    append_log(paste0("  MAE: $", round(evaluation$test_mae, 2)))
    append_log(paste0("  MAPE: ", round(evaluation$test_mape, 2), "%"))
    append_log(paste0("  Direction: ", round(evaluation$direction_accuracy, 2), "%"))
    
    # Multi-day forecast
    progress$set(value = 0.95, message = "Forecasting...")
    append_log(paste0("Forecasting next ", n_days, " days..."))
    
    latest_date <- tail(crypto_df$Date, 1)
    current_price <- tail(crypto_df$Close, 1)
    
    forecast_prices <- predict_multi_day(
      model = model,
      scaled_features = data_prep$scaled_features,
      time_steps = data_prep$time_steps,
      n_days = n_days,
      target_mean = data_prep$target_mean,
      target_sd = data_prep$target_sd,
      systematic_bias = evaluation$systematic_bias,
      current_price = current_price
    )
    
    forecast_dates <- seq.Date(latest_date + 1, by = "day", length.out = n_days)
    
    final_price <- forecast_prices[n_days]
    total_change <- final_price - current_price
    total_pct_change <- (total_change / current_price) * 100
    
    # Confidence & Signal
    direction_accuracy <- evaluation$direction_accuracy
    
    confidence <- if (is.na(direction_accuracy)) {
      "LOW"
    } else if (direction_accuracy > 70) {
      "HIGH"
    } else if (direction_accuracy > 60) {
      "MODERATE"
    } else {
      "LOW"
    }
    
    signal <- if (abs(total_pct_change) < 0.5) {
      "HOLD"
    } else if (total_pct_change > 0) {
      if (total_pct_change > 5 && confidence == "HIGH") "STRONG BUY" 
      else if (total_pct_change > 2) "BUY" 
      else "WEAK BUY"
    } else {
      if (total_pct_change < -5 && confidence == "HIGH") "STRONG SELL" 
      else if (total_pct_change < -2) "SELL" 
      else "WEAK SELL"
    }
    
    append_log("✓ Forecast:")
    append_log(paste0("  Day ", n_days, ": $", round(final_price, 2)))
    append_log(paste0("  Total change: ", round(total_pct_change, 2), "%"))
    append_log(paste0("  Signal: ", signal, " (", confidence, ")"))
    
    # Save model
    append_log("Saving model...")
    save_crypto_model(model, data_prep, evaluation, history, symbol, MODEL_DIR)
    append_log("✓ Model saved")
    
    progress$set(value = 1, message = "Done!")
    
    # Store results
    prediction_results(list(
      latest_date = latest_date,
      current_price = current_price,
      forecast_prices = forecast_prices,
      forecast_dates = forecast_dates,
      final_price = final_price,
      total_change = total_change,
      total_pct_change = total_pct_change,
      confidence = confidence,
      signal = signal,
      test_rmse = evaluation$test_rmse,
      test_mae = evaluation$test_mae,
      test_mape = evaluation$test_mape,
      direction_accuracy = direction_accuracy,
      systematic_bias = evaluation$systematic_bias,
      history = history,
      test_dates = evaluation$test_dates,
      test_actual = evaluation$test_actual,
      test_pred = evaluation$test_pred,
      n_days = n_days,
      is_continual = is_continual,
      training_time = training_time
    ))
    
    append_log("\n========== COMPLETE ==========\n")
    
    showNotification(
      paste0("✓ ", crypto_name, " forecast complete! (", round(training_time, 1), " min)"),
      type = "message",
      duration = 5
    )
  })
  
  # ============================================
  # OUTPUTS - Value Boxes
  # ============================================
  
  output$current_price_box <- renderValueBox({
    results <- prediction_results()
    if (is.null(results)) {
      valueBox("--", "Current Price", icon = icon("dollar-sign"), color = "blue")
    } else {
      valueBox(
        paste0("$", format(round(results$current_price, 2), big.mark = ",")),
        paste("Current Price", format(results$latest_date, "(%Y-%m-%d)")),
        icon = icon("dollar-sign"),
        color = "blue"
      )
    }
  })
  
  output$forecast_price_box <- renderValueBox({
    results <- prediction_results()
    if (is.null(results)) {
      valueBox("--", "Forecast Price", icon = icon("crystal-ball"), color = "purple")
    } else {
      valueBox(
        paste0("$", format(round(results$final_price, 2), big.mark = ",")),
        paste0("+", results$n_days, " Days Forecast"),
        icon = icon("crystal-ball"),
        color = "purple"
      )
    }
  })
  
  output$change_box <- renderValueBox({
    results <- prediction_results()
    if (is.null(results)) {
      valueBox("--", "Expected Change", icon = icon("chart-line"), color = "yellow")
    } else {
      color <- if (results$total_pct_change > 0) "green" 
      else if (results$total_pct_change < 0) "red" 
      else "yellow"
      
      icon_name <- if (results$total_pct_change > 0) "arrow-up" 
      else if (results$total_pct_change < 0) "arrow-down" 
      else "minus"
      
      valueBox(
        paste0(ifelse(results$total_pct_change > 0, "+", ""), 
               round(results$total_pct_change, 2), "%"),
        paste0(results$signal, " (", results$confidence, ")"),
        icon = icon(icon_name),
        color = color
      )
    }
  })
  
  # ============================================
  # OUTPUTS - Live Price Chart
  # ============================================
  
  output$live_price_plot <- renderPlotly({
    data <- live_data()
    
    if (is.null(data) || nrow(data) == 0) {
      return(plotly_empty() %>% 
               layout(title = "Waiting for data..."))
    }
    
    plot_ly(data, x = ~Date, y = ~Close, type = "scatter", mode = "lines",
            line = list(color = "#1f77b4", width = 2),
            name = "Close Price") %>%
      add_trace(y = ~High, name = "High", 
                line = list(color = "rgba(76,175,80,0.3)", width = 1, dash = "dot")) %>%
      add_trace(y = ~Low, name = "Low",
                line = list(color = "rgba(244,67,54,0.3)", width = 1, dash = "dot")) %>%
      layout(
        title = list(text = paste0("Historical Price - ", input$crypto_select), font = list(size = 18)),
        xaxis = list(title = "Date"),
        yaxis = list(title = "Price (USD)", tickformat = "$,.0f"),
        hovermode = "x unified",
        legend = list(x = 0.02, y = 0.98)
      )
  })
  
  # ============================================
  # OUTPUTS - Forecast Details
  # ============================================
  
  output$forecast_details <- renderUI({
    results <- prediction_results()
    if (is.null(results)) {
      return(HTML("<p>Click 'Train & Predict' to generate forecast...</p>"))
    }
    
    forecast_table <- data.frame(
      Day = 1:results$n_days,
      Date = format(results$forecast_dates, "%Y-%m-%d"),
      Price = paste0("$", format(round(results$forecast_prices, 2), big.mark = ",")),
      Change = paste0(ifelse(results$forecast_prices > c(results$current_price, results$forecast_prices[-results$n_days]), "+", ""),
                      round((results$forecast_prices - c(results$current_price, results$forecast_prices[-results$n_days])) / 
                              c(results$current_price, results$forecast_prices[-results$n_days]) * 100, 2), "%")
    )
    
    HTML(paste0("
      <div style='max-height: 400px; overflow-y: auto;'>
        <table class='table table-striped table-sm'>
          <thead>
            <tr>
              <th>Day</th>
              <th>Date</th>
              <th>Predicted Price</th>
              <th>Daily Change</th>
            </tr>
          </thead>
          <tbody>",
                paste0(apply(forecast_table, 1, function(row) {
                  paste0("<tr><td>", row[1], "</td><td>", row[2], "</td><td>", row[3], "</td><td>", row[4], "</td></tr>")
                }), collapse = ""),
                "</tbody>
        </table>
      </div>
      <hr>
      <p><strong>Total Forecast Change:</strong> ", round(results$total_pct_change, 2), "%</p>
      <p><strong>Signal:</strong> <span style='font-size: 18px; font-weight: bold;'>", results$signal, "</span></p>
      <p><strong>Confidence:</strong> ", results$confidence, "</p>
      <p><strong>Mode:</strong> ", ifelse(results$is_continual, "🔄 Continual Learning", "🆕 From Scratch"), "</p>
      <p><strong>Training Time:</strong> ", round(results$training_time, 2), " minutes</p>
    "))
  })
  
  # ============================================
  # OUTPUTS - Performance Metrics
  # ============================================
  
  output$performance_metrics <- renderUI({
    results <- prediction_results()
    if (is.null(results)) {
      return(HTML("<p>No metrics available yet...</p>"))
    }
    
    HTML(paste0("
      <table class='table table-bordered'>
        <tr><th>Metric</th><th>Value</th><th>Rating</th></tr>
        <tr><td>RMSE</td><td>$", format(round(results$test_rmse, 2), big.mark = ","), "</td><td>", 
                ifelse(results$test_rmse/results$current_price < 0.05, "✓ Excellent", 
                       ifelse(results$test_rmse/results$current_price < 0.10, "○ Good", "✗ Fair")), "</td></tr>
        <tr><td>MAE</td><td>$", format(round(results$test_mae, 2), big.mark = ","), "</td><td>", 
                ifelse(results$test_mae/results$current_price < 0.03, "✓ Excellent", 
                       ifelse(results$test_mae/results$current_price < 0.06, "○ Good", "✗ Fair")), "</td></tr>
        <tr><td>MAPE</td><td>", round(results$test_mape, 2), "%</td><td>",
                ifelse(results$test_mape < 5, "✓ Excellent", 
                       ifelse(results$test_mape < 10, "○ Good", "✗ Fair")), "</td></tr>
        <tr><td>Direction Acc.</td><td>", round(results$direction_accuracy, 2), "%</td><td>",
                ifelse(results$direction_accuracy > 65, "✓ Excellent", 
                       ifelse(results$direction_accuracy > 55, "○ Good", "✗ Fair")), "</td></tr>
      </table>
      <p style='margin-top: 10px;'><small><strong>Rating:</strong> ✓ Excellent | ○ Good | ✗ Fair</small></p>
    "))
  })
  
  # ============================================
  # OUTPUTS - Forecast Plot
  # ============================================
  
  output$forecast_plot <- renderPlotly({
    results <- prediction_results()
    
    if (is.null(results)) return(NULL)
    
    dates <- c(results$latest_date, results$forecast_dates)
    prices <- c(results$current_price, results$forecast_prices)
    
    df <- data.frame(Date = dates, Price = prices)
    
    plot_ly(df, x = ~Date, y = ~Price, type = "scatter", mode = "lines+markers",
            line = list(color = "#28a745", width = 3),
            marker = list(size = 8, color = "#28a745")) %>%
      layout(
        title = paste0(results$n_days, "-Day Price Forecast"),
        xaxis = list(title = "Date"),
        yaxis = list(title = "Price (USD)", tickformat = "$,.0f"),
        hovermode = "x unified"
      )
  })
  
  # ============================================
  # OUTPUTS - Test Plot
  # ============================================
  
  output$test_plot <- renderPlotly({
    results <- prediction_results()
    
    if (is.null(results) || is.null(results$test_dates)) return(NULL)
    
    df <- data.frame(
      Date = results$test_dates,
      Actual = results$test_actual,
      Predicted = results$test_pred
    )
    
    plot_ly(df, x = ~Date) %>%
      add_trace(y = ~Actual, name = "Actual", type = "scatter", mode = "lines",
                line = list(color = "#2196F3", width = 2)) %>%
      add_trace(y = ~Predicted, name = "Predicted", type = "scatter", mode = "lines",
                line = list(color = "#FF5722", width = 2, dash = "dash")) %>%
      layout(
        title = "Test Set: Actual vs Predicted",
        xaxis = list(title = "Date"),
        yaxis = list(title = "Price (USD)", tickformat = "$,.0f"),
        hovermode = "x unified",
        legend = list(x = 0.02, y = 0.98)
      )
  })
  
  # ============================================
  # OUTPUTS - Trading Advice
  # ============================================
  
  output$trading_advice <- renderUI({
    results <- prediction_results()
    
    if (is.null(results)) {
      return(HTML("<p>Run prediction to get trading advice...</p>"))
    }
    
    signal_color <- if (grepl("BUY", results$signal)) {
      "success"
    } else if (grepl("SELL", results$signal)) {
      "danger"
    } else {
      "warning"
    }
    
    advice <- if (grepl("STRONG BUY", results$signal)) {
      paste0("Strong upward momentum expected over ", results$n_days, " days. Consider entering a long position.")
    } else if (grepl("BUY", results$signal)) {
      paste0("Moderate upward trend predicted. Small long position recommended.")
    } else if (grepl("STRONG SELL", results$signal)) {
      paste0("Strong downward pressure expected. Consider closing long positions.")
    } else if (grepl("SELL", results$signal)) {
      paste0("Moderate downward trend. Reduce exposure or hedge.")
    } else {
      paste0("No clear direction. Hold current positions.")
    }
    
    risk_note <- if (results$confidence == "HIGH") {
      "Confidence is HIGH. Model has strong directional accuracy (>70%)."
    } else if (results$confidence == "MODERATE") {
      "Confidence is MODERATE (60-70%)."
    } else {
      "Confidence is LOW. Use extra caution."
    }
    
    HTML(paste0("
      <div class='alert alert-", signal_color, "'>
        <h4>", results$signal, "</h4>
        <p>", advice, "</p>
        <hr>
        <p><strong>Confidence:</strong> ", risk_note, "</p>
        <p><strong>Forecast Horizon:</strong> ", results$n_days, " days</p>
      </div>
      <div class='alert alert-danger' style='font-size: 12px;'>
        <strong>⚠️ DISCLAIMER:</strong> NOT financial advice. For educational purposes only.
      </div>
    "))
  })
  
  # ============================================
  # OUTPUTS - Training History
  # ============================================
  
  output$training_plot <- renderPlotly({
    results <- prediction_results()
    
    if (is.null(results) || is.null(results$history)) return(NULL)
    
    history <- results$history
    
    df <- data.frame(
      epoch = 1:length(history$metrics$loss),
      train_loss = history$metrics$loss,
      val_loss = history$metrics$val_loss
    )
    
    plot_ly(df, x = ~epoch) %>%
      add_trace(y = ~train_loss, name = "Training Loss", type = "scatter", mode = "lines",
                line = list(color = "#2196F3")) %>%
      add_trace(y = ~val_loss, name = "Validation Loss", type = "scatter", mode = "lines",
                line = list(color = "#FF5722")) %>%
      layout(
        title = "Training History (Latest Run)",
        xaxis = list(title = "Epoch"),
        yaxis = list(title = "Loss (MAE)", type = "log"),
        hovermode = "x unified",
        legend = list(x = 0.7, y = 0.9)
      )
  })
  
  # ============================================
  # OUTPUTS - Models Table
  # ============================================
  
  output$models_table <- renderDT({
    model_files <- list.files(MODEL_DIR, pattern = "_metadata.rds$", full.names = TRUE)
    
    if (length(model_files) == 0) {
      return(data.frame(Message = "No trained models yet"))
    }
    
    models_info <- lapply(model_files, function(f) {
      meta <- readRDS(f)
      data.frame(
        Cryptocurrency = meta$symbol,
        Last_Trained = format(meta$trained_date, "%Y-%m-%d %H:%M"),
        RMSE = round(meta$test_rmse, 2),
        MAPE = paste0(round(meta$test_mape, 2), "%"),
        Direction_Acc = paste0(round(meta$direction_accuracy, 1), "%"),
        Total_Epochs = meta$total_epochs_trained
      )
    })
    
    df <- do.call(rbind, models_info)
    
    datatable(df, options = list(pageLength = 10), rownames = FALSE)
  })
  
  # ============================================
  # OUTPUTS - Training Log
  # ============================================
  
  output$training_log <- renderText({
    training_log_text()
  })
}

# ============================================
# RUN APP
# ============================================

shinyApp(ui = ui, server = server)
