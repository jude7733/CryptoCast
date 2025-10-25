# ============================================
# CRYPTO PRICE PREDICTION DASHBOARD

# Load required libraries
library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(shinycssloaders)
library(DT)
library(plotly)


# Source the functions file
source("lstm_functions.R")

# ============================================
# CONFIGURATION

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

# Create directories
if (!dir.exists(MODEL_DIR)) dir.create(MODEL_DIR, recursive = TRUE)

# ============================================
# UI

ui <- dashboardPage(
  skin = "blue",
  
  dashboardHeader(title = "Crypto Price Predictor", titleWidth = 300),
  
  dashboardSidebar(
    width = 300,
    sidebarMenu(
      menuItem("Prediction", tabName = "prediction", icon = icon("chart-line")),
      menuItem("Model Training", tabName = "training", icon = icon("cogs")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    ),
    
    br(),
    
    div(style = "padding: 15px;",
        selectInput("crypto_select", 
                    "Select Cryptocurrency:",
                    choices = crypto_list,
                    selected = "BTC-USD"),
        
        actionButton("predict_btn", 
                     "Quick Prediction",
                     icon = icon("bolt"),
                     class = "btn-primary btn-block",
                     style = "margin-top: 10px; width: 250px;"),
        
        actionButton("train_btn", 
                     "Train New Model",
                     icon = icon("play"),
                     class = "btn-warning btn-block",
                     style = "margin-top: 10px; width: 250px;"),
        
        hr(),
        
        p(strong("Status:"), style = "margin-bottom: 5px;"),
        verbatimTextOutput("status_text", placeholder = TRUE),
        
        hr(),
        
        checkboxInput("use_cache", "Use cached model (faster)", value = TRUE),
        
        conditionalPanel(
          condition = "input.use_cache == false",
          numericInput("epochs", "Training Epochs:", value = 150, min = 50, max = 300, step = 50)
        )
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper { background-color: #ecf0f5; }
        .box { border-top: 3px solid #3c8dbc; }
        .value-box { font-size: 24px; font-weight: bold; }
        .btn-primary { background-color: #3c8dbc; }
        .btn-warning { background-color: #f39c12; }
      "))
    ),
    
    tabItems(
      # Prediction tab
      tabItem(
        tabName = "prediction",
        
        fluidRow(
          valueBoxOutput("current_price_box", width = 3),
          valueBoxOutput("tomorrow_price_box", width = 3),
          valueBoxOutput("change_box", width = 3),
          valueBoxOutput("signal_box", width = 3)
        ),
        
        fluidRow(
          box(
            title = "Tomorrow's Forecast Details",
            status = "primary",
            solidHeader = TRUE,
            width = 6,
            withSpinner(uiOutput("forecast_details"))
          ),
          
          box(
            title = "Model Performance Metrics",
            status = "info",
            solidHeader = TRUE,
            width = 6,
            withSpinner(uiOutput("performance_metrics"))
          )
        ),
        
        fluidRow(
          box(
            title = "Price Prediction Visualization",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            withSpinner(plotlyOutput("prediction_plot", height = "400px"))
          )
        ),
        
        fluidRow(
          box(
            title = "Training History",
            status = "info",
            solidHeader = TRUE,
            width = 6,
            withSpinner(plotlyOutput("training_plot", height = "300px"))
          ),
          
          box(
            title = "Trading Recommendation",
            status = "warning",
            solidHeader = TRUE,
            width = 6,
            withSpinner(uiOutput("trading_advice"))
          )
        ),
        
        fluidRow(
          box(
            title = "Prediction Data Table",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            withSpinner(DTOutput("prediction_table"))
          )
        )
      ),
      
      # Training tab
      tabItem(
        tabName = "training",
        
        fluidRow(
          box(
            title = "Available Models",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            withSpinner(DTOutput("models_table"))
          )
        ),
        
        fluidRow(
          box(
            title = "Training Log",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            verbatimTextOutput("training_log")
          )
        )
      ),
      
      # About tab
      tabItem(
        tabName = "about",
        box(
          title = "About This Application",
          status = "primary",
          solidHeader = TRUE,
          width = 12,
          HTML("
            <h3>🚀 Cryptocurrency Price Prediction Dashboard</h3>
            <p>Advanced LSTM-based cryptocurrency price forecasting with intelligent model caching.</p>
            
            <h4>✨ Key Features:</h4>
            <ul>
              <li><strong>Smart Caching:</strong> Trained models are saved and reused (prediction in <5 seconds)</li>
              <li><strong>10 Cryptocurrencies:</strong> Bitcoin, Ethereum, and 8 other major coins</li>
              <li><strong>17 Technical Indicators:</strong> RSI, MACD, Bollinger Bands, volatility, volume, etc.</li>
              <li><strong>Deep Learning:</strong> 2-layer LSTM with batch normalization and dropout</li>
              <li><strong>Bias Correction:</strong> Systematic bias removed for better accuracy</li>
              <li><strong>Trading Signals:</strong> Buy/Sell/Hold recommendations with confidence levels</li>
            </ul>
            
            <h4>📖 How to Use:</h4>
            <ol>
              <li><strong>Quick Prediction:</strong> Select a coin and click 'Quick Prediction'
                <ul>
                  <li>If model exists: Instant prediction (<5 sec)</li>
                  <li>If new coin: Trains automatically (5-10 min first time)</li>
                </ul>
              </li>
              <li><strong>Train New Model:</strong> Force retrain with latest data</li>
              <li><strong>View Results:</strong> See price forecast, metrics, charts, and trading advice</li>
            </ol>
            
            <h4>🧠 Model Architecture:</h4>
            <table class='table table-bordered'>
              <tr><th>Component</th><th>Details</th></tr>
              <tr><td>Input</td><td>Past 30 days × 17 features</td></tr>
              <tr><td>LSTM Layer 1</td><td>96 units + Batch Norm + Dropout (30%)</td></tr>
              <tr><td>LSTM Layer 2</td><td>48 units + Batch Norm + Dropout (30%)</td></tr>
              <tr><td>Dense Layers</td><td>32 → 16 units with L2 regularization</td></tr>
              <tr><td>Output</td><td>Tomorrow's % price change</td></tr>
              <tr><td>Training</td><td>Max 150 epochs with early stopping</td></tr>
              <tr><td>Optimizer</td><td>Adam (LR: 0.0001, gradient clipping)</td></tr>
            </table>
            
            <h4>📊 Performance Metrics:</h4>
            <ul>
              <li><strong>RMSE:</strong> Root Mean Squared Error in USD</li>
              <li><strong>MAE:</strong> Mean Absolute Error in USD</li>
              <li><strong>MAPE:</strong> Mean Absolute Percentage Error</li>
              <li><strong>Direction Accuracy:</strong> % of correct up/down predictions</li>
            </ul>
            
            <h4>💡 Tips:</h4>
            <ul>
              <li>Use <strong>cached models</strong> for instant predictions</li>
              <li>Train new models weekly to include latest market data</li>
              <li>Higher direction accuracy (>65%) = more reliable signals</li>
              <li>Confidence level indicates prediction reliability</li>
            </ul>
            
            <h4>⚠️ Disclaimer:</h4>
            <div class='alert alert-danger'>
              <strong>NOT FINANCIAL ADVICE!</strong> This tool is for educational purposes only. 
              Cryptocurrency trading is highly risky. Always do your own research and never invest 
              more than you can afford to lose.
            </div>
            
            <hr>
            <p><em>Built with R Shiny, Keras3, TensorFlow, and quantmod</em></p>
            <p><small>Version 2.0 | Last updated: October 2025</small></p>
          ")
        )
      )
    )
  )
)

# ============================================
# SERVER

server <- function(input, output, session) {
  
  # Reactive values
  prediction_results <- reactiveVal(NULL)
  training_log_text <- reactiveVal("Ready to train or predict.\n")
  
  # Helper function to append log
  append_log <- function(text) {
    current_log <- training_log_text()
    training_log_text(paste0(current_log, text, "\n"))
  }
  
  # Status text
  output$status_text <- renderText({
    results <- prediction_results()
    if (is.null(results)) {
      "Ready to predict"
    } else {
      paste("Last updated:", format(Sys.time(), "%H:%M:%S"))
    }
  })
  
  # Check if model exists
  model_exists <- reactive({
    symbol <- input$crypto_select
    model_path <- file.path(MODEL_DIR, paste0(symbol, "_modal.keras"))
    metadata_path <- file.path(MODEL_DIR, paste0(symbol, "_metadata.rds"))
    
    file.exists(model_path) && file.exists(metadata_path)
  })
  
  # ============================================
  # QUICK PREDICTION (Use cached model if available)
  
observeEvent(input$predict_btn, {
  
  symbol <- input$crypto_select
  crypto_name <- names(crypto_list)[crypto_list == symbol]
  
  append_log(paste0("\n========== QUICK PREDICTION: ", crypto_name, " =========="))
  append_log(paste0("Time: ", Sys.time()))
  
  # Check for cached model FIRST
  model_path <- file.path(MODEL_DIR, paste0(symbol, "_model.keras"))
  metadata_path <- file.path(MODEL_DIR, paste0(symbol, "_metadata.rds"))
  
  model_exists <- file.exists(model_path) && file.exists(metadata_path)
  
  if (input$use_cache && !model_exists) {
    
    showModal(modalDialog(
      title = tags$div(
        icon("exclamation-triangle", style = "color: orange;"),
        " No Trained Model Found"
      ),
      
      tags$div(
        style = "font-size: 16px;",
        
        p(strong(paste0("No saved model found for ", crypto_name))),
        
        p("The quick prediction feature requires a pre-trained model, but no model 
          was found at:"),
        
        tags$code(
          style = "display: block; background-color: #f5f5f5; padding: 10px; 
                   margin: 10px 0; border-radius: 5px; font-size: 14px;",
          model_path
        ),
        
        hr(),
        
        h4("What would you like to do?"),
        
        tags$ol(
          tags$li(strong("Train a new model"), " (Recommended) - Takes 5-10 minutes"),
          tags$li(strong("Disable 'Use cached model'"), " option and predict again")
        ),
        
        hr(),
        
        tags$div(
          class = "alert alert-info",
          style = "margin-top: 15px;",
          icon("info-circle"),
          " Once trained, the model will be saved for instant predictions in the future!"
        )
      ),
      
      footer = tagList(
        actionButton(
          "start_training",
          "Train New Model Now",
          icon = icon("play"),
          class = "btn-primary"
        ),
        modalButton("Cancel", icon = icon("times"))
      ),
      
      size = "m",
      easyClose = TRUE
    ))
    
    append_log("ERROR: No cached model found")
    append_log(paste0("  Path checked: ", model_path))
    
    return()
  }
  
  # Continue with normal prediction flow...
  progress <- Progress$new(session, min = 0, max = 1)
  on.exit(progress$close())
  progress$set(message = "Processing...", value = 0)
  
  if (input$use_cache && model_exists) {
    
    append_log("✓ Found cached model, loading...")
    progress$set(value = 0.2, detail = "Loading model...")
    
    loaded <- load_crypto_model(symbol, MODEL_DIR)
    
    if (!is.null(loaded)) {
      model <- loaded$model
      metadata <- loaded$metadata
      
      append_log(paste0("  Model trained: ", metadata$trained_date))
      append_log(paste0("  Test MAPE: ", round(metadata$test_mape, 2), "%"))
      
      # Download fresh data
      progress$set(value = 0.4, detail = "Downloading latest data...")
      crypto_df <- get_crypto_data(symbol)
      
      if (is.null(crypto_df)) {
        showNotification("Failed to download data!", type = "error", duration = 5)
        append_log("ERROR: Failed to download data")
        return()
      }
      
      append_log(paste0("✓ Downloaded ", nrow(crypto_df), " days of data"))
      
      # Add features
      progress$set(value = 0.6, detail = "Calculating indicators...")
      crypto_df <- add_technical_features(crypto_df)
      
      # Prepare data for prediction only
      feature_cols <- get_feature_columns()
      feature_data <- as.matrix(crypto_df[, feature_cols])
      
      # Use saved scaling params
      scaling_params <- metadata$scaling_params
      scaled_features <- matrix(0, nrow = nrow(feature_data), ncol = ncol(feature_data))
      
      for (i in 1:ncol(feature_data)) {
        scaled_features[, i] <- (feature_data[, i] - scaling_params$mean[i]) / 
          (scaling_params$sd[i] + 1e-8)
      }
      
      scaled_features[scaled_features > 4] <- 4
      scaled_features[scaled_features < -4] <- -4
      scaled_features[is.na(scaled_features)] <- 0
      scaled_features[is.infinite(scaled_features)] <- 0
      
      # Predict tomorrow
      progress$set(value = 0.8, detail = "Predicting...")
      
      time_steps <- metadata$time_steps
      latest_features <- tail(scaled_features, time_steps)
      latest_sequence <- array(latest_features, dim = c(1, time_steps, ncol(scaled_features)))
      
      latest_date <- tail(crypto_df$Date, 1)
      current_price <- tail(crypto_df$Close, 1)
      
      tomorrow_pct_scaled <- model %>% predict(latest_sequence, verbose = 0)
      tomorrow_pct_raw <- as.numeric(tomorrow_pct_scaled) * metadata$target_sd + metadata$target_mean
      tomorrow_pct_corrected <- tomorrow_pct_raw - metadata$systematic_bias
      
      tomorrow_price_raw <- current_price * (1 + tomorrow_pct_corrected / 100)
      tomorrow_price <- max(current_price * 0.90, min(current_price * 1.10, tomorrow_price_raw))
      
      price_change <- tomorrow_price - current_price
      pct_change <- (price_change / current_price) * 100
      
      direction_accuracy <- metadata$direction_accuracy
      
      confidence <- if (is.na(direction_accuracy)) {
        "LOW"
      } else if (direction_accuracy > 70) {
        "HIGH"
      } else if (direction_accuracy > 60) {
        "MODERATE"
      } else {
        "LOW"
      }
      
      signal <- if (abs(pct_change) < 0.5) {
        "HOLD"
      } else if (pct_change > 0) {
        if (pct_change > 3 && confidence == "HIGH") "STRONG BUY" else if (pct_change > 1.5) "BUY" else "WEAK BUY"
      } else {
        if (pct_change < -3 && confidence == "HIGH") "STRONG SELL" else if (pct_change < -1.5) "SELL" else "WEAK SELL"
      }
      
      progress$set(value = 1, detail = "Done!")
      
      # Store results
      prediction_results(list(
        latest_date = latest_date,
        current_price = current_price,
        tomorrow_price = tomorrow_price,
        price_change = price_change,
        pct_change = pct_change,
        confidence = confidence,
        signal = signal,
        test_rmse = metadata$test_rmse,
        test_mae = metadata$test_mae,
        test_mape = metadata$test_mape,
        direction_accuracy = direction_accuracy,
        systematic_bias = metadata$systematic_bias,
        history = metadata$history,
        test_dates = NULL,
        test_actual = NULL,
        test_pred = NULL
      ))
      
      append_log("✓ Prediction complete!")
      append_log(paste0("  Tomorrow: $", round(tomorrow_price, 2), " (", round(pct_change, 2), "%)"))
      append_log(paste0("  Signal: ", signal, " (", confidence, ")"))
      
      showNotification(
        paste0("Quick prediction for ", crypto_name, " complete!"), 
        type = "message",
        duration = 3
      )
      
    } else {
      showNotification("Failed to load cached model. Try training new model.", 
                      type = "warning", duration = 5)
      append_log("ERROR: Failed to load model")
    }
    
  } else {
    # No cache, train new model
    append_log("No cached model found or cache disabled. Training new model...")
    training_triggered$trigger <- TRUE
  }
})

# Handle "Train New Model Now" button from dialog
observeEvent(input$start_training, {
  removeModal()  # Close the dialog
  
  # Trigger training
  training_triggered$trigger <- TRUE
  
  # Show notification
  showNotification(
    "Starting training process...",
    type = "message",
    duration = 3
  )
})
  
  # ============================================
  # TRAIN NEW MODEL
  
  training_triggered <- reactiveValues(trigger = FALSE)
  
  observeEvent(c(input$train_btn, training_triggered$trigger), {
    
    if (!input$train_btn && !training_triggered$trigger) return()
    training_triggered$trigger <- FALSE
    
    symbol <- input$crypto_select
    crypto_name <- names(crypto_list)[crypto_list == symbol]
    
    append_log(paste0("\n========== TRAINING NEW MODEL: ", crypto_name, " =========="))
    append_log(paste0("Time: ", Sys.time()))
    
    progress <- Progress$new(session, min = 0, max = 1)
    on.exit(progress$close())
    progress$set(message = "Training model...", value = 0)
    
    # Download data
    progress$set(value = 0.05, detail = "Downloading data...")
    append_log("Downloading data...")
    
    crypto_df <- get_crypto_data(symbol)
    
    if (is.null(crypto_df)) {
      showNotification("Failed to download data!", type = "error")
      append_log("ERROR: Failed to download data")
      return()
    }
    
    append_log(paste0("✓ Downloaded ", nrow(crypto_df), " days of data"))
    
    # Add features
    progress$set(value = 0.1, detail = "Calculating technical indicators...")
    append_log("Calculating 17 technical indicators...")
    
    crypto_df <- add_technical_features(crypto_df)
    append_log(paste0("✓ Features added, clean data: ", nrow(crypto_df), " rows"))
    
    # Prepare data
    progress$set(value = 0.15, detail = "Preparing LSTM data...")
    append_log("Preparing sequences for LSTM...")
    
    feature_cols <- get_feature_columns()
    data_prep <- prepare_lstm_data(crypto_df, feature_cols, time_steps = 30, test_size = 30)
    
    append_log(paste0("✓ Created ", dim(data_prep$X_train)[1], " training samples"))
    append_log(paste0("✓ Created ", dim(data_prep$X_test)[1], " test samples"))
    
    # Build model
    progress$set(value = 0.2, detail = "Building LSTM model...")
    append_log("Building LSTM architecture...")
    
    model <- build_lstm_model(data_prep$time_steps, length(feature_cols))
    append_log("✓ Model built: 2 LSTM layers + dense layers")
    
    # Train model
    progress$set(value = 0.25, detail = "Training (this takes 5-10 min)...")
    append_log(paste0("Training for max ", input$epochs, " epochs..."))
    append_log("(This will take 5-10 minutes)")
    
    start_time <- Sys.time()
    
    history <- train_lstm_model(model, data_prep$X_train, data_prep$y_train, 
                                epochs = input$epochs, batch_size = 32)
    
    end_time <- Sys.time()
    training_time <- difftime(end_time, start_time, units = "mins")
    
    append_log(paste0("✓ Training completed in ", round(training_time, 2), " minutes"))
    append_log(paste0("  Epochs: ", length(history$metrics$loss)))
    append_log(paste0("  Final loss: ", round(tail(history$metrics$loss, 1), 6)))
    
    # Evaluate
    progress$set(value = 0.9, detail = "Evaluating performance...")
    append_log("Evaluating model performance...")
    
    evaluation <- evaluate_model(model, data_prep, crypto_df)
    
    append_log("✓ Evaluation complete:")
    append_log(paste0("  RMSE: $", round(evaluation$test_rmse, 2)))
    append_log(paste0("  MAE: $", round(evaluation$test_mae, 2)))
    append_log(paste0("  MAPE: ", round(evaluation$test_mape, 2), "%"))
    append_log(paste0("  Direction Accuracy: ", round(evaluation$direction_accuracy, 2), "%"))
    
    # Predict tomorrow
    progress$set(value = 0.95, detail = "Predicting tomorrow...")
    append_log("Predicting tomorrow's price...")
    
    tomorrow <- predict_tomorrow(model, data_prep, crypto_df, evaluation)
    
    append_log("✓ Tomorrow's forecast:")
    append_log(paste0("  Price: $", round(tomorrow$tomorrow_price, 2)))
    append_log(paste0("  Change: ", round(tomorrow$pct_change, 2), "%"))
    append_log(paste0("  Signal: ", tomorrow$signal, " (", tomorrow$confidence, ")"))
    
    # Save model
    append_log("Saving trained model...")
    save_crypto_model(model, data_prep, evaluation, history, symbol, MODEL_DIR)
    append_log("✓ Model saved successfully")
    
    progress$set(value = 1, detail = "Done!")
    
    # Store complete results
    prediction_results(list(
      latest_date = tomorrow$latest_date,
      current_price = tomorrow$current_price,
      tomorrow_price = tomorrow$tomorrow_price,
      price_change = tomorrow$price_change,
      pct_change = tomorrow$pct_change,
      confidence = tomorrow$confidence,
      signal = tomorrow$signal,
      test_rmse = evaluation$test_rmse,
      test_mae = evaluation$test_mae,
      test_mape = evaluation$test_mape,
      direction_accuracy = evaluation$direction_accuracy,
      systematic_bias = evaluation$systematic_bias,
      history = history,
      test_dates = evaluation$test_dates,
      test_actual = evaluation$test_actual,
      test_pred = evaluation$test_pred
    ))
    
    append_log("\n========== TRAINING COMPLETE ==========\n")
    
    showNotification(
      paste0("Model trained successfully for ", crypto_name, "!"),
      type = "message",
      duration = 5
    )
  })
  
  # ============================================
  # OUTPUTS - Value Boxes
  
  output$current_price_box <- renderValueBox({
    results <- prediction_results()
    if (is.null(results)) {
      valueBox("--", "Current Price", icon = icon("dollar-sign"), color = "blue")
    } else {
      valueBox(
        paste0("$", format(round(results$current_price, 2), big.mark = ",")),
        "Current Price",
        icon = icon("dollar-sign"),
        color = "blue"
      )
    }
  })
  
  output$tomorrow_price_box <- renderValueBox({
    results <- prediction_results()
    if (is.null(results)) {
      valueBox("--", "Tomorrow's Price", icon = icon("eye"), color = "purple")
    } else {
      valueBox(
        paste0("$", format(round(results$tomorrow_price, 2), big.mark = ",")),
        "Tomorrow's Price",
        icon = icon("eye"),
        color = "purple"
      )
    }
  })
  
  output$change_box <- renderValueBox({
    results <- prediction_results()
    if (is.null(results)) {
      valueBox("--", "Expected Change", icon = icon("chart-line"), color = "yellow")
    } else {
      color <- if (results$pct_change > 0) "green" else if (results$pct_change < 0) "red" else "yellow"
      icon_name <- if (results$pct_change > 0) "arrow-up" else if (results$pct_change < 0) "arrow-down" else "minus"
      
      valueBox(
        paste0(ifelse(results$pct_change > 0, "+", ""), round(results$pct_change, 2), "%"),
        "Expected Change",
        icon = icon(icon_name),
        color = color
      )
    }
  })
  
  output$signal_box <- renderValueBox({
    results <- prediction_results()
    if (is.null(results)) {
      valueBox("--", "Signal", icon = icon("signal"), color = "light-blue")
    } else {
      color <- if (grepl("BUY", results$signal)) "green" else if (grepl("SELL", results$signal)) "red" else "yellow"
      
      valueBox(
        results$signal,
        paste("Confidence:", results$confidence),
        icon = icon("signal"),
        color = color
      )
    }
  })
  
  # ============================================
  # OUTPUTS - Details
  
  output$forecast_details <- renderUI({
    results <- prediction_results()
    if (is.null(results)) {
      return(HTML("<p>Run a prediction to see forecast details...</p>"))
    }
    
    HTML(paste0("
      <table class='table table-striped'>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Date</td><td>", as.character(results$latest_date + 1), "</td></tr>
        <tr><td>Current Price</td><td>$", format(round(results$current_price, 2), big.mark = ","), "</td></tr>
        <tr><td>Predicted Price</td><td>$", format(round(results$tomorrow_price, 2), big.mark = ","), "</td></tr>
        <tr><td>Price Change</td><td>$", format(round(results$price_change, 2), big.mark = ","), "</td></tr>
        <tr><td>% Change</td><td>", round(results$pct_change, 2), "%</td></tr>
        <tr><td>Signal</td><td><strong>", results$signal, "</strong></td></tr>
        <tr><td>Confidence</td><td>", results$confidence, "</td></tr>
        <tr><td>Bias Correction</td><td>", round(results$systematic_bias, 3), "%</td></tr>
      </table>
    "))
  })
  
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
      <p style='margin-top: 10px;'><small><strong>Rating Guide:</strong> ✓ Excellent | ○ Good | ✗ Fair</small></p>
    "))
  })
  
  # ============================================
  # OUTPUTS - Plots
  
  output$prediction_plot <- renderPlotly({
    results <- prediction_results()
    if (is.null(results) || is.null(results$test_dates)) return(NULL)
    
    df <- data.frame(
      Date = results$test_dates,
      Actual = results$test_actual,
      Predicted = results$test_pred
    )
    
    plot_ly(df, x = ~Date) %>%
      add_trace(y = ~Actual, name = "Actual", type = "scatter", mode = "lines",
                line = list(color = "blue", width = 2)) %>%
      add_trace(y = ~Predicted, name = "Predicted", type = "scatter", mode = "lines",
                line = list(color = "red", width = 2, dash = "dash")) %>%
      layout(
        title = "Actual vs Predicted Prices (Test Set)",
        xaxis = list(title = "Date"),
        yaxis = list(title = "Price (USD)"),
        hovermode = "x unified",
        legend = list(x = 0.1, y = 0.9)
      )
  })
  
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
                line = list(color = "blue")) %>%
      add_trace(y = ~val_loss, name = "Validation Loss", type = "scatter", mode = "lines",
                line = list(color = "red")) %>%
      layout(
        title = "Model Training History",
        xaxis = list(title = "Epoch"),
        yaxis = list(title = "Loss (MAE)", type = "log"),
        hovermode = "x unified",
        legend = list(x = 0.7, y = 0.9)
      )
  })
  
  # ============================================
  # OUTPUTS - Trading Advice
  
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
      "Strong upward momentum expected. Consider entering a long position. Use stop-loss at -5%."
    } else if (grepl("BUY", results$signal)) {
      "Moderate upward movement predicted. Small long position recommended. Monitor closely."
    } else if (grepl("STRONG SELL", results$signal)) {
      "Strong downward pressure expected. Consider closing long positions. Use stop-loss at +5%."
    } else if (grepl("SELL", results$signal)) {
      "Moderate downward movement predicted. Reduce exposure or consider hedging."
    } else {
      "No clear direction. Hold current positions. Wait for stronger signals."
    }
    
    risk_note <- if (results$confidence == "HIGH") {
      "Confidence is HIGH. Model has strong directional accuracy (>70%)."
    } else if (results$confidence == "MODERATE") {
      "Confidence is MODERATE. Model has reasonable accuracy (60-70%)."
    } else {
      "Confidence is LOW. Use extra caution. Model accuracy is below 60%."
    }
    
    HTML(paste0("
      <div class='alert alert-", signal_color, "'>
        <h4 class='alert-heading'>", results$signal, "</h4>
        <p>", advice, "</p>
        <hr>
        <p class='mb-0'><strong>Confidence:</strong> ", risk_note, "</p>
      </div>
      <div class='alert alert-danger'>
        <strong>⚠️ DISCLAIMER:</strong> This is NOT financial advice. Crypto trading is extremely risky. 
        Do your own research and never invest more than you can afford to lose.
      </div>
    "))
  })
  
  # ============================================
  # OUTPUTS - Tables
  
  output$prediction_table <- renderDT({
    results <- prediction_results()
    if (is.null(results) || is.null(results$test_dates)) {
      return(data.frame(Message = "No data available"))
    }
    
    if (length(results$test_dates) == 0) {
      return(data.frame(Message = "No test predictions available (cached model)"))
    }
    
    df <- data.frame(
      Date = results$test_dates,
      Actual = round(results$test_actual, 2),
      Predicted = round(results$test_pred, 2),
      Error = round(results$test_pred - results$test_actual, 2),
      Pct_Error = round((results$test_pred - results$test_actual) / results$test_actual * 100, 2)
    )
    
    if (nrow(df) == 0) {
      return(data.frame(Message = "No data to display"))
    }
    
    # Simple 2-color scheme (Good vs Bad)
    datatable(df, 
              options = list(pageLength = 10, scrollX = TRUE),
              rownames = FALSE) %>%
      formatCurrency(c("Actual", "Predicted", "Error"), "$") %>%
      formatStyle(
        "Pct_Error",
        backgroundColor = styleInterval(
          cuts = 0,  # 1 cut
          values = c("#f8d7da", "#d4edda")  # 2 colors (negative = red, positive = green)
        )
      )
  })
  
  output$models_table <- renderDT({
    model_files <- list.files(MODEL_DIR, pattern = "_metadata.rds$", full.names = TRUE)
    
    if (length(model_files) == 0) {
      return(data.frame(Message = "No trained models yet. Train a model to get started!"))
    }
    
    models_info <- lapply(model_files, function(f) {
      meta <- readRDS(f)
      data.frame(
        Cryptocurrency = meta$symbol,
        Trained = format(meta$trained_date, "%Y-%m-%d %H:%M"),
        RMSE = round(meta$test_rmse, 2),
        MAPE = round(meta$test_mape, 2),
        Direction_Acc = round(meta$direction_accuracy, 2),
        Status = "Ready"
      )
    })
    
    df <- do.call(rbind, models_info)
    
    datatable(df, options = list(pageLength = 10), rownames = FALSE) %>%
      formatStyle(
        "Direction_Acc",
        backgroundColor = styleInterval(c(55, 65), c("#f8d7da", "#fff3cd", "#d4edda"))
      )
  })
  
  output$training_log <- renderText({
    training_log_text()
  })
}

# ============================================
# RUN APP

shinyApp(ui = ui, server = server)
