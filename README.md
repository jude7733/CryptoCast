Model              |  Type                |  Pros                                          |  Cons                             |  Scaling y?       
-------------------+----------------------+------------------------------------------------+-----------------------------------+-------------------
Linear Regression  |  Statistical         |  Simple, interpretable                         |  Assumes linearity                |  No               
Ridge/Lasso        |  Regularized Linear  |  Feature selection, handles multicollinearity  |  Still linear                     |  No               
Random Forest      |  Tree Ensemble       |  Robust, handles non-linearity                 |  Slow on large data               |  No               
XGBoost            |  Gradient Boosting   |  Fast, accurate, handles missing data          |  Hyperparameter tuning needed     |  No               
LightGBM           |  Gradient Boosting   |  Faster than XGBoost                           |  Less stable on small data        |  No               
CatBoost           |  Gradient Boosting   |  Handles categorical features                  |  Slower training                  |  No               
GBM                |  Gradient Boosting   |  Good performance                              |  Slower than XGBoost/LightGBM     |  No               
SVR                |  Kernel Method       |  Handles non-linearity well                    |  Slow, sensitive to parameters    |  Yes (recommended)
KNN                |  Instance-based      |  Simple, no training                           |  Slow prediction, sensitive to k  |  Yes (recommended)
Neural Network     |  Deep Learning       |  Flexible, powerful                            |  Needs tuning, overfitting risk   |  Yes (required)   
LSTM               |  Deep Learning       |  Captures sequences                            |  Complex, needs lots of data      |  Yes (required)   
ARIMA              |  Time Series         |  Classic statistical                           |  Only univariate                  |  No               
Prophet            |  Time Series         |  Handles seasonality, trends                   |  Assumes additive structure       |  No               