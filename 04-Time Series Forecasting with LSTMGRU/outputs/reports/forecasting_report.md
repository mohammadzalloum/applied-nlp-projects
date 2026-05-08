
# Time Series Forecasting on the Jena Climate Dataset

## 1. Introduction

This project focuses on forecasting future temperature values using historical climate measurements from the Jena Climate dataset. Since the data is sequential and collected over time, recurrent neural network models such as LSTM and GRU were used to learn temporal patterns.

The main target variable was `T (degC)`, which represents temperature in Celsius.

## 2. Dataset Overview

The dataset contains climate measurements collected at regular 10-minute intervals. The main variables include temperature, pressure, humidity, wind speed, wind direction, vapor pressure, air density, and related atmospheric features.

Exploratory data analysis showed clear seasonal patterns in temperature. Temperature values rise during warmer periods and decrease during colder periods, which confirms that the dataset has strong temporal and seasonal behavior.

## 3. Data Cleaning and Preprocessing

The `Date Time` column was converted into a datetime format and used as the time index. Invalid placeholder values in wind speed columns were handled before modeling.

The dataset was split chronologically into training, validation, and test sets to preserve the time order and avoid data leakage.

Standardization was applied using statistics learned from the training data only. The same scalers were then applied to the validation and test sets.

## 4. Time Series Windowing

A sliding window approach was used to transform the time series into supervised learning samples.

The input window size was set to 144 time steps. Since the dataset is recorded every 10 minutes, 144 steps represent one full day of historical observations.

The forecast horizon was set to 1 time step, meaning the models predict the temperature 10 minutes ahead.

## 5. Model Architectures

Two recurrent neural network models were implemented:

- LSTM model
- GRU model

Both models used recurrent layers followed by dropout and dense layers. Dropout and L2 regularization were used to reduce overfitting.

The models were trained using the Adam optimizer and Mean Squared Error loss. Early stopping, model checkpointing, and learning rate reduction were used during training.

## 6. Evaluation Results

The models were evaluated using MAE and RMSE after converting predictions back to Celsius.

| Model | Test MAE (°C) | Test RMSE (°C) |
|---|---:|---:|
| Persistence Baseline | 0.156399 | 0.235913 |
| LSTM | 0.229640 | 0.326621 |
| GRU | 0.161125 | 0.227562 |

The GRU model achieved the lowest RMSE, which means it was better at reducing larger forecasting errors. The persistence baseline achieved a slightly lower MAE than GRU, which is expected because the forecast horizon is only one step ahead.

The LSTM model performed worse than both the GRU model and the persistence baseline.

## 7. Visualization Analysis

The actual-vs-predicted plot shows that all methods follow the general temperature trend closely. This is expected because the forecast horizon is only 10 minutes ahead.

The GRU model tracks the actual temperature more accurately than the LSTM model, especially during sharper changes. The persistence baseline is also very strong because temperature usually changes gradually over short time intervals.

## 8. Challenges

The main technical challenge was managing memory and GPU stability while training recurrent models on a large time series dataset. A memory-safe dataset pipeline was used to avoid large TensorFlow tensor allocation issues.

Another important modeling challenge was the strength of the persistence baseline. Since the prediction horizon is very short, a simple baseline can be highly competitive.

## 9. Future Improvements

Future improvements could include:

- Predicting further into the future, such as 1 hour ahead or 24 hours ahead.
- Adding cyclic time features such as hour-of-day and day-of-year.
- Testing deeper or bidirectional recurrent architectures.
- Comparing against non-neural models such as Random Forest, XGBoost, or ARIMA.
- Performing feature selection to remove highly redundant climate variables.
- Training models on longer forecast horizons where recurrent models may show stronger advantages.

## 10. Conclusion

This project successfully implemented and compared LSTM and GRU models for temperature forecasting on the Jena Climate dataset.

The GRU model achieved the best RMSE on the test set, making it the strongest neural model in this experiment. However, the persistence baseline remained highly competitive due to the short 10-minute forecasting horizon.

Overall, the results show that recurrent neural networks can model climate time series effectively, but baseline comparison is essential for interpreting forecasting performance correctly.
