# Jena Climate Temperature Forecasting System

A complete time series forecasting project for predicting short-horizon temperature values using the Jena Climate dataset. The project includes data cleaning, exploratory analysis, time-aware preprocessing, baseline modeling, LSTM and GRU neural networks, evaluation, visualization, and final reporting.

---

## Project Overview

This project forecasts the next temperature value from historical climate measurements recorded at 10-minute intervals. The target variable is:

```text
T (degC)
```

The forecasting setup uses the previous 24 hours of climate observations to predict the temperature 10 minutes ahead.

The project compares three approaches:

- Persistence Baseline
- LSTM neural network
- GRU neural network

The final evaluation shows that the GRU model achieved the lowest test RMSE, while the persistence baseline remained highly competitive due to the very short forecasting horizon.

---

## Repository Structure

```text
.
├── data/
│   └── Data card.docx
├── outputs/
│   ├── plots/
│   │   ├── actual_vs_predicted_temperature.png
│   │   ├── daily_humidity.png
│   │   ├── daily_pressure.png
│   │   ├── daily_temperature.png
│   │   ├── daily_wind_speed.png
│   │   ├── feature_correlation_temperature.png
│   │   ├── gru_training_history.png
│   │   └── lstm_training_history.png
│   ├── reports/
│   │   └── forecasting_report.md
│   ├── feature_correlation_temperature.csv
│   ├── model_comparison_results.csv
│   ├── scaled_dataset_evaluation.csv
│   └── temperature_predictions.csv
├── jena_climate_forecasting.ipynb
├── README.md
├── requirements.txt
├── setup.sh
└── .gitignore
```

---

## Dataset

The project uses the Jena Climate dataset, which contains meteorological measurements collected at 10-minute intervals.

Main variables include:

- Temperature
- Pressure
- Relative humidity
- Wind speed
- Wind direction
- Vapor pressure features
- Air density
- Dew point
- Potential temperature

The raw dataset is intentionally not included in this repository to avoid uploading large data files.

Expected local path:

```text
data/jena_climate_2009_2016.csv
```

---

## Main Features

- Chronological train, validation, and test splitting
- Missing and invalid value handling
- Time-aware preprocessing without data leakage
- Standardization using training data only
- Sliding-window forecasting setup
- Memory-safe dataset pipeline
- Persistence baseline comparison
- LSTM and GRU model training
- Early stopping and model checkpointing
- RMSE and MAE evaluation in Celsius
- Final plots and saved prediction outputs

---

## Methodology

### 1. Data Cleaning and Exploration

The dataset is loaded, inspected, and cleaned. The `Date Time` column is converted into a datetime index. Invalid wind speed placeholder values are handled before modeling.

Exploratory analysis includes:

- Daily average temperature trends
- Daily average humidity trends
- Daily average pressure trends
- Daily average wind speed trends
- Feature correlation with temperature

### 2. Feature Engineering and Scaling

The target variable is `T (degC)`. The selected input features include temperature and related climate variables.

The dataset is split chronologically to preserve the time order:

```text
Train      -> earliest period
Validation -> middle period
Test       -> latest period
```

Standardization is fitted on the training set only, then applied to validation and test data.

### 3. Forecast Windowing

The model uses a sliding window approach:

```text
Input window: 144 time steps
Forecast horizon: 1 time step
```

Since each time step represents 10 minutes:

```text
144 time steps = 24 hours
1-step horizon = 10 minutes ahead
```

### 4. Modeling

Three methods are evaluated:

#### Persistence Baseline

Predicts the next temperature as the last observed temperature in the input window.

#### LSTM Model

Uses recurrent layers with dropout and dense layers to learn temporal patterns.

#### GRU Model

Uses GRU recurrent layers with dropout and dense layers. GRU is generally simpler and faster than LSTM while often achieving comparable or better performance.

---

## Results

Final test performance after converting predictions back to Celsius:

| Model | Test MAE (°C) | Test RMSE (°C) |
|---|---:|---:|
| Persistence Baseline | 0.156399 | 0.235913 |
| LSTM | 0.229640 | 0.326621 |
| GRU | 0.161125 | 0.227562 |

### Key Findings

- GRU achieved the lowest RMSE, making it the strongest model for reducing larger forecasting errors.
- The persistence baseline achieved the lowest MAE, which is expected because the model predicts only 10 minutes ahead.
- LSTM performed worse than both GRU and the persistence baseline in this experiment.
- The actual-vs-predicted plot shows that all methods follow the overall temperature trend closely.

---

## Visualizations

### Temperature Trend

```text
outputs/plots/daily_temperature.png
```

### Feature Correlation

```text
outputs/plots/feature_correlation_temperature.png
```

### Training Curves

```text
outputs/plots/lstm_training_history.png
outputs/plots/gru_training_history.png
```

### Actual vs Predicted Temperature

```text
outputs/plots/actual_vs_predicted_temperature.png
```

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or run the setup script if available:

```bash
bash setup.sh
```

---

## How to Run

1. Place the dataset at:

```text
data/jena_climate_2009_2016.csv
```

2. Open the notebook:

```text
jena_climate_forecasting.ipynb
```

3. Run all cells from top to bottom.

4. Review generated outputs in:

```text
outputs/
```

---

## Generated Outputs

The notebook generates:

```text
outputs/model_comparison_results.csv
outputs/scaled_dataset_evaluation.csv
outputs/temperature_predictions.csv
outputs/feature_correlation_temperature.csv
outputs/reports/forecasting_report.md
outputs/plots/
```

---

## Technical Notes

The project uses a memory-safe dataset pipeline to avoid large TensorFlow tensor allocation issues when working with long time series data.

The models are trained one at a time to reduce memory pressure and improve stability.

---

## Limitations

The forecast horizon is only one time step ahead, equivalent to 10 minutes. This makes the persistence baseline very strong because temperature usually changes gradually over short intervals.

For a more challenging forecasting problem, future experiments should test longer horizons such as:

```text
1 hour ahead  -> 6 time steps
24 hours ahead -> 144 time steps
```

---

## Future Improvements

Potential improvements include:

- Predicting further into the future
- Adding cyclic time features such as hour-of-day and day-of-year
- Testing deeper or bidirectional recurrent models
- Comparing against classical forecasting models
- Testing tree-based models such as Random Forest or XGBoost
- Performing feature selection to reduce redundant variables
- Evaluating model performance across different seasons

---

## Conclusion

This project implements a complete temperature forecasting workflow using the Jena Climate dataset. It demonstrates how to prepare time series data, create forecast windows, train recurrent neural networks, compare against a strong baseline, and evaluate predictions using MAE and RMSE.

The GRU model achieved the best RMSE on the test set, while the persistence baseline remained highly competitive due to the short forecasting horizon.
