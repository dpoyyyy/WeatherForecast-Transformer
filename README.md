# Transformer Architecture for Time Series Forecasting 

## Overview
This repository implements a Transformer-based model for time series forecasting using PyTorch. The model is designed to predict future temperature values based on historical weather data, utilizing deep learning techniques for sequential data processing.

## Project Details
- **Framework**: PyTorch
- **Data Preprocessing**: MinMaxScaler for normalization
- **Feature Selection**: Temperature (`temp`), Dew Point (`dwpt`), and Wind Speed (`wspd`)
- **Model Architecture**: Transformer
- **Training Method**: Supervised Learning with a sliding window approach
- **Hardware Support**: GPU acceleration (if available)

## Transformer Model Explanation
Transformers are deep learning models primarily used for sequential data. Unlike RNNs, they leverage self-attention mechanisms to capture dependencies across long time steps more effectively.

### Key Components:
1. **Embedding Layer**: Converts input numerical values into a higher-dimensional space.
2. **Positional Encoding**: Injects information about the order of data points since Transformers process input in parallel.
3. **Multi-Head Self-Attention**: Allows the model to weigh different input time steps differently, capturing long-range dependencies.
4. **Feed-Forward Network**: Applies transformations after attention mechanisms.
5. **Output Layer**: Predicts the target variable.

## Installation
To set up the project environment, install the necessary dependencies:
```bash
pip install pandas numpy scikit-learn torch matplotlib
```

## Dataset
The dataset contains hourly weather measurements, including:
- Temperature (`temp`)
- Dew Point (`dwpt`)
- Wind Speed (`wspd`)


## Model Training
The Transformer model is trained using sequences of past weather data to predict future temperature values. The training pipeline includes:
- Data Loading using `TensorDataset` and `DataLoader`.
- Defining a Transformer model with an encoder-only architecture.
- Training with Mean Squared Error (MSE) loss and Adam optimizer.
- Evaluating the model’s performance on unseen data.

## Usage
Run the training script to preprocess data and train the model:
```python
python train.py
```

## Results
The model learns temporal dependencies and improves prediction accuracy over time. Performance evaluation is conducted using:
MSE Loss: 0.0020029402803629637
RMSE Loss: 0.04475422203540802

## License
This project is licensed under the MIT License.



