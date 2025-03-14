# Fashion MNIST Classification & Recommendation System

## Project Description
This project addresses Fashion MNIST classification using a convolutional neural network (CNN), generates synthetic sales trends with LSTM, and builds a hybrid recommendation system. The notebook includes interactive visualization with Gradio and explores time-series forecasting for fashion trends.

## Problem Statement & Approach
- **Classification**: Train a CNN to classify Fashion MNIST images (10 clothing categories) with ~91% test accuracy  
- **Trend Prediction**: Forecast future sales/social trends using synthetic data and LSTM  
- **Recommendation**: Create a hybrid (content + collaborative filtering) system using LightFM and image embeddings  
- **Interactive Demo**: Deploy results via Gradio for real-time predictions

## Technologies & Libraries
- `TensorFlow/Keras` (CNN and LSTM models)
- `LightFM` (recommendation system)
- `Gradio` (interactive UI)
- `Pandas/NumPy` (data processing)
- `Matplotlib` (visualization) 
- `scikit-learn` (data preprocessing)

## Results & Insights
- Achieved **91.25% test accuracy** on Fashion MNIST classification
- Generated synthetic sales/user interaction datasets for realistic trend modeling
- Built hybrid recommendation system combining CNN image features and user behavior
- Visualized sample predictions and forecasted trends

## Instructions

### 1. Clone the repository
```bash
git clone https://github.com/[your-username]/Fashion-MNIST-Classification-Recommendation.git
cd Fashion-MNIST-Classification-Recommendation
```

### 2. Install dependencies
```bash
pip install tensorflow keras lightfm gradio pandas numpy matplotlib scikit-learn
```

### 3. Run the notebook
```bash
jupyter notebook ML_Group5.ipynb
```

*Or open in Google Colab*

Execute cells sequentially to:
- Train CNN on Fashion MNIST
- Generate synthetic sales/user interaction data
- Forecast trends with LSTM
- Build recommendation system
- Launch Gradio demo

### 4. Interact with the demo
Use Gradio's interface to:
- Upload Fashion MNIST images for real-time classification
- Explore recommendation predictions
- View trend forecasts

## Dataset
- **Fashion MNIST**: Loaded via `tf.keras.datasets.fashion_mnist.load_data()`
- **Synthetic Data**:
  - Sales trends (100 weeks of simulated sales volume/social mentions)
  - User interactions (50 users with 20 synthetic engagements each)

## Model Architecture

### CNN Classifier
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'), 
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### LSTM Trend Predictor
```python
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(3, 2)),
    Dense(2)
])
```

## Key Features
- Interactive Gradio UI for model testing
- Hybrid recommendation system combining:
  - Image embeddings from CNN
  - Collaborative filtering with LightFM
- Time-series analysis with synthetic trend forecasting

## Note
1. Enable GPU acceleration in Colab (Runtime > Change runtime type)
2. Allow 10-15 minutes for full model training
3. First-time LightFM installation may require compiler tools:
```bash
sudo apt-get install build-essential python-dev
```
