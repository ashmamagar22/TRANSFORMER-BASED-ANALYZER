# Transformer-Based Product Sentiment Analyzer

## Overview
This project is an interactive web application built with Streamlit that uses a custom-built Transformer model to analyze the sentiment of product reviews. The model predicts whether a given review is "Positive" or "Negative", providing a confidence score along with the class probabilities.

The key feature of this application is its custom implementation of the Transformer architecture. The inference logic, including Multi-Head Attention, Feed-Forward Networks, and Positional Encoding, is implemented purely using NumPy.

## Features
- **Custom Transformer Architecture:** Inference is performed using a custom-built Transformer model with NumPy.
- **Interactive Web Interface:** A user-friendly Streamlit UI to enter product reviews and view results instantly.
- **Dynamic Hyperparameters:** Users can adjust the maximum sequence length (Max Tokens) dynamically from the sidebar. The model reloads seamlessly.
- **Real-time Preprocessing Visualization:** Displays the clean tokens, their mapping to vocabulary IDs, and the final padded sequence.
- **Model Metrics Visualization:** Shows the model's performance on the validation set, including a Confusion Matrix, Accuracy, Precision, Recall, and F1-score.

## System Requirements
- Python 3.7+
- Dependencies: `streamlit`, `pandas`, `numpy`, `matplotlib`

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ashmamagar22/TRANSFORMER-BASED-ANALYZER.git
   cd TRANSFORMER-BASED-ANALYZER
   # Navigate to the correct directory containing app.py
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. Install the required packages via pip:
   ```bash
   pip install streamlit pandas numpy matplotlib
   ```

3. **Ensure Required Files are Present:**
   Make sure `transformer_checkpoint.pkl` (the model checkpoint) and `model_metrics.pkl` (the evaluation metrics) are in the same directory as `app.py`.

4. **Run the Application:**
   Start the Streamlit development server:
   ```bash
   streamlit run app.py
   ```

5. **Access the Web App:**
   Open your browser and navigate to `http://localhost:8501`.

## How It Works
1. **Input:** The user types a product review into the text area.
2. **Cleaning & Tokenization:** The application removes special characters, converts the text to lowercase, and splits it into individual tokens.
3. **Encoding & Padding:** The tokens are converted into numerical IDs using the model's vocabulary. If the sequence is shorter than the configured "Sequence Length", it is padded; if it's longer, it's truncated.
4. **Inference:** The array of token IDs is passed through the custom Transformer model (`model_utils.py`). The model computes the Self-Attention and Feed-Forward layers to extract contextual representations.
5. **Output:** The raw logits from the model are passed through a softmax function to determine the probability of each class. The class with the highest probability is displayed as the final sentiment.
