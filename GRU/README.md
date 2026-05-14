# Advanced GRU Sentiment Analysis

This project implements an advanced Gated Recurrent Unit (GRU) model for sentiment analysis using the IMDB movie reviews dataset.

## Files

- `advanced_gru_sentiment_analysis.ipynb` - full notebook implementation
- `requirements.txt` - Python dependencies

## Setup

```powershell
cd "D:\projects personal\1_random\gru"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

Open:

```text
advanced_gru_sentiment_analysis.ipynb
```

## What The Notebook Covers

- IMDB sentiment dataset loading
- sequence padding
- GRU architecture
- bidirectional GRU model
- dropout and regularization
- early stopping
- training curves
- confusion matrix
- custom review prediction
