# Turkish Universities Sentiment Analysis

A comprehensive sentiment analysis platform for Turkish university student tweets using state-of-the-art deep learning models.

## Contributors
- **Emre Çelik**
- **Alihan Uludağ**

This project performs sentiment analysis on Turkish university tweets using Transformer models (BERTurk, ELECTRA) and classical deep learning models (CNN, BiLSTM, CNN-BiLSTM).

---

## Highlights

- **5 Production Models**: BERTurk, Turkish ELECTRA, CNN-BiLSTM, BiLSTM, CNN
- **Interactive Dashboard**: Real-time predictions with Streamlit UI
- **Batch Processing**: CLI tools for bulk predictions and evaluation
- **Comprehensive Analysis**: Model comparison, temporal trends, university-specific insights
- **High Performance**: BERTurk achieves 90.18% accuracy, ELECTRA 91.98%

---

## Project Structure

```
tweet-sentiment-analysis/
│
├── app.py                      # Streamlit Web Application
├── predict.py                  # Core Prediction Engine
├── batch_predict.py            # Backward compatibility wrapper
│
├── scripts/                    # Batch Processing Tools
│   ├── predict_batch.py           # Bulk predictions (5 models)
│   └── evaluate_predictions.py   # Model evaluation framework
│
├── models/                     # Trained Models
│   ├── berturk_model/             # BERTurk (Hugging Face format)
│   ├── electra_model/             # Turkish ELECTRA (Hugging Face format)
│   ├── cnn_model.pt               # CNN weights
│   ├── bilstm_model.pt            # BiLSTM weights
│   ├── hybrid_model.pt            # CNN-BiLSTM weights
│   └── spm_uni.model              # SentencePiece tokenizer (classical models)
│
├── data/                       # Datasets
│   └── tweetVeriseti.xlsx         # Main tweet dataset
│
├── .streamlit/                 # Streamlit Configuration
│   └── config.toml                # Theme and UI settings
│
└── requirements.txt            # Python dependencies
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/EmreCelik23/university-tweets-sentiment-analysis.git
cd university-tweets-sentiment-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Batch Predictions

Predict sentiments for multiple tweets at once:

```bash
python scripts/predict_batch.py \
  --input data/tweets.xlsx \
  --output results/predictions.xlsx \
  --text-column text
```

### Evaluate Predictions

Compare models against ground truth labels:

```bash
python scripts/evaluate_predictions.py \
  --input results/predictions.xlsx \
  --ground-truth labels \
  --output results/evaluation.csv
```

---

## Features

### Web Application (Streamlit)

- **Live Analysis**: Real-time sentiment prediction with all 5 models
- **Model Dashboard**: Performance metrics and comparison
  - Accuracy, F1 Score, Precision, Recall for each model
  - Class-wise performance breakdown
- **Temporal Analysis**: University trends over time
  - Hype Graph: Normalized tweet volume by year
  - Sentiment Trends: Positive sentiment percentage evolution
  - Yearly Sentiment Heatmap
- **Data Lab**: Sample predictions with model agreement visualization
- **Active Learning**: Add corrected predictions to training pool

### Batch Processing Scripts

#### predict_batch.py
- Process Excel/CSV files with thousands of tweets
- Get predictions from all 5 models simultaneously
- Output includes prediction + confidence for each model
- Progress tracking with tqdm

#### evaluate_predictions.py
- Calculate accuracy, precision, recall, F1 score
- Generate confusion matrices
- Compare models side-by-side
- Identify best performing model
- Export results to CSV/Excel

---

## Models

### Transformer Models

| Model | Accuracy | Macro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **BERTurk** | 90.18% | 0.8796 | 0.8642 | 0.9013 |
| **Turkish ELECTRA** | 91.98% | 0.8948 | 0.9010 | 0.8892 |

### Classical Models (with SentencePiece)

| Model | Accuracy | Macro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **CNN-BiLSTM** | 88.11% | 0.8446 | 0.8486 | 0.8408 |
| **BiLSTM** | 85.32% | 0.8216 | 0.8073 | 0.8440 |
| **CNN** | 85.23% | 0.8082 | 0.8096 | 0.8068 |

---

## Dataset

The dataset consists of **real tweets** collected from Turkish university students:

- **Total Tweets**: 5,043 real tweets
- **Positive Samples**: 1,374 (27.3%)
- **Negative Samples**: 3,669 (72.7%)
- **Time Range**: 2020-2025
- **Language**: Turkish

### Data Columns

- `text`: Tweet content
- `tags`: Sentiment label (0: negative, 1: positive)
- `university`: Associated university
- `createdAt`: Tweet timestamp
- `authorUserName`: Tweet author (anonymized)
- `location`: User location
- `type`: Data source type
- `url`: Tweet URL

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Streamlit 1.29+
- pandas, numpy, scikit-learn
- plotly (for visualizations)
- sentencepiece (for classical models)
- tqdm (for progress bars)

See `requirements.txt` for complete list.

---

## Platform Specific Instructions

### macOS / Linux

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Windows

```cmd
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## UI Customization

The Streamlit interface uses a custom dark theme with glassmorphism effects. Modify `.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor = "#58a6ff"
backgroundColor = "#0d1117"
secondaryBackgroundColor = "#161b22"
textColor = "#c9d1d9"
font = "sans serif"
```

---

## Troubleshooting

### Common Issues

**"Module not found" Error**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

**Model Loading Error**
- Verify `models/` directory contains all model files
- Check file permissions

**Streamlit Won't Start**
- Ensure port 8501 is not in use
- Try: `streamlit run app.py --server.port 8502`

**Slow Predictions**
- Transformer models require significant compute
- Consider using GPU if available
- For batch processing, adjust batch size in scripts

---

## Usage Examples

### Example 1: Analyze Single Tweet

```python
from predict import get_multi_model_prediction

text = "Üniversitemizin kütüphanesi harika ama yemekhane kötü"
results = get_multi_model_prediction(text, university="Genel")

for model, (prediction, confidence) in results.items():
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"{model}: {sentiment} ({confidence:.2f})")
```

### Example 2: Batch Process with Specific Model

Use `scripts/predict_batch.py` to process large datasets efficiently.

### Example 3: Model Comparison

Use `scripts/evaluate_predictions.py` to:
- Compare all models against ground truth
- Generate performance reports
- Identify best model for your use case

---

## Related Repositories

- **Training Code**: [university-tweets-sentiment-analysis-model-training](https://github.com/uldagalihan/university-tweets-sentiment-analysis-model-training)
  - Model training scripts
  - Data collection and preprocessing
  - Dataset splitting strategies

---

## License

This project is part of a university graduation project. Please contact contributors for licensing information.

---

## Contact

For questions, suggestions, or collaboration:
- **Emre Çelik**: GitHub [@EmreCelik23](https://github.com/EmreCelik23)
- **Alihan Uludağ**: GitHub [@uldagalihan](https://github.com/uldagalihan)

---