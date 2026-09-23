# Sentiment AI

A Streamlit app for sentiment analysis on customer reviews and text data.

## 🚀 Live Demo

Try the application here:

https://sentiment-ai-e9l2556cjfc3cxmkgzsabn.streamlit.app/

## Overview

This project uses a Hugging Face transformer model to classify text as positive, neutral, or negative. It can analyze:

- a single sentence entered by the user
- a CSV file with customer reviews or comments
- messy or partially unstructured text data

## Features

- Single-text sentiment analysis
- Batch analysis from CSV files
- Automatic detection of the most relevant text column
- Text cleaning and normalization
- Friendly dashboard UI
- Confidence score for each prediction
- Support for noisy and imperfect data

## Tech Stack

- Python
- Streamlit
- Transformers
- Pandas
- Plotly
- Hugging Face models

## Project Structure

- `app.py` — main Streamlit application
- `test_model.py` — automated regression tests
- `demo_reviews.csv` — sample dataset for testing
- `requirements.txt` — project dependencies

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the environment:

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

## Usage

### Single text analysis

- Paste a sentence or review in the text box
- Click "Analyze sentiment"
- View the predicted label and confidence score

### CSV analysis

- Upload a CSV file
- Select the text column if needed
- Click "Analyze file"
- Explore the summary and sentiment distribution charts

## Example Dataset

The project includes a demo file called `demo_reviews.csv`.

## Notes

This project is designed for demo and portfolio use and can be extended with:

- model comparison
- multilingual support
- more advanced preprocessing
- business dashboards
- deployment to a cloud platform

## License

This project is intended for educational and portfolio use.
