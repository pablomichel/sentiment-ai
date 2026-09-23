import io

import pandas as pd
import pytest

from app import analyze_text, analyze_reviews, detect_text_column, normalize_text, read_csv_safely


def test_analyze_text_returns_label_and_score():
    fake_classifier = lambda text: [{"label": "Positive", "score": 0.91}]

    label, score = analyze_text("I like this project", fake_classifier)

    assert label == "Positive"
    assert score == 0.91


def test_analyze_text_rejects_empty_input():
    fake_classifier = lambda text: [{"label": "Neutral", "score": 0.5}]

    with pytest.raises(ValueError, match="empty"):
        analyze_text("   ", fake_classifier)


def test_analyze_reviews_returns_summary_counts():
    df = pd.DataFrame({"text": ["Great job", "I hate this", "Meh"]})

    def fake_classifier(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [
            {"label": "Positive", "score": 0.9},
            {"label": "Negative", "score": 0.8},
            {"label": "Neutral", "score": 0.6},
        ]

    result = analyze_reviews(df, "text", fake_classifier)

    assert list(result["sentiment"].values) == ["Positive", "Negative", "Neutral"]
    assert result["summary"]["Positive"] == 1
    assert result["summary"]["Negative"] == 1
    assert result["summary"]["Neutral"] == 1


def test_normalize_text_removes_noise():
    cleaned = normalize_text("   I LOVE!!! this project???  ")

    assert cleaned == "I LOVE this project"


def test_detect_text_column_handles_messy_names():
    df = pd.DataFrame({
        "customer_review_text": ["Good service"],
        "id": [1],
    })

    assert detect_text_column(df) == "customer_review_text"


def test_read_csv_safely_skips_bad_lines():
    csv_data = b"text,score\nGood service,4\nBad service,1\nThis row is malformed\n"
    uploaded = io.BytesIO(csv_data)
    uploaded.getvalue = lambda: csv_data

    df = read_csv_safely(uploaded)

    assert not df.empty
    assert "text" in df.columns
