import io
import re

import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px


def read_csv_safely(uploaded_file):
    raw_data = uploaded_file.getvalue()
    strategies = [
        {"sep": ",", "engine": "python", "on_bad_lines": "skip"},
        {"sep": ";", "engine": "python", "on_bad_lines": "skip"},
        {"sep": "\t", "engine": "python", "on_bad_lines": "skip"},
        {"sep": None, "engine": "python", "on_bad_lines": "skip"},
    ]

    last_error = None
    for strategy in strategies:
        try:
            return pd.read_csv(io.BytesIO(raw_data), **strategy)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"The uploaded file could not be parsed: {last_error}")


def normalize_text(text):
    if text is None:
        return ""

    cleaned = str(text)
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def detect_text_column(df):
    if df.empty:
        raise ValueError("The dataset is empty.")

    candidate_columns = [
        "text",
        "review",
        "comment",
        "feedback",
        "customer_review",
        "customer_review_text",
        "message",
        "description",
        "sentence",
        "content",
    ]

    normalized = {str(col).lower().replace(" ", "_"): col for col in df.columns}

    for key in candidate_columns:
        if key in normalized:
            return normalized[key]

    for col in df.columns:
        if df[col].map(lambda x: isinstance(x, str) and len(str(x).strip()) > 0).any():
            return col

    raise ValueError("No valid text column found in the dataset.")


def analyze_text(text, classifier):
    cleaned = normalize_text(text)
    if not cleaned:
        raise ValueError("Input text cannot be empty.")

    result = classifier(cleaned)
    if isinstance(result, list):
        result = result[0]

    label = result["label"]
    score = float(result["score"])
    return label, score


def analyze_reviews(df, text_column, classifier):
    if df.empty or text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found or dataset is empty.")

    cleaned_texts = []
    for value in df[text_column].tolist():
        cleaned_value = normalize_text(value)
        if not cleaned_value:
            cleaned_texts.append("")
        else:
            cleaned_texts.append(cleaned_value)

    predictions = classifier(cleaned_texts)

    results = []
    for value, prediction in zip(df[text_column].tolist(), predictions):
        cleaned_value = normalize_text(value)
        if not cleaned_value:
            sentiment = "Neutral"
            score = 0.0
        else:
            sentiment = prediction["label"]
            score = float(prediction["score"])
        results.append({"text": cleaned_value or value, "sentiment": sentiment, "score": score})

    output = pd.DataFrame(results)
    summary = output["sentiment"].value_counts().to_dict()

    return {"data": output, "summary": summary, "sentiment": output["sentiment"]}


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis",
    )


st.set_page_config(page_title="Sentiment AI", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #0b1120 0%, #111827 100%);
        }
        h1, h2, h3, p {
            color: #e5e7eb;
        }
        .stButton > button {
            border-radius: 12px;
            background: linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
            border: none;
            font-weight: 600;
        }
        .stTextArea textarea {
            border-radius: 12px;
            border: 1px solid #374151;
            background: rgba(17, 24, 39, 0.7);
            color: white;
        }
        .stSelectbox > div > div {
            background: rgba(17, 24, 39, 0.7);
            color: white;
        }
        .stDataFrame, .stFileUploader, .stAlert {
            border-radius: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 1rem;">
        <div style="width: 52px; height: 52px; border-radius: 16px; background: linear-gradient(135deg, #8b5cf6 0%, #22d3ee 100%); display: flex; align-items: center; justify-content: center; font-size: 26px; box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);">
            🧠
        </div>
        <div>
            <div style="font-size: 2rem; font-weight: 800; color: #f8fafc; line-height: 1.1;">Sentiment AI</div>
            <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 4px;">AI-powered sentiment analysis dashboard</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(34,211,238,0.12)); border: 1px solid rgba(148,163,184,0.25); border-radius: 18px; padding: 1.25rem 1.4rem; margin-bottom: 1.5rem;">
        <div style="font-size: 1.05rem; color: #e2e8f0;">
            <strong>Turn customer feedback into business insight.</strong>
            Detect mood, emotion, and sentiment from reviews, comments, and support messages in real time.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

classifier = load_model()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", "Multilingual")
with col2:
    st.metric("Use case", "Reviews")
with col3:
    st.metric("Output", "Label + score")

st.markdown("---")

with st.expander("🧩 How it works"):
    st.write(
        "The app cleans the input text, detects the best text column in a CSV file, and sends the content to a transformer model that estimates whether the sentiment is positive, neutral, or negative."
    )
    st.write(
        "It is designed to be resilient to messy real-world data, including noisy text, inconsistent column names, and partially unstructured feedback."
    )

with st.expander("📄 Demo dataset"):
    st.write("Use the demo file to explore the dashboard immediately.")
    st.download_button(
        label="Download demo CSV",
        data=open("demo_reviews.csv", "rb").read(),
        file_name="demo_reviews.csv",
        mime="text/csv",
    )

st.markdown("---")

with st.container():
    st.subheader("Single text analysis")
    text = st.text_area(
        "Write your sentence:",
        height=140,
        placeholder="Type a review, comment, or message...",
    )

    if st.button("Analyze sentiment", type="primary"):
        if not text or not text.strip():
            st.warning("Please enter a sentence before analyzing.")
        else:
            try:
                label, score = analyze_text(text, classifier)
                st.subheader("Result")

                if label in {"Very Positive", "Positive"}:
                    st.success(f"😊 {label}")
                elif label == "Neutral":
                    st.info(f"😐 {label}")
                elif label in {"Negative", "Very Negative"}:
                    st.warning(f"🙁 {label}")
                else:
                    st.error(f"😞 {label}")

                st.write(f"Confidence: {score:.2%}")
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

st.divider()

st.header("📂 Batch sentiment analysis")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = read_csv_safely(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head(), use_container_width=True)

    try:
        recommended_column = detect_text_column(df)
        text_column = st.selectbox(
            "Select the text column",
            options=df.columns.tolist(),
            index=df.columns.get_loc(recommended_column),
        )
    except ValueError:
        st.warning("No clear text column detected. Please choose one manually.")
        text_column = st.selectbox("Select the text column", options=df.columns.tolist())

    if st.button("Analyze file"):
        try:
            result = analyze_reviews(df, text_column, classifier)
            analysis_df = result["data"].copy()

            st.subheader("Summary")
            summary = pd.Series(result["summary"]).sort_index()
            st.bar_chart(summary)

            st.subheader("Detailed results")
            st.dataframe(analysis_df.head(20), use_container_width=True)

            fig = px.pie(
                analysis_df["sentiment"].value_counts().reset_index(),
                names="index",
                values="sentiment",
                title="Sentiment distribution",
                color_discrete_sequence=["#34d399", "#fbbf24", "#f87171"],
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"The file could not be processed: {exc}")