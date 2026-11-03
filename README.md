# WikiPulse AI

**An Interactive Wikipedia Article Intelligence & Analytics Platform**

WikiPulse AI is a data analytics dashboard designed to explore Wikipedia article completeness, reference distribution, quality indicators, and language-level patterns.

## Features

- **Overview:** Dataset statistics and key performance indicators.
- **Article Quality Ranking:** Explore and rank articles using completeness indicators.
- **Article Completeness:** Analyze article completeness scores.
- **Reference Intelligence:** Understand reference distribution and density.
- **Language Comparison:** Compare English and French Wikipedia datasets.
- **Article Search:** Search and filter articles.
- **Knowledge Explorer:** Explore article metadata.
- **Article Detail Explorer:** Inspect individual article information.

## Technology Stack

- Python
- Streamlit
- Pandas
- DuckDB
- Plotly
- PyArrow

## Dataset

This project uses Wikimedia Structured Contents data from Kaggle.

Processed datasets:

- English Wikipedia: 7.59 million articles
- French Wikipedia: 2.87 million articles

Large Parquet datasets are managed separately using Git LFS.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Important Note

Completeness scores and reference indicators are analytical metrics based on the processed dataset. They are not official Wikipedia quality ratings and do not establish factual accuracy.

## Deployment

The application is designed for deployment using Streamlit Community Cloud.
