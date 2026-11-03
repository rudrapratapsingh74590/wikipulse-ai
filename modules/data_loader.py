
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_FILES = {
    "English": DATA_DIR / "enwiki" / "enwiki_full_analysis.parquet",
    "French": DATA_DIR / "frwiki" / "frwiki_full_analysis.parquet",
}


@st.cache_data(show_spinner="Loading Wikimedia articles...")
def load_language_data(language):

    file_path = DATA_FILES.get(language)

    if file_path is None or not file_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(file_path)

        if df.empty:
            return pd.DataFrame()

        return df

    except Exception as error:
        st.error(f"Unable to load {language} data: {error}")
        return pd.DataFrame()


def load_all_data():

    frames = []

    for language in DATA_FILES:

        df = load_language_data(language)

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def get_data_status():

    status = {}

    for language, path in DATA_FILES.items():

        status[language] = {
            "available": path.exists(),
            "path": str(path),
        }

    return status


# --------------------------------------------------
# DUCKDB LANGUAGE COMPARISON
# --------------------------------------------------

@st.cache_data(show_spinner="Analyzing language datasets...")
def get_language_comparison():

    queries = []

    language_codes = {
        "English": "en",
        "French": "fr",
    }

    for language, path in DATA_FILES.items():

        if not path.exists():
            continue

        safe_path = path.as_posix().replace("'", "''")
        language_code = language_codes[language]

        queries.append(f"""
            SELECT
                '{language_code}' AS language,
                COUNT(article) AS total_articles,

                ROUND(
                    AVG(TRY_CAST(completeness_score AS DOUBLE)),
                    2
                ) AS average_score,

                ROUND(
                    AVG(TRY_CAST("references" AS DOUBLE)),
                    2
                ) AS average_references,

                COALESCE(
                    SUM(TRY_CAST(characters AS BIGINT)),
                    0
                ) AS total_characters

            FROM read_parquet('{safe_path}')
        """)

    if not queries:
        return pd.DataFrame()

    sql = " UNION ALL ".join(queries)

    try:

        with duckdb.connect(database=":memory:") as con:

            con.execute("SET threads = 2")
            con.execute("SET memory_limit = '1GB'")

            result = con.execute(sql).fetchdf()

        return result

    except Exception as error:

        st.error(f"Language comparison failed: {error}")
        return pd.DataFrame()
