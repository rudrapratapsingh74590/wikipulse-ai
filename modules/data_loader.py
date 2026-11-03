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
LANGUAGE_CODES = {"English": "en", "French": "fr"}

def get_data_status():
    return {lang: {"available": path.exists(), "path": str(path)} for lang, path in DATA_FILES.items()}

def _safe_path(language):
    path = DATA_FILES.get(language)
    if path is None:
        raise ValueError(f"Unsupported language: {language}")
    if not path.exists():
        raise FileNotFoundError(f"{language} dataset not found: {path}")
    return path.as_posix().replace("'", "''")

def query_language(language, sql, params=None):
    """Run bounded DuckDB queries directly on Parquet without loading full datasets into Pandas."""
    statement = sql.replace("__PARQUET__", f"read_parquet('{_safe_path(language)}')")
    with duckdb.connect(database=":memory:") as con:
        con.execute("SET threads = 2")
        con.execute("SET memory_limit = '768MB'")
        con.execute("SET preserve_insertion_order = false")
        return con.execute(statement, params or []).fetchdf()

@st.cache_data(show_spinner="Calculating dataset overview...")
def get_overview_metrics(language):
    result = query_language(language, """
        SELECT COUNT(*) AS total_articles,
               ROUND(AVG(TRY_CAST(completeness_score AS DOUBLE)), 2) AS average_score,
               COUNT(*) FILTER (WHERE reference_status = 'No recorded references') AS no_references,
               COUNT(DISTINCT language) AS languages
        FROM __PARQUET__
    """)
    return result.iloc[0].to_dict() if not result.empty else {}

@st.cache_data(show_spinner="Analyzing article quality...")
def get_quality_distribution_sql(language):
    return query_language(language, """
        SELECT CASE
            WHEN TRY_CAST(completeness_score AS DOUBLE) >= 80 THEN 'Strong'
            WHEN TRY_CAST(completeness_score AS DOUBLE) >= 60 THEN 'Good'
            WHEN TRY_CAST(completeness_score AS DOUBLE) >= 40 THEN 'Needs Improvement'
            ELSE 'Poor' END AS category, COUNT(*) AS articles
        FROM __PARQUET__
        WHERE TRY_CAST(completeness_score AS DOUBLE) IS NOT NULL
        GROUP BY 1
        ORDER BY CASE category WHEN 'Strong' THEN 1 WHEN 'Good' THEN 2
                 WHEN 'Needs Improvement' THEN 3 ELSE 4 END
    """)

@st.cache_data(show_spinner="Analyzing reference coverage...")
def get_reference_distribution_sql(language):
    return query_language(language, """
        SELECT reference_status AS status, COUNT(*) AS articles
        FROM __PARQUET__ GROUP BY reference_status ORDER BY articles DESC
    """)

@st.cache_data(show_spinner="Finding articles...")
def get_article_results(language, mode="weak", threshold=60, min_characters=2000, limit=1000):
    limit=max(1,min(int(limit),50000))
    if mode=="top":
        order="TRY_CAST(completeness_score AS DOUBLE) DESC NULLS LAST, article"
        where="TRY_CAST(completeness_score AS DOUBLE) IS NOT NULL"
    elif mode=="bottom":
        order="TRY_CAST(completeness_score AS DOUBLE) ASC NULLS LAST, article"
        where="TRY_CAST(completeness_score AS DOUBLE) IS NOT NULL"
    elif mode=="below":
        order="TRY_CAST(completeness_score AS DOUBLE) ASC NULLS LAST, article"
        where=f"TRY_CAST(completeness_score AS DOUBLE) < {float(threshold)}"
    elif mode=="low_reference":
        order="TRY_CAST(characters AS BIGINT) DESC NULLS LAST"
        where=f"TRY_CAST(characters AS BIGINT) >= {int(min_characters)} AND TRY_CAST(reference_density AS DOUBLE) < 0.5"
    else:
        order="TRY_CAST(completeness_score AS DOUBLE) ASC NULLS LAST, article"
        where="TRY_CAST(completeness_score AS DOUBLE) IS NOT NULL"
    return query_language(language,f"""
        SELECT article,url,TRY_CAST(completeness_score AS DOUBLE) AS completeness_score,
               TRY_CAST(characters AS BIGINT) AS characters,
               TRY_CAST("references" AS BIGINT) AS "references",
               TRY_CAST(reference_density AS DOUBLE) AS reference_density,reference_status
        FROM __PARQUET__ WHERE {where} ORDER BY {order} LIMIT {limit}
    """)

@st.cache_data(show_spinner="Calculating quality summary...")
def get_quality_summary_sql(language):
    result=query_language(language,"""
        SELECT COUNT(*) FILTER (WHERE TRY_CAST(completeness_score AS DOUBLE)>=80) AS strong,
        COUNT(*) FILTER (WHERE TRY_CAST(completeness_score AS DOUBLE)>=60 AND TRY_CAST(completeness_score AS DOUBLE)<80) AS good,
        COUNT(*) FILTER (WHERE TRY_CAST(completeness_score AS DOUBLE)>=40 AND TRY_CAST(completeness_score AS DOUBLE)<60) AS needs_improvement,
        COUNT(*) FILTER (WHERE TRY_CAST(completeness_score AS DOUBLE)<40) AS poor
        FROM __PARQUET__
    """)
    return result.iloc[0].to_dict() if not result.empty else {}

@st.cache_data(show_spinner="Comparing language datasets...")
def get_language_comparison():
    queries=[]
    for language,path in DATA_FILES.items():
        if path.exists():
            safe=path.as_posix().replace("'","''")
            code=LANGUAGE_CODES[language]
            queries.append(f"""
                SELECT '{code}' AS language,COUNT(article) AS total_articles,
                ROUND(AVG(TRY_CAST(completeness_score AS DOUBLE)),2) AS average_score,
                ROUND(AVG(TRY_CAST("references" AS DOUBLE)),2) AS average_references,
                COALESCE(SUM(TRY_CAST(characters AS BIGINT)),0) AS total_characters
                FROM read_parquet('{safe}')
            """)
    if not queries: return pd.DataFrame()
    with duckdb.connect(database=":memory:") as con:
        con.execute("SET threads=2")
        con.execute("SET memory_limit='768MB'")
        return con.execute(" UNION ALL ".join(queries)).fetchdf()

@st.cache_data(show_spinner="Loading a limited article sample...")
def load_language_data(language,limit=5000):
    if not DATA_FILES.get(language,Path("/missing")).exists(): return pd.DataFrame()
    return query_language(language,"SELECT * FROM __PARQUET__ LIMIT ?",[max(1,min(int(limit),50000))])

def load_all_data():
    frames=[load_language_data(lang) for lang in DATA_FILES]
    frames=[frame for frame in frames if not frame.empty]
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
