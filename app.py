import math
import re
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import (
    load_language_data,
    get_data_status,
    get_language_comparison,
    get_overview_metrics,
    get_quality_distribution_sql,
    get_reference_distribution_sql,
    get_article_results,
    get_quality_summary_sql,
    query_language,
)

from modules.analytics import (
    get_dashboard_metrics,
    get_quality_distribution,
    get_reference_distribution,
    find_weak_articles,
    find_long_articles_with_low_references,
    get_article_rankings,
    get_quality_summary,
    get_knowledge_graph_data,
)

from modules.charts import (
    quality_chart,
    reference_chart,
    language_chart,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WikiPulse AI | Wikipedia Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME — NAVY / BLUE
# ============================================================

BG = "#0b1120"
PANEL = "#111827"
PANEL_LIGHT = "#172033"
BORDER = "#263449"
BLUE = "#60a5fa"
TEXT = "#f8fafc"
MUTED = "#94a3b8"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    [data-testid="stSidebar"] {{
        background: {PANEL};
        border-right: 1px solid {BORDER};
    }}

    .block-container {{
        max-width: 1550px;
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }}

    .main-heading {{
        font-size: 38px;
        font-weight: 800;
        color: {BLUE};
        letter-spacing: -1px;
        line-height: 1.15;
    }}

    .sub-heading {{
        color: {MUTED};
        font-size: 15px;
        margin-top: 7px;
        margin-bottom: 4px;
    }}

    .section-heading {{
        font-size: 21px;
        font-weight: 750;
        color: {TEXT};
        margin-bottom: 5px;
    }}

    .helper-text {{
        color: {MUTED};
        font-size: 13px;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(
            145deg,
            #172554,
            {PANEL_LIGHT}
        );
        padding: 18px 20px;
        border-radius: 13px;
        border: 1px solid #334155;
        min-height: 112px;
        transition: border-color .2s ease, transform .2s ease;
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: {BLUE};
        transform: translateY(-2px);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}

    div[data-testid="stMetricValue"] {{
        color: {TEXT};
        font-weight: 750;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid #334155;
        border-radius: 11px;
        overflow: hidden;
    }}

    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 9px;
        border: 1px solid #3b82f6;
        min-height: 40px;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: {BLUE};
        color: {BLUE};
    }}

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] {{
        border-radius: 9px;
    }}

    hr {{
        border-color: {BORDER};
    }}

    @media (max-width: 768px) {{
        .main-heading {{
            font-size: 29px;
        }}

        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

QUALITY_ORDER = [
    "Strong",
    "Good",
    "Needs Improvement",
    "Poor",
]

PARQUET_PATHS = {
    "English": "data/enwiki/enwiki_full_analysis.parquet",
    "French": "data/frwiki/frwiki_full_analysis.parquet",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def quality_category(score):
    """Convert completeness score into a quality category."""

    try:
        score = float(score)
    except (TypeError, ValueError):
        return "Unknown"

    if score >= 80:
        return "Strong"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Needs Improvement"

    return "Poor"


def clean_wikipedia_url(value):
    """Extract a usable Wikipedia URL."""

    if pd.isna(value):
        return ""

    value = str(value).strip()

    match = re.search(
        r"\]\((https?://.+?)\)\s*$",
        value,
    )

    if match:
        value = match.group(1)

    value = value.replace(r"\(", "(").replace(r"\)", ")")

    if value.startswith(("https://", "http://")):
        return value

    return ""


def prepare_article_table(data, include_category=True):
    """Prepare article data for display."""

    if data is None or data.empty:
        return pd.DataFrame()

    result = data.copy()

    if "completeness_score" in result.columns:
        result["completeness_score"] = pd.to_numeric(
            result["completeness_score"],
            errors="coerce",
        ).round(2)

    if include_category and "completeness_score" in result.columns:
        result["quality_category"] = (
            result["completeness_score"].apply(quality_category)
        )

    if "url" in result.columns:
        result["Open Article"] = result["url"].apply(
            clean_wikipedia_url
        )

    preferred = [
        "Rank",
        "article",
        "completeness_score",
        "quality_category",
        "characters",
        "references",
        "reference_density",
        "reference_status",
        "Open Article",
    ]

    columns = [
        column for column in preferred
        if column in result.columns
    ]

    return result, columns


def show_article_table(
    data,
    key,
    height=440,
    show_category=True,
):
    """Display article data with readable columns."""

    if data is None or data.empty:
        st.info("No articles match the selected filters.")
        return

    prepared = prepare_article_table(
        data,
        include_category=show_category,
    )

    if isinstance(prepared, tuple):
        result, columns = prepared
    else:
        result = prepared
        columns = list(result.columns)

    display = result[columns].copy()

    rename = {
        "article": "Wikipedia Article",
        "completeness_score": "Completeness Score",
        "quality_category": "Quality Category",
        "characters": "Characters",
        "references": "References",
        "reference_density": "Reference Density",
        "reference_status": "Reference Status",
    }

    display = display.rename(columns=rename)

    column_config = {
        "Wikipedia Article": st.column_config.TextColumn(
            "Wikipedia Article",
            width="large",
        ),
        "Rank": st.column_config.NumberColumn(
            "Rank",
            format="%d",
            width="small",
        ),
        "Open Article": st.column_config.LinkColumn(
            "Open Article",
            display_text="Open ↗",
            width="small",
        ),
        "Completeness Score": st.column_config.ProgressColumn(
            "Completeness Score",
            min_value=0,
            max_value=100,
            format="%.2f",
            width="medium",
        ),
        "Quality Category": st.column_config.TextColumn(
            "Quality Category",
            width="medium",
        ),
        "Characters": st.column_config.NumberColumn(
            "Characters",
            format="%d",
            width="small",
        ),
        "References": st.column_config.NumberColumn(
            "References",
            format="%d",
            width="small",
        ),
        "Reference Density": st.column_config.NumberColumn(
            "Reference Density",
            format="%.3f",
            width="small",
        ),
        "Reference Status": st.column_config.TextColumn(
            "Reference Status",
            width="medium",
        ),
    }

    st.dataframe(
        display,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=height,
        key=key,
    )


def article_search_filter(data, query):
    """Filter a Pandas result by article title."""

    if (
        not query
        or data is None
        or data.empty
        or "article" not in data.columns
    ):
        return data

    return data[
        data["article"].astype("string").str.contains(
            query,
            case=False,
            na=False,
            regex=False,
        )
    ]


def csv_bytes(data):
    return data.to_csv(index=False).encode("utf-8-sig")


def show_download(
    data,
    filename,
    label="📥 Download CSV",
):
    if data is not None and not data.empty:
        st.download_button(
            label,
            data=csv_bytes(data),
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}",
        )


def section_title(title, description=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )

    if description:
        st.caption(description)


def get_parquet_path(language):
    return PARQUET_PATHS.get(
        language,
        PARQUET_PATHS["English"],
    )


def get_duckdb_connection():
    """Create a memory-limited DuckDB connection."""

    con = duckdb.connect(database=":memory:")
    con.execute("SET threads = 2")
    con.execute("SET memory_limit = '1GB'")

    return con


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("# 🌐 WikiPulse AI")

st.sidebar.caption(
    "Wikipedia Quality Intelligence Platform"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Article Quality Ranking",
        "Article Completeness",
        "Reference Intelligence",
        "Language Comparison",
        "Article Search",
        "Knowledge Explorer",
        "Article Detail Explorer",
    ],
    index=5,
)

st.sidebar.divider()

language = st.sidebar.selectbox(
    "SELECT LANGUAGE",
    ["English", "French"],
)

if st.sidebar.button(
    "🔄 Refresh Dataset",
    use_container_width=True,
):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.caption(
    "Built with Wikimedia Structured Contents Dataset"
)

st.sidebar.caption("Project: WikiPulse AI")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-heading">WikiPulse AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-heading">'
    'Discover. Analyze. Improve. | '
    'Multilingual Wikipedia Intelligence Platform'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# DATA LOADING — DuckDB keeps full Parquet files out of Pandas
# ============================================================
data_status = get_data_status()
df = pd.DataFrame()  # Dashboard pages query Parquet and materialize bounded results.

# ============================================================
# DATA AVAILABILITY
# ============================================================
if page != "Language Comparison" and not data_status.get(language, {}).get("available"):
    st.warning(f"No processed {language} Parquet dataset is available.")
    st.stop()


# ============================================================
# PART 1: OVERVIEW
# ============================================================
if page == "Overview":
    section_title("Platform Overview","Monitor article completeness, reference coverage and dataset health.")
    metrics=get_overview_metrics(language)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("📚 Total Articles",f"{int(metrics.get('total_articles') or 0):,}")
    c2.metric("⭐ Average Completeness",f"{float(metrics.get('average_score') or 0):.2f}/100")
    c3.metric("📖 No Recorded References",f"{int(metrics.get('no_references') or 0):,}")
    c4.metric("🌐 Languages",int(metrics.get("languages") or 1))
    st.divider()
    left,right=st.columns(2,gap="large")
    with left:
        section_title("Article Quality Distribution")
        distribution=get_quality_distribution_sql(language)
        if not distribution.empty:
            fig=quality_chart(distribution)
            fig.update_layout(height=380,paper_bgcolor=PANEL,plot_bgcolor=PANEL,font=dict(color="#e2e8f0"),title=None)
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False,"responsive":True})
        else: st.info("Quality distribution is unavailable.")
    with right:
        section_title("Reference Intelligence")
        reference_data=get_reference_distribution_sql(language)
        if not reference_data.empty:
            fig=reference_chart(reference_data)
            fig.update_layout(height=380,paper_bgcolor=PANEL,plot_bgcolor=PANEL,font=dict(color="#e2e8f0"),title=None,showlegend=False)
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False,"responsive":True})
        else: st.info("Reference distribution is unavailable.")
    st.divider()
    section_title("⚠️ Article Quality Explorer","Explore the 1,000 lowest-scoring articles in the selected dataset.")
    weak=get_article_results(language,mode="weak",limit=1000)
    if not weak.empty:
        weak["quality_category"]=weak["completeness_score"].apply(quality_category)
        search_col,category_col=st.columns([2,1])
        with search_col:
            query=st.text_input("🔎 Search article",placeholder="Enter Wikipedia article title...",key="overview_article_search")
        with category_col:
            selected_category=st.selectbox("Quality category",["All"]+QUALITY_ORDER,key="overview_quality_filter")
        filtered=article_search_filter(weak,query)
        if selected_category!="All": filtered=filtered[filtered["quality_category"]==selected_category]
        st.markdown(f"**{len(filtered):,} articles found in the selected result set**")
        st.caption("The explorer searches within the 1,000 lowest-scoring records, not the entire dataset.")
        if not filtered.empty:
            show_article_table(filtered.head(500),key="overview_quality_table",height=470)
            show_download(filtered,f"wikipulse_{language.lower()}_quality_explorer.csv","📥 Download filtered results")
        else: st.info("No articles match your search and category filter.")
    else: st.success("No articles needing improvement were found.")
    st.divider()
    section_title("Dataset Information")
    i1,i2,i3=st.columns(3)
    i1.metric("Current Dataset",language)
    i2.metric("Articles Analyzed",f"{int(metrics.get('total_articles') or 0):,}")
    i3.metric("Dataset Columns","8")


# ============================================================
# PART 2: ARTICLE QUALITY RANKING
# ============================================================
elif page == "Article Quality Ranking":
    section_title("🏆 Article Quality Ranking","Explore the highest and lowest-scoring Wikipedia articles.")
    summary=get_quality_summary_sql(language)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Strong",f"{int(summary.get('strong') or 0):,}")
    c2.metric("Good",f"{int(summary.get('good') or 0):,}")
    c3.metric("Needs Improvement",f"{int(summary.get('needs_improvement') or 0):,}")
    c4.metric("Poor",f"{int(summary.get('poor') or 0):,}")
    st.divider()
    st.subheader("🥇 Top 10 Articles")
    st.caption("Highest completeness scores. Ties are ordered alphabetically.")
    top10=get_article_results(language,mode="top",limit=10)
    if not top10.empty:
        top10["quality_category"]=top10["completeness_score"].apply(quality_category)
        top10.insert(0,"Rank",range(1,len(top10)+1))
        show_article_table(top10,key="ranking_top_10",height=400)
        show_download(top10,"wikipulse_top_10_articles.csv","📥 Download Top 10")
    st.divider()
    st.subheader("⚠️ Bottom 10 Articles")
    st.caption("Lowest completeness scores. Ties are ordered alphabetically.")
    bottom10=get_article_results(language,mode="bottom",limit=10)
    if not bottom10.empty:
        bottom10["quality_category"]=bottom10["completeness_score"].apply(quality_category)
        bottom10.insert(0,"Rank",range(1,len(bottom10)+1))
        show_article_table(bottom10,key="ranking_bottom_10",height=400)
        show_download(bottom10,"wikipulse_bottom_10_articles.csv","📥 Download Bottom 10")
    st.divider()
    st.subheader("📊 Full Article Rankings")
    st.caption("Search, filter and select how many of the highest-ranked articles to display.")
    s,c,l=st.columns([2,1,1])
    with s: ranking_search=st.text_input("Search article",placeholder="Enter article title...",key="full_ranking_search")
    with c: ranking_category=st.selectbox("Quality category",["All"]+QUALITY_ORDER,key="full_ranking_category")
    with l: ranking_limit=st.selectbox("Display articles",[10,25,50,100,500],index=2,key="full_ranking_limit")
    full_ranking=get_article_results(language,mode="top",limit=ranking_limit)
    if not full_ranking.empty:
        full_ranking["quality_category"]=full_ranking["completeness_score"].apply(quality_category)
        full_ranking=article_search_filter(full_ranking,ranking_search)
        if ranking_category!="All": full_ranking=full_ranking[full_ranking["quality_category"]==ranking_category]
        full_ranking.insert(0,"Rank",range(1,len(full_ranking)+1))
        st.markdown(f"**{len(full_ranking):,} matching articles**")
        show_article_table(full_ranking,key="full_article_ranking",height=500)
        show_download(full_ranking,"wikipulse_full_article_rankings.csv","📥 Download Full Ranking Results")
    else: st.info("No articles match your selected filters.")
    st.info("Completeness scores are experimental dataset-based indicators. They are not official Wikipedia quality ratings or assessments of factual accuracy.")


# ============================================================
# PART 3: ARTICLE COMPLETENESS
# ============================================================
elif page == "Article Completeness":
    section_title("📊 Article Completeness Analyzer","Find records below a selected completeness threshold.")
    threshold=st.slider("Show articles below this score",min_value=0,max_value=100,value=60,step=5)
    weak=get_article_results(language,mode="below",threshold=threshold,limit=50000)
    c1,c2,c3=st.columns(3)
    c1.metric("Articles Below Threshold",f"{len(weak):,}")
    c2.metric("Selected Threshold",f"{threshold}/100")
    c3.metric("Average Score of Results",f"{weak['completeness_score'].mean():.2f}" if not weak.empty else "—")
    st.divider()
    section_title("Article Results")
    search=st.text_input("Search within results",placeholder="Enter article title...",key="completeness_search")
    filtered=article_search_filter(weak,search)
    st.caption(f"{len(filtered):,} matching records (up to 50,000 retrieved)")
    if not filtered.empty:
        show_article_table(filtered.head(1000),key="completeness_results",height=470)
        show_download(filtered,"wikipulse_completeness_report.csv","📥 Download retrieved completeness results")


# ============================================================
# PART 4: REFERENCE INTELLIGENCE
# ============================================================
elif page == "Reference Intelligence":
    section_title("📚 Reference Intelligence","Analyze recorded references and identify long articles with low reference density.")
    distribution=get_reference_distribution_sql(language)
    if not distribution.empty:
        fig=reference_chart(distribution)
        fig.update_layout(height=420,paper_bgcolor=PANEL,plot_bgcolor=PANEL,font=dict(color="#e2e8f0"),title=None,showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.divider()
    section_title("Long Articles With Low Reference Density")
    min_characters=st.slider("Minimum article length (characters)",min_value=500,max_value=10000,value=2000,step=500)
    low_reference=get_article_results(language,mode="low_reference",min_characters=min_characters,limit=50000)
    st.metric("Articles Identified (up to 50,000)",f"{len(low_reference):,}")
    if not low_reference.empty:
        show_article_table(low_reference.head(1000),key="low_reference_articles",height=470,show_category=False)
        show_download(low_reference,"wikipulse_reference_intelligence.csv","📥 Download retrieved reference report")
    else: st.success("No articles match the selected conditions.")
    st.caption("A low or zero recorded reference count describes this processed dataset snapshot; it does not prove that the current live article has no citations.")


# ============================================================
# PART 5: LANGUAGE COMPARISON
# ============================================================

elif page == "Language Comparison":

    section_title(
        "🌍 English vs French Comparison",
        "Compare article counts, average completeness and reference coverage.",
    )

    with st.spinner("Analyzing English and French datasets..."):
        comparison = get_language_comparison()

    if comparison is None or comparison.empty:

        st.warning(
            "Language comparison data is unavailable. "
            "Check that both Parquet files exist and contain valid data."
        )

    else:

        required_columns = [
            "language",
            "total_articles",
            "average_score",
            "average_references",
            "total_characters",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in comparison.columns
        ]

        if missing_columns:

            st.error(
                "Language comparison is missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            comparison["language"] = (
                comparison["language"].astype(str).str.lower()
            )

            language_rows = {
                str(row["language"]).lower(): row
                for _, row in comparison.iterrows()
            }

            english = language_rows.get("en")
            french = language_rows.get("fr")

            if english is not None and french is not None:

                difference = (
                    float(english["average_score"])
                    - float(french["average_score"])
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "English Articles",
                    f"{int(english['total_articles']):,}",
                )

                c2.metric(
                    "French Articles",
                    f"{int(french['total_articles']):,}",
                )

                c3.metric(
                    "Completeness Difference",
                    f"{difference:+.2f} points",
                )

                st.divider()

                st.subheader("Comparison Highlights")

                highlight1, highlight2 = st.columns(2)

                with highlight1:

                    if english["average_score"] > french["average_score"]:

                        st.info(
                            "English has the higher average completeness "
                            "score in these dataset snapshots."
                        )

                    elif french["average_score"] > english["average_score"]:

                        st.info(
                            "French has the higher average completeness "
                            "score in these dataset snapshots."
                        )

                    else:

                        st.info(
                            "Both language datasets have the same "
                            "average completeness score."
                        )

                with highlight2:

                    if (
                        english["average_references"]
                        > french["average_references"]
                    ):

                        st.info(
                            "English has the higher average recorded "
                            "reference count."
                        )

                    elif (
                        french["average_references"]
                        > english["average_references"]
                    ):

                        st.info(
                            "French has the higher average recorded "
                            "reference count."
                        )

                    else:

                        st.info(
                            "Both language datasets have the same "
                            "average recorded reference count."
                        )

            else:

                st.warning(
                    "Both English and French language records were not "
                    "found. Check the language values in your Parquet files."
                )

            st.divider()

            st.subheader("Language Comparison Table")

            display_comparison = comparison.copy()

            display_comparison["language"] = (
                display_comparison["language"]
                .str.upper()
            )

            display_comparison = display_comparison.rename(
                columns={
                    "language": "Language",
                    "total_articles": "Total Articles",
                    "average_score": "Average Completeness",
                    "average_references": "Average References",
                    "total_characters": "Total Characters",
                }
            )

            st.dataframe(
                display_comparison,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            st.subheader("Average Completeness by Language")

            chart_comparison = comparison.copy()

            chart_comparison["language"] = (
                chart_comparison["language"]
                .map({
                    "en": "English",
                    "fr": "French",
                })
                .fillna(chart_comparison["language"])
            )

            fig = language_chart(chart_comparison)

            fig.update_layout(
                height=420,
                paper_bgcolor=PANEL,
                plot_bgcolor=PANEL,
                font=dict(color="#e2e8f0"),
                title=None,
                xaxis_title="Language",
                yaxis_title="Average Completeness Score",
                showlegend=False,
                margin=dict(
                    l=40,
                    r=30,
                    t=30,
                    b=50,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

            show_download(
                comparison,
                "wikipulse_language_comparison.csv",
                "📥 Download language comparison",
            )

            st.divider()

            section_title("Analysis Notes")

            st.markdown(
                """
                - **Total Articles:** Number of article records in each dataset.
                - **Average Completeness:** Mean calculated completeness score.
                - **Average References:** Mean recorded reference count.
                - **Total Characters:** Combined character count of articles.

                **Note:** These results describe the selected dataset snapshots.
                Differences between languages do not automatically indicate
                differences in factual accuracy or overall article quality.
                """
            )


# ============================================================
# PART 6: SMART ARTICLE SEARCH — DUCKDB
# ============================================================

elif page == "Article Search":

    section_title(
        "🔍 Smart Article Search",
        "Search and filter the complete selected-language dataset "
        "without loading all records into Pandas.",
    )

    parquet_path = get_parquet_path(language)

    if not Path(parquet_path).exists():

        st.error(
            f"Dataset file not found: {parquet_path}"
        )

        st.stop()

    search = st.text_input(
        "Search Wikipedia articles",
        placeholder="Enter an article title or keyword...",
        key="article_search_page",
    )

    st.divider()

    st.subheader("Advanced Filters")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:

        quality_filter = st.selectbox(
            "Quality Category",
            ["All"] + QUALITY_ORDER,
            key="search_quality_filter",
        )

    with filter_col2:

        reference_filter = st.selectbox(
            "Reference Status",
            [
                "All",
                "Has References",
                "No References",
            ],
            key="search_reference_filter",
        )

    with filter_col3:

        sort_option = st.selectbox(
            "Sort Results",
            [
                "Highest Score",
                "Lowest Score",
                "Most References",
                "Longest Articles",
            ],
            key="search_sort_option",
        )

    st.markdown("**Completeness Score Range**")

    score_col1, score_col2 = st.columns(2)

    with score_col1:

        min_score = st.number_input(
            "Minimum Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="search_min_score",
        )

    with score_col2:

        max_score = st.number_input(
            "Maximum Score",
            min_value=0,
            max_value=100,
            value=100,
            step=5,
            key="search_max_score",
        )

    display_limit = st.selectbox(
        "Results per page",
        [10, 25, 50, 100, 500],
        index=2,
        key="search_display_limit",
    )

    if min_score > max_score:

        st.error(
            "Minimum score cannot be greater than maximum score."
        )

        st.stop()

    # --------------------------------------------------------
    # BUILD SQL FILTERS
    # --------------------------------------------------------

    conditions = [
        "completeness_score BETWEEN ? AND ?"
    ]

    parameters = [
        float(min_score),
        float(max_score),
    ]

    if search.strip():

        conditions.append(
            "LOWER(article) LIKE LOWER(?)"
        )

        parameters.append(
            f"%{search.strip()}%"
        )

    if quality_filter == "Strong":

        conditions.append(
            "completeness_score >= 80"
        )

    elif quality_filter == "Good":

        conditions.append(
            "completeness_score >= 60 "
            "AND completeness_score < 80"
        )

    elif quality_filter == "Needs Improvement":

        conditions.append(
            "completeness_score >= 40 "
            "AND completeness_score < 60"
        )

    elif quality_filter == "Poor":

        conditions.append(
            "completeness_score < 40"
        )

    if reference_filter == "Has References":

        conditions.append(
            '"references" > 0'
        )

    elif reference_filter == "No References":

        conditions.append(
            '"references" = 0'
        )

    where_clause = " AND ".join(conditions)

    sort_columns = {
        "Highest Score": "completeness_score DESC",
        "Lowest Score": "completeness_score ASC",
        "Most References": '"references" DESC',
        "Longest Articles": "characters DESC",
    }

    order_clause = (
        sort_columns[sort_option]
        + ", article ASC"
    )

    # --------------------------------------------------------
    # QUERY DATASET
    # --------------------------------------------------------

    con = None

    try:

        with st.spinner("Searching the Wikimedia dataset..."):

            con = get_duckdb_connection()

            count_sql = f"""
                SELECT COUNT(*)
                FROM read_parquet(?)
                WHERE {where_clause}
            """

            count_parameters = [
                parquet_path,
                *parameters,
            ]

            total_results = con.execute(
                count_sql,
                count_parameters,
            ).fetchone()[0]

        st.divider()

        st.subheader("Search Results")

        metric_col1, metric_col2 = st.columns(2)

        metric_col1.metric(
            "Matching Articles",
            f"{total_results:,}",
        )

        metric_col2.metric(
            "Selected Language",
            language,
        )

        if total_results == 0:

            st.warning(
                "No articles found. Try a different keyword "
                "or adjust the filters."
            )

        else:

            total_pages = max(
                1,
                (
                    total_results + display_limit - 1
                ) // display_limit,
            )

            page_number = st.number_input(
                "Select page",
                min_value=1,
                max_value=int(total_pages),
                value=1,
                step=1,
                key=f"search_page_{total_pages}",
            )

            offset = (
                int(page_number) - 1
            ) * display_limit

            result_sql = f"""
                SELECT
                    language,
                    article,
                    url,
                    characters,
                    "references",
                    reference_density,
                    reference_status,
                    completeness_score
                FROM read_parquet(?)
                WHERE {where_clause}
                ORDER BY {order_clause}
                LIMIT ?
                OFFSET ?
            """

            result_parameters = [
                parquet_path,
                *parameters,
                display_limit,
                offset,
            ]

            with st.spinner("Loading selected page..."):

                display_results = con.execute(
                    result_sql,
                    result_parameters,
                ).fetchdf()

            st.caption(
                f"Page {int(page_number)} of {total_pages:,} "
                f"• Showing {len(display_results):,} of "
                f"{total_results:,} matching articles."
            )

            show_article_table(
                display_results,
                key="smart_article_search_results",
                height=500,
            )

            st.info(
                "Only the current page is loaded into memory. "
                "This keeps article search more memory-efficient."
            )

            st.download_button(
                "📥 Download Current Page (CSV)",
                data=csv_bytes(display_results),
                file_name=(
                    f"wikipulse_{language.lower()}_search_page_"
                    f"{int(page_number)}.csv"
                ),
                mime="text/csv",
                key="download_search_current_page",
            )

    except Exception as error:

        st.error(
            f"Article search failed: {error}"
        )

        st.caption(
            "If DuckDB reports a memory error, reduce its memory limit "
            "inside get_duckdb_connection() from 1GB to 512MB."
        )

    finally:

        if con is not None:
            con.close()

    st.caption(
        "Quality categories use this project's experimental "
        "completeness-score thresholds. They are not official "
        "Wikipedia ratings."
    )
    # ============================================================
# PART 7: KNOWLEDGE EXPLORER
# ============================================================


# ============================================================
# PART 7: KNOWLEDGE EXPLORER
# ============================================================

elif page == "Knowledge Explorer":
    section_title("🧠 Knowledge Explorer","Explore Wikipedia articles using completeness, references and content information.")
    st.caption("This explorer works with article-level metadata; the source dataset does not contain semantic entity relationships.")
    s,q=st.columns(2)
    with s: search_query=st.text_input("Search article",placeholder="Enter article title...",key="knowledge_search")
    with q: selected_quality=st.selectbox("Quality Category",["All"]+QUALITY_ORDER,key="knowledge_quality")
    f1,f2=st.columns(2)
    with f1: min_references=st.number_input("Minimum references",min_value=0,value=0,step=1,key="knowledge_min_references")
    with f2: sort_by=st.selectbox("Sort articles by",["Highest Completeness","Lowest Completeness","Most References","Longest Articles"],key="knowledge_sort")
    mode={"Highest Completeness":"top","Lowest Completeness":"bottom","Most References":"references","Longest Articles":"characters"}[sort_by]
    if mode in ("top","bottom"):
        results=get_article_results(language,mode=mode,limit=50000)
    else:
        sort_expr='TRY_CAST("references" AS BIGINT)' if mode=="references" else "TRY_CAST(characters AS BIGINT)"
        results=query_language(language,f"""
            SELECT article,url,TRY_CAST(completeness_score AS DOUBLE) AS completeness_score,
            TRY_CAST(characters AS BIGINT) AS characters,TRY_CAST("references" AS BIGINT) AS "references",
            TRY_CAST(reference_density AS DOUBLE) AS reference_density,reference_status
            FROM __PARQUET__ WHERE TRY_CAST("references" AS BIGINT)>={int(min_references)}
            ORDER BY {sort_expr} DESC NULLS LAST LIMIT 50000
        """)
    if not results.empty:
        if search_query.strip(): results=article_search_filter(results,search_query.strip())
        if selected_quality!="All":
            results["quality_category"]=results["completeness_score"].apply(quality_category)
            results=results[results["quality_category"]==selected_quality]
        results=results[results["references"].fillna(0)>=min_references]
    st.divider()
    st.subheader("Article Results")
    st.metric("Matching Articles (retrieved sample)",f"{len(results):,}")
    st.caption("DuckDB scans the Parquet dataset and returns at most 50,000 matching rows; a maximum of 1,000 rows are rendered.")
    if results.empty: st.info("No articles match the selected filters.")
    else:
        display_data=results.head(1000).copy()
        display_data["quality_category"]=display_data["completeness_score"].apply(quality_category)
        show_article_table(display_data,key="knowledge_explorer_table",height=500)
        show_download(results,"wikipulse_knowledge_explorer.csv","📥 Download Results (up to 50,000 records)")
    st.divider()
    st.subheader("Top 10 Articles by Completeness")
    top_articles=get_article_results(language,mode="top",limit=10)
    if not top_articles.empty:
        top_articles["quality_category"]=top_articles["completeness_score"].apply(quality_category)
        show_article_table(top_articles,key="knowledge_top_articles",height=400)
    st.caption("This explorer uses article-level metadata. It does not represent entity relationships or a semantic knowledge graph.")


# ============================================================
# PART 8: ARTICLE DETAIL EXPLORER
# ============================================================

elif page == "Article Detail Explorer":

    section_title(
        "📄 Article Detail Explorer",
        "Search for an individual Wikipedia article and inspect "
        "its completeness, references and dataset information.",
    )

    parquet_path = get_parquet_path(language)

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not Path(parquet_path).exists():

        st.error(
            f"Dataset file not found: {parquet_path}"
        )

        st.info(
            "Check that the selected language's Parquet file "
            "exists inside the data folder."
        )

        st.stop()

    # --------------------------------------------------------
    # ARTICLE SEARCH
    # --------------------------------------------------------

    search_query = st.text_input(
        "🔎 Search Wikipedia article",
        placeholder="Enter an article title, e.g. Artificial Intelligence",
        key="detail_article_search",
    )

    st.caption(
        f"Searching the {language} Wikipedia dataset."
    )

    if not search_query.strip():

        st.info(
            "Enter an article title to find matching articles."
        )

        st.markdown(
            """
            **What you can explore here:**

            - Individual article completeness score
            - Recorded reference count
            - Article character count
            - Reference density
            - Reference status
            - Article quality category
            - Wikipedia article link
            - Downloadable article report
            """
        )

    else:

        con = None

        try:

            con = get_duckdb_connection()

            # ------------------------------------------------
            # FIND MATCHING ARTICLES
            # ------------------------------------------------

            with st.spinner("Finding matching articles..."):

                matches = con.execute(
                    """
                    SELECT article
                    FROM read_parquet(?)
                    WHERE LOWER(article) LIKE LOWER(?)
                    ORDER BY
                        CASE
                            WHEN LOWER(article) = LOWER(?) THEN 0
                            ELSE 1
                        END,
                        article ASC
                    LIMIT 20
                    """,
                    [
                        parquet_path,
                        f"%{search_query.strip()}%",
                        search_query.strip(),
                    ],
                ).fetchdf()

            if matches.empty:

                st.warning(
                    "No matching articles found. "
                    "Try another title or keyword."
                )

            else:

                st.success(
                    f"{len(matches):,} matching article titles found."
                )

                selected_article = st.selectbox(
                    "Select an article",
                    matches["article"].tolist(),
                    key="detail_article_selector",
                )

                # --------------------------------------------
                # LOAD SELECTED ARTICLE
                # --------------------------------------------

                article_data = con.execute(
                    """
                    SELECT
                        language,
                        article,
                        url,
                        characters,
                        "references",
                        reference_density,
                        reference_status,
                        completeness_score
                    FROM read_parquet(?)
                    WHERE LOWER(article) = LOWER(?)
                    LIMIT 1
                    """,
                    [
                        parquet_path,
                        selected_article,
                    ],
                ).fetchdf()

                if article_data.empty:

                    st.warning(
                        "Article details could not be loaded."
                    )

                else:

                    article = article_data.iloc[0]

                    # ----------------------------------------
                    # SAFE VALUE FORMATTING
                    # ----------------------------------------

                    def safe_number(value, default=0):

                        if pd.isna(value):
                            return default

                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return default

                    score = safe_number(
                        article["completeness_score"]
                    )

                    references = safe_number(
                        article["references"]
                    )

                    characters = safe_number(
                        article["characters"]
                    )

                    density = safe_number(
                        article["reference_density"]
                    )

                    score = max(
                        0,
                        min(score, 100),
                    )

                    category = quality_category(score)

                    article_title = str(
                        article["article"]
                    )

                    article_language = str(
                        article["language"]
                    )

                    reference_status = str(
                        article["reference_status"]
                    )

                    article_url = clean_wikipedia_url(
                        article["url"]
                    )

                    # ----------------------------------------
                    # ARTICLE HEADER
                    # ----------------------------------------

                    st.divider()

                    st.markdown(
                        f"## {article_title}"
                    )

                    st.caption(
                        f"Language: {article_language} "
                        f"| Reference Status: {reference_status}"
                    )

                    if article_url:

                        st.link_button(
                            "🌐 Open Wikipedia Article",
                            article_url,
                        )

                    else:

                        st.info(
                            "A valid Wikipedia URL is not available "
                            "for this dataset record."
                        )

                    st.divider()

                    # ----------------------------------------
                    # ARTICLE METRICS
                    # ----------------------------------------

                    st.subheader("Article Statistics")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Completeness Score",
                        f"{score:.2f}/100",
                    )

                    col2.metric(
                        "References",
                        f"{int(references):,}",
                    )

                    col3.metric(
                        "Characters",
                        f"{int(characters):,}",
                    )

                    col4.metric(
                        "Reference Density",
                        f"{density:.3f}",
                    )

                    st.divider()

                    # ----------------------------------------
                    # QUALITY ASSESSMENT
                    # ----------------------------------------

                    st.subheader("Quality Assessment")

                    quality_col, reference_col = st.columns(2)

                    with quality_col:

                        st.markdown(
                            "**Experimental Quality Category**"
                        )

                        st.metric(
                            "Article Category",
                            category,
                        )

                        st.progress(
                            score / 100
                        )

                        st.caption(
                            f"Completeness score: {score:.2f}%"
                        )

                    with reference_col:

                        st.markdown(
                            "**Reference Information**"
                        )

                        st.metric(
                            "Reference Status",
                            reference_status,
                        )

                        st.metric(
                            "Recorded References",
                            f"{int(references):,}",
                        )

                    st.divider()

                    # ----------------------------------------
                    # ARTICLE SCORE INTERPRETATION
                    # ----------------------------------------

                    st.subheader("Article Assessment")

                    if category == "Strong":

                        st.success(
                            "This article falls in the Strong category "
                            "according to the project's experimental "
                            "completeness-score thresholds."
                        )

                    elif category == "Good":

                        st.info(
                            "This article falls in the Good category "
                            "according to the project's experimental "
                            "completeness-score thresholds."
                        )

                    elif category == "Needs Improvement":

                        st.warning(
                            "This article falls in the Needs Improvement "
                            "category according to the project's "
                            "experimental completeness-score thresholds."
                        )

                    else:

                        st.error(
                            "This article falls in the Poor category "
                            "according to the project's experimental "
                            "completeness-score thresholds."
                        )

                    st.caption(
                        "These categories are experimental indicators "
                        "based on this project's score thresholds. "
                        "They are not official Wikipedia ratings and "
                        "do not measure factual accuracy."
                    )

                    st.divider()

                    # ----------------------------------------
                    # ARTICLE DATA TABLE
                    # ----------------------------------------

                    st.subheader("Complete Article Data")

                    report = article_data.copy()

                    report["quality_category"] = category

                    st.dataframe(
                        report,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # ----------------------------------------
                    # DOWNLOAD ARTICLE REPORT
                    # ----------------------------------------

                    st.download_button(
                        "📥 Download Article Report (CSV)",
                        data=csv_bytes(report),
                        file_name=(
                            "wikipulse_article_detail_report.csv"
                        ),
                        mime="text/csv",
                        key="download_article_detail_report",
                    )

        except Exception as error:

            st.error(
                f"Unable to load article details: {error}"
            )

            st.caption(
                "Check the Parquet file, DuckDB installation "
                "and dataset column names."
            )

        finally:

            if con is not None:
                con.close()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "WikiPulse AI | Wikipedia Quality Intelligence | "
    "Wikimedia Structured Contents Dataset"
)

st.caption(
    "Completeness and reference indicators describe the processed "
    "dataset. They are not official Wikipedia quality or factual-accuracy scores."
)