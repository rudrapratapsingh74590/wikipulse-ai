
import pandas as pd


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

def get_dashboard_metrics(df):

    if df.empty:
        return {
            "total_articles": 0,
            "average_score": 0,
            "no_references": 0,
            "languages": 0
        }

    return {
        "total_articles": len(df),

        "average_score": round(
            df["completeness_score"].mean(), 2
        ),

        "no_references": int(
            (df["reference_status"] == "No recorded references").sum()
        ),

        "languages": (
            df["language"].nunique()
            if "language" in df.columns else 1
        )
    }


# --------------------------------------------------
# ARTICLE QUALITY DISTRIBUTION
# --------------------------------------------------

def get_quality_distribution(df):

    if df.empty:
        return pd.DataFrame()

    scores = pd.to_numeric(
        df["completeness_score"],
        errors="coerce"
    ).dropna()

    categories = pd.cut(
        scores,
        bins=[
            -float("inf"),
            40,
            60,
            80,
            float("inf")
        ],
        labels=[
            "Poor",
            "Needs Improvement",
            "Good",
            "Strong"
        ],
        right=False
    )

    return (
        categories
        .value_counts()
        .reindex(
            ["Strong", "Good", "Needs Improvement", "Poor"],
            fill_value=0
        )
        .rename_axis("category")
        .reset_index(name="articles")
    )


# --------------------------------------------------
# REFERENCE DISTRIBUTION
# --------------------------------------------------

def get_reference_distribution(df):

    if df.empty:
        return pd.DataFrame()

    return (
        df["reference_status"]
        .value_counts()
        .reset_index(name="articles")
        .rename(columns={"reference_status": "status"})
    )


# --------------------------------------------------
# FIND WEAK ARTICLES
# --------------------------------------------------

def find_weak_articles(df, limit=20):

    if df.empty:
        return pd.DataFrame()

    return df.nsmallest(
        limit,
        "completeness_score"
    )


# --------------------------------------------------
# LONG ARTICLES WITH LOW REFERENCES
# --------------------------------------------------

def find_long_articles_with_low_references(
    df,
    min_characters=2000
):

    if df.empty:
        return pd.DataFrame()

    filtered = df.loc[
        (df["characters"] >= min_characters)
        & (df["reference_density"] < 0.5)
    ]

    return filtered.sort_values(
        "characters",
        ascending=False
    )


# --------------------------------------------------
# ARTICLE QUALITY RANKING
# --------------------------------------------------

def get_article_rankings(df, limit=50):

    if df.empty or "completeness_score" not in df.columns:
        return pd.DataFrame()

    scores = pd.to_numeric(
        df["completeness_score"],
        errors="coerce"
    )

    valid = scores.notna()

    if not valid.any():
        return pd.DataFrame()

    # Select top scores without sorting the entire dataset.
    top_indices = scores.loc[valid].nlargest(limit).index

    result = df.loc[top_indices].copy()

    result["completeness_score"] = scores.loc[top_indices]

    def classify_quality(score):

        if score >= 80:
            return "Strong"

        elif score >= 60:
            return "Good"

        elif score >= 40:
            return "Needs Improvement"

        return "Poor"

    result["quality_category"] = (
        result["completeness_score"].apply(classify_quality)
    )

    # Preserve global rank, including ties.
    result["quality_rank"] = (
        scores.rank(method="min", ascending=False)
        .loc[top_indices]
        .astype(int)
    )

    sort_columns = ["completeness_score"]
    ascending_order = [False]

    if "article" in result.columns:
        sort_columns.append("article")
        ascending_order.append(True)

    result = result.sort_values(
        by=sort_columns,
        ascending=ascending_order
    )

    columns = [
        "quality_rank",
        "article",
        "language",
        "completeness_score",
        "quality_category",
        "references",
        "reference_status"
    ]

    available_columns = [
        col for col in columns
        if col in result.columns
    ]

    return result[available_columns]


# --------------------------------------------------
# ARTICLE QUALITY SUMMARY
# --------------------------------------------------

def get_quality_summary(df):

    if df.empty or "completeness_score" not in df.columns:
        return {}

    scores = pd.to_numeric(
        df["completeness_score"],
        errors="coerce"
    ).dropna()

    return {
        "strong": int((scores >= 80).sum()),

        "good": int(
            ((scores >= 60) & (scores < 80)).sum()
        ),

        "needs_improvement": int(
            ((scores >= 40) & (scores < 60)).sum()
        ),

        "poor": int((scores < 40).sum())
    }


# --------------------------------------------------
# KNOWLEDGE GRAPH DATA
# --------------------------------------------------

def get_knowledge_graph_data(df):

    if df.empty:
        return pd.DataFrame()

    required_columns = [
        "main_entity",
        "additional_entities"
    ]

    if not all(col in df.columns for col in required_columns):
        return pd.DataFrame()

    relationships = []

    for _, row in df.iterrows():

        source = row["main_entity"]
        targets = row["additional_entities"]

        if isinstance(source, dict):
            source = source.get("name") or source.get("label")

        if not isinstance(source, str) or not source.strip():
            continue

        if not isinstance(targets, list):
            continue

        for target in targets:

            if isinstance(target, dict):
                target = target.get("name") or target.get("label")

            if isinstance(target, str) and target.strip():

                if source != target:
                    relationships.append({
                        "source": source,
                        "target": target,
                        "article": row.get("article", "Unknown")
                    })

    return pd.DataFrame(relationships)
