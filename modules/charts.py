import plotly.graph_objects as go


# ==================================================
# WIKIPULSE AI — CHART THEME
# ==================================================

BACKGROUND = "#111827"
PANEL = "#111827"
TEXT_COLOR = "#F8FAFC"
MUTED_COLOR = "#94A3B8"
GRID_COLOR = "#263449"
BORDER_COLOR = "#334155"

QUALITY_COLORS = {
    "Strong": "#6366F1",
    "Good": "#F0523A",
    "Needs Improvement": "#06C89B",
    "Poor": "#A855F7",
}

REFERENCE_COLORS = {
    "Moderate reference density": "#6366F1",
    "Low reference density": "#F0523A",
    "No recorded references": "#06C89B",
    "High reference density": "#A855F7",
}

LANGUAGE_COLORS = {
    "English": "#6366F1",
    "French": "#F0523A",
}


# ==================================================
# COMMON CHART THEME
# ==================================================

def apply_chart_theme(fig, height=400):

    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,

        font=dict(
            family="Arial, sans-serif",
            color=TEXT_COLOR,
            size=12,
        ),

        margin=dict(
            l=30,
            r=30,
            t=55,
            b=45,
        ),

        hoverlabel=dict(
            bgcolor="#1E293B",
            font_color=TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            font_size=12,
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(
                color=TEXT_COLOR,
                size=11,
            ),
        ),

        hovermode="closest",

    )

    fig.update_xaxes(
        color=MUTED_COLOR,
        gridcolor=GRID_COLOR,
        linecolor=BORDER_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(size=11),
    )

    fig.update_yaxes(
        color=MUTED_COLOR,
        gridcolor=GRID_COLOR,
        linecolor=BORDER_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(size=11),
    )

    return fig


# ==================================================
# ARTICLE QUALITY DONUT CHART
# ==================================================

def quality_chart(df):

    if df is None or df.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            labels=df["category"],
            values=df["articles"],

            hole=0.68,
            sort=False,

            marker=dict(
                colors=[
                    QUALITY_COLORS.get(category, "#64748B")
                    for category in df["category"]
                ],

                line=dict(
                    color=BACKGROUND,
                    width=3,
                ),
            ),

            textinfo="percent",
            textposition="inside",

            textfont=dict(
                size=13,
                color="#FFFFFF",
            ),

            hovertemplate=(
                "<b>%{label}</b><br>"
                "Articles: %{value:,}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Article Quality Distribution",
            x=0.02,
            font=dict(
                size=16,
                color=TEXT_COLOR,
            ),
        ),

        showlegend=True,

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.17,
            xanchor="center",
            x=0.5,
        ),

        uniformtext=dict(
            minsize=10,
            mode="hide",
        ),
    )

    return apply_chart_theme(fig, height=390)


# ==================================================
# REFERENCE INTELLIGENCE BAR CHART
# ==================================================

def reference_chart(df):

    if df is None or df.empty:
        return go.Figure()

    chart_df = df.sort_values(
        "articles",
        ascending=True,
    )

    max_value = chart_df["articles"].max()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["articles"],
            y=chart_df["status"],

            orientation="h",

            marker=dict(
                color=[
                    REFERENCE_COLORS.get(status, "#64748B")
                    for status in chart_df["status"]
                ],

                line=dict(width=0),

            ),

            text=chart_df["articles"].apply(
                lambda value: f"{value:,}"
            ),

            textposition="outside",
            cliponaxis=False,

            textfont=dict(
                color=TEXT_COLOR,
                size=11,
            ),

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Articles: %{x:,}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Reference Coverage",
            x=0.02,
            font=dict(
                size=16,
                color=TEXT_COLOR,
            ),
        ),

        showlegend=False,

        xaxis=dict(
            title="Number of Articles",
            range=[0, max_value * 1.30],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            tickformat="~s",
        ),

        yaxis=dict(
            title=None,
            showgrid=False,
            automargin=True,
        ),
    )

    fig = apply_chart_theme(fig, height=390)

    fig.update_layout(
        margin=dict(
            l=25,
            r=85,
            t=55,
            b=45,
        )
    )

    return fig


# ==================================================
# LANGUAGE COMPARISON CHART
# ==================================================

def language_chart(df):

    if df is None or df.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["language"],
            y=df["average_score"],

            marker=dict(
                color=[
                    LANGUAGE_COLORS.get(language, "#06C89B")
                    for language in df["language"]
                ],

                line=dict(width=0),
            ),

            text=df["average_score"].apply(
                lambda value: f"{value:.2f}"
            ),

            textposition="outside",

            textfont=dict(
                color=TEXT_COLOR,
                size=12,
            ),

            hovertemplate=(
                "<b>%{x}</b><br>"
                "Average Completeness: %{y:.2f}/100"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Language-wise Completeness",
            x=0.02,
            font=dict(
                size=16,
                color=TEXT_COLOR,
            ),
        ),

        showlegend=False,

        xaxis_title="Language",
        yaxis_title="Average Completeness Score",

        yaxis=dict(
            range=[0, 105],
            showgrid=True,
            gridcolor=GRID_COLOR,
        ),
    )

    return apply_chart_theme(fig, height=390)