
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as component
import base64
from pathlib import Path
import streamlit.components.v1 as components

from trustnet_backend import (
    COMPONENT_LABELS,
    analyze_transactions,
    evaluate_scenarios,
    make_demo_transactions,
)

st.set_page_config(
    page_title="TrustNet | Fraud Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# TRUSTNET GLOBAL BACKGROUND
# ============================================================

APP_DIR = Path(__file__).resolve().parent
BACKGROUND_PATH = APP_DIR / "assets" / "background.png"

if BACKGROUND_PATH.exists():
    background_base64 = base64.b64encode(
        BACKGROUND_PATH.read_bytes()
    ).decode("utf-8")

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(
                    rgba(5, 12, 27, 0.30),
                    rgba(5, 12, 27, 0.48)
                ),
                url("data:image/png;base64,{background_base64}");

            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(5, 12, 25, 0.93);
            border-right: 1px solid rgba(96, 165, 250, 0.16);
        }}

        [data-testid="stMain"] {{
            background: transparent;
        }}

        [data-testid="stMainBlockContainer"] {{
            background: transparent;
        }}

        /* Keep content readable over the background */
        [data-testid="stMetric"],
        [data-testid="stExpander"],
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning(
        "Background image not found. "
        "Place background.png inside the assets folder."
    )


# ============================================================
# TRUSTNET INTRO SCREEN — MUST BE BEFORE DASHBOARD CONTENT
# ============================================================

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "trustnet_duck_logo.png"

if "intro_complete" not in st.session_state:
    st.session_state.intro_complete = False

if not st.session_state.intro_complete:

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu,
        [data-testid="stToolbar"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if LOGO_PATH.exists():
        logo = base64.b64encode(
            LOGO_PATH.read_bytes()
        ).decode("utf-8")

        logo_html = (
            f'<img src="data:image/png;base64,{logo}" '
            'alt="TrustNet duck">'
        )
    else:
        logo_html = (
            '<div style="color:#60a5fa;font-size:24px;">'
            'Logo missing'
            '</div>'
        )
        st.warning(
            f"Logo not found: {LOGO_PATH}\n\n"
            "Place trustnet_duck_logo.png in your assets folder."
        )

    intro_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
        * {{
            box-sizing: border-box;
        }}

        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
            color: #f8fafc;
        }}

        .welcome {{
            max-width: 1000px;
            min-height: 390px;
            margin: 20px auto;
            padding: 38px;
            display: grid;
            grid-template-columns: 0.85fr 1.15fr;
            align-items: center;
            gap: 36px;
            background: rgba(6, 16, 33, 0.94);
            border: 1px solid rgba(96, 165, 250, 0.22);
            border-radius: 24px;
            box-shadow: 0 22px 65px rgba(0, 0, 0, 0.3);
        }}

        .duck {{
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 260px;
            background: radial-gradient(
                circle,
                rgba(37, 99, 235, 0.15),
                transparent 70%
            );
        }}

        .duck img {{
            display: block;
            width: 100%;
            max-width: 270px;
            height: auto;
            object-fit: contain;
            animation: float 4s ease-in-out infinite;
        }}

        .eyebrow {{
            color: #60a5fa;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 3px;
            margin-bottom: 16px;
        }}

        h1 {{
            font-size: clamp(38px, 5vw, 56px);
            line-height: 1.08;
            letter-spacing: -2px;
            margin: 0;
        }}

        h1 span {{
            color: #60a5fa;
        }}

        .description {{
            color: #afbed2;
            font-size: 14px;
            line-height: 1.8;
            margin-top: 20px;
        }}

        .tagline {{
            margin-top: 24px;
            color: #e2e8f0;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
        }}

        .steps {{
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin-top: 22px;
        }}

        .step {{
            padding: 8px 10px;
            border: 1px solid #2b405e;
            border-radius: 30px;
            background: rgba(30, 41, 59, 0.6);
            color: #cbd5e1;
            font-size: 10px;
        }}

        @keyframes float {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-7px); }}
        }}

        @media (max-width: 650px) {{
            .welcome {{
                grid-template-columns: 1fr;
                padding: 24px;
                gap: 10px;
            }}

            .duck {{
                min-height: 150px;
            }}

            .duck img {{
                max-width: 170px;
            }}
        }}
    </style>
    </head>
    <body>
        <div class="welcome">
            <div class="duck">{logo_html}</div>

            <div>
               

                <h1>
                    Welcome to<br>
                    <span>TrustNet.</span>
                </h1>

                
                <div class="tagline">
                    TRACING FLOWS. BUILDING TRUST.
                </div>

            </div>
        </div>
    </body>
    </html>
    """

    components.html(
        intro_html,
        height=440,
        scrolling=False,
    )

    _, button_col, _ = st.columns([1, 1.2, 1])

    with button_col:
        if st.button(
            "Enter",
            type="primary",
            use_container_width=True,
            key="enter_trustnet",
        ):
            st.session_state.intro_complete = True
            st.rerun()

     

    # Do not render the dashboard until Enter TrustNet is clicked.
    st.stop()


# Your existing sidebar navigation and dashboard code
# must continue below this point.



# ============================================================
# THEME AND BACKGROUND
# ============================================================



st.markdown(
    f"""
    <style>
    .stApp {{
        
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: rgba(9, 15, 28, .98);
        border-right: 1px solid #263449;
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }}

    h1, h2, h3 {{
        color: #f1f5f9;
        letter-spacing: -.45px;
    }}

    p, label {{
        color: #b2bfd1;
    }}

    .eyebrow {{
        color: #f05261;
        font-size: 11px;
        letter-spacing: 3px;
        font-weight: 750;
    }}

    .muted {{
        color: #91a0b6;
        font-size: 13px;
    }}

    .metric-card {{
        background: rgba(17, 27, 45, .92);
        border: 1px solid #29364c;
        border-radius: 13px;
        padding: 18px 19px;
        min-height: 118px;
    }}

    .metric-label {{
        color: #94a3b8;
        font-size: 11px;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }}

    .metric-value {{
        color: #f8fafc;
        font-size: 29px;
        font-weight: 750;
        margin-top: 10px;
    }}

    .metric-note {{
        color: #94a3b8;
        font-size: 12px;
        margin-top: 4px;
    }}

    .panel {{
        background: rgba(15, 23, 42, .90);
        border: 1px solid #29364c;
        border-radius: 13px;
        padding: 17px;
    }}

    hr {{
        border-color: #29364c;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid #29364c;
        border-radius: 9px;
    }}

    .small-label {{
        text-transform: uppercase;
        color: #94a3b8;
        font-size: 10px;
        letter-spacing: 1.2px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TRUSTNET INTRO AND LOGO STYLES
# ============================================================

st.markdown(
    """
    <style>
    .tn-intro-wrap {
        min-height: 76vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 28px 0;
    }

    .tn-intro-card {
        width: 100%;
        max-width: 1080px;
        display: grid;
        grid-template-columns: 0.9fr 1.1fr;
        align-items: center;
        gap: 30px;
        padding: 42px;
        background:
            radial-gradient(
                circle at 15% 45%,
                rgba(37, 99, 235, 0.13),
                transparent 45%
            ),
            rgba(10, 18, 32, 0.94);
        border: 1px solid #263a55;
        border-radius: 24px;
        box-shadow: 0 25px 80px rgba(0, 0, 0, 0.28);
        animation: tn-enter 0.8s ease-out both;
    }

    .tn-intro-art {
        min-height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 18px;
        background:
            radial-gradient(
                circle,
                rgba(37, 99, 235, 0.17),
                transparent 68%
            );
    }

    .tn-intro-art img {
        width: 100%;
        max-width: 350px;
        height: auto;
        object-fit: contain;
        filter: drop-shadow(0 0 25px rgba(59, 130, 246, 0.2));
        animation: tn-float 4s ease-in-out infinite;
    }

    .tn-intro-eyebrow {
        color: #60a5fa;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .tn-intro-title {
        color: #f8fafc;
        font-size: clamp(38px, 5vw, 58px);
        line-height: 1.04;
        font-weight: 800;
        letter-spacing: -2px;
        margin: 0;
    }

    .tn-intro-title span {
        color: #60a5fa;
    }

    .tn-intro-description {
        color: #a9b9cf;
        font-size: 16px;
        line-height: 1.8;
        margin-top: 20px;
        max-width: 450px;
    }

    .tn-intro-tagline {
        color: #e2e8f0;
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 650;
        margin-top: 22px;
    }

    .tn-intro-steps {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 26px;
    }

    .tn-intro-step {
        color: #cbd5e1;
        border: 1px solid #2c405c;
        border-radius: 999px;
        padding: 8px 12px;
        background: rgba(30, 41, 59, 0.45);
        font-size: 12px;
    }

    .tn-sidebar-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 5px 0 16px;
    }

    .tn-sidebar-brand img {
        width: 66px;
        height: 66px;
        object-fit: contain;
    }

    .tn-sidebar-name {
        color: #f8fafc;
        font-weight: 800;
        font-size: 19px;
        letter-spacing: 1.5px;
    }

    .tn-sidebar-name span {
        color: #60a5fa;
    }

    .tn-sidebar-caption {
        color: #8496af;
        font-size: 9px;
        letter-spacing: 1.6px;
        margin-top: 4px;
    }

    @keyframes tn-float {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-8px);
        }
    }

    @keyframes tn-enter {
        from {
            opacity: 0;
            transform: translateY(14px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (max-width: 760px) {
        .tn-intro-card {
            grid-template-columns: 1fr;
            padding: 24px;
            gap: 12px;
        }

        .tn-intro-art {
            min-height: 180px;
        }

        .tn-intro-art img {
            max-width: 220px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# SESSION STATE AND INITIAL DEMO
# ============================================================

if "analysis" not in st.session_state:
    demo_df, ground_truth = make_demo_transactions(
        scenario="easy",
        seed=42,
        n_accounts=100,
        ring_sizes=[5],
    )

    st.session_state.analysis = analyze_transactions(
        demo_df
    )

    st.session_state.ground_truth = ground_truth
    st.session_state.data_mode = "Synthetic demo"
    st.session_state.original_input = demo_df
    st.session_state.evaluation_report = None


def apply_dataset(
    data: pd.DataFrame,
    mode: str,
    planted_rings=None,
) -> None:
    analysis = analyze_transactions(data)

    st.session_state.analysis = analysis
    st.session_state.ground_truth = planted_rings
    st.session_state.data_mode = mode
    st.session_state.original_input = data.copy()


def metric_card(
    col,
    label: str,
    value: str,
    note: str,
) -> None:
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ring_match_metrics(
    detected_df: pd.DataFrame,
    truth_rings: list[list[str]] | None,
):
    if truth_rings is None:
        return None

    detected = []

    if not detected_df.empty:
        detected = [
            set(r)
            for r in detected_df["accounts"].tolist()
        ]

    truth = [set(r) for r in truth_rings]

    if not truth:
        return {
            "recall": None,
            "precision": 0.0 if detected else 1.0,
            "false_alarms": len(detected),
            "no_ring_case": True,
        }

    candidates = []

    for i, d in enumerate(detected):
        for j, t in enumerate(truth):
            union = d | t
            score = (
                len(d & t) / len(union)
                if union else 0.0
            )

            if score >= 0.70:
                candidates.append((score, i, j))

    matched_truth = set()
    matched_detected = set()

    for score, i, j in sorted(
        candidates, reverse=True
    ):
        if (
            i not in matched_detected
            and j not in matched_truth
        ):
            matched_detected.add(i)
            matched_truth.add(j)

    return {
        "recall": len(matched_truth) / len(truth),
        "precision": (
            len(matched_detected) / len(detected)
            if detected else 0.0
        ),
        "false_alarms": len(detected) - len(matched_detected),
        "no_ring_case": False,
    }


# ============================================================
# INTERACTIVE NETWORK FIGURE
# ============================================================

def build_network_figure(
    analysis: dict,
    max_nodes: int = 110,
) -> go.Figure | None:

    account_df = analysis["accounts"]
    edge_df = analysis["edges"]

    if account_df.empty:
        return None

    ring_members = account_df.loc[
        account_df["flagged"], "account_id"
    ].tolist()

    chosen = list(dict.fromkeys(ring_members))

    for account in account_df["account_id"].tolist():
        if account not in chosen and len(chosen) < max_nodes:
            chosen.append(account)

    chosen = chosen[:max_nodes]
    chosen_set = set(chosen)

    sub_edges = edge_df[
        edge_df["sender"].isin(chosen_set)
        & edge_df["receiver"].isin(chosen_set)
    ].copy()

    ring_edges = sub_edges[sub_edges["is_ring_edge"]]

    ordinary_edges = sub_edges[
        ~sub_edges["is_ring_edge"]
    ].nlargest(350, "amount_total")

    sub_edges = pd.concat(
        [ring_edges, ordinary_edges]
    ).drop_duplicates(
        subset=["sender", "receiver"]
    )

    graph = nx.DiGraph()
    graph.add_nodes_from(chosen)

    graph.add_edges_from(
        zip(sub_edges["sender"], sub_edges["receiver"])
    )

    positions = nx.spring_layout(
        graph,
        seed=42,
        k=1.0 / max(np.sqrt(len(chosen)), 1),
        iterations=70,
    )

    normal_x, normal_y = [], []
    ring_x, ring_y = [], []
    arrow_annotations = []

    for row in sub_edges.itertuples(index=False):
        x0, y0 = positions[row.sender]
        x1, y1 = positions[row.receiver]

        if row.is_ring_edge:
            ring_x.extend([x0, x1, None])
            ring_y.extend([y0, y1, None])

            arrow_annotations.append({
                "x": x1,
                "y": y1,
                "ax": x0,
                "ay": y0,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 0.8,
                "arrowwidth": 1.2,
                "arrowcolor": "#ef4444",
                "opacity": 0.8,
            })
        else:
            normal_x.extend([x0, x1, None])
            normal_y.extend([y0, y1, None])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=normal_x,
        y=normal_y,
        mode="lines",
        line=dict(
            width=0.6,
            color="rgba(124,143,169,.22)",
        ),
        hoverinfo="skip",
        name="Other transactions",
    ))

    fig.add_trace(go.Scatter(
        x=ring_x,
        y=ring_y,
        mode="lines",
        line=dict(
            width=2,
            color="rgba(239,68,68,.88)",
        ),
        hoverinfo="skip",
        name="Detected ring links",
    ))

    risk_lookup = account_df.set_index(
        "account_id"
    ).to_dict("index")

    node_list = list(graph.nodes())

    node_colors = [
        "#ef4444"
        if risk_lookup[n]["flagged"]
        else "#22b8a5"
        for n in node_list
    ]

    node_sizes = [
        9 + 16 * float(risk_lookup[n]["risk_score"])
        for n in node_list
    ]

    custom = [
        [
            n,
            float(risk_lookup[n]["risk_score"]),
            str(risk_lookup[n]["risk_band"]),
            str(risk_lookup[n]["ring_ids"] or "—"),
        ]
        for n in node_list
    ]

    fig.add_trace(go.Scatter(
        x=[positions[n][0] for n in node_list],
        y=[positions[n][1] for n in node_list],
        mode="markers+text",
        text=node_list,
        textposition="top center",
        textfont=dict(size=9, color="#dce5f2"),
        customdata=custom,
        hovertemplate=(
            "Account: %{customdata[0]}<br>"
            "Risk score: %{customdata[1]:.3f}<br>"
            "Band: %{customdata[2]}<br>"
            "Ring: %{customdata[3]}<extra></extra>"
        ),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="#dbeafe"),
            opacity=0.95,
        ),
        name="Accounts",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=700,
        margin=dict(l=5, r=5, t=30, b=5),
        title=(
            "Directed transaction network · "
            "red edges = detected-ring connections"
        ),
        showlegend=True,
        annotations=arrow_annotations[:250],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig



# ============================================================
# TRUSTNET WELCOME SCREEN
# Place before SIDEBAR NAVIGATION
# ============================================================

    



LOGO_PATH = Path(__file__).resolve().parent / "assets" / "trustnet_duck_logo.png"


if "intro_complete" not in st.session_state:
    st.session_state.intro_complete = False

if not st.session_state.intro_complete:

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu,
        [data-testid="stToolbar"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if LOGO_PATH.exists():
        logo = base64.b64encode(
        LOGO_PATH.read_bytes()
        ).decode("utf-8")
        logo_html = (
        f'<img src="data:image/png;base64,{logo}" '
        'alt="trustnet_duck_logo.png">')
    else:
        logo_html = '<div class="logo-missing">TN</div>'

    intro_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
        * {{
            box-sizing: border-box;
        }}

        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
            color: #f8fafc;
        }}

        .welcome {{
            max-width: 1000px;
            min-height: 390px;
            margin: 20px auto;
            padding: 38px;
            display: grid;
            grid-template-columns: 0.85fr 1.15fr;
            align-items: center;
            gap: 36px;
            background: rgba(6, 16, 33, 0.94);
            border: 1px solid rgba(96, 165, 250, 0.22);
            border-radius: 24px;
            box-shadow: 0 22px 65px rgba(0, 0, 0, 0.3);
        }}

        .duck {{
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 260px;
            background: radial-gradient(
                circle,
                rgba(37, 99, 235, 0.15),
                transparent 70%
            );
        }}

        .duck img {{
            display: block;
            width: 100%;
            max-width: 270px;
            height: auto;
            object-fit: contain;
            animation: float 4s ease-in-out infinite;
        }}

        .eyebrow {{
            color: #60a5fa;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 3px;
            margin-bottom: 16px;
        }}

        h1 {{
            font-size: clamp(38px, 5vw, 56px);
            line-height: 1.08;
            letter-spacing: -2px;
            margin: 0;
        }}

        h1 span {{
            color: #60a5fa;
        }}

        .description {{
            color: #afbed2;
            font-size: 14px;
            line-height: 1.8;
            margin-top: 20px;
        }}

        .tagline {{
            margin-top: 24px;
            color: #e2e8f0;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
        }}

        .steps {{
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin-top: 22px;
        }}

        .step {{
            padding: 8px 10px;
            border: 1px solid #2b405e;
            border-radius: 30px;
            background: rgba(30, 41, 59, 0.6);
            color: #cbd5e1;
            font-size: 10px;
        }}

        @keyframes float {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-7px); }}
        }}

        @media (max-width: 650px) {{
            .welcome {{
                grid-template-columns: 1fr;
                padding: 24px;
                gap: 10px;
            }}

            .duck {{
                min-height: 150px;
            }}

            .duck img {{
                max-width: 170px;
            }}
        }}
    </style>
    </head>
    <body>
        <div class="welcome">
            <div class="duck">{logo_html}</div>

            <div>
               

                <h1>
                    Welcome to<br>
                    <span>TrustNet.</span>
                </h1>

                
                <div class="tagline">
                    TRACING FLOWS. BUILDING TRUST.
                </div>

            </div>
        </div>
    </body>
    </html>
    """

    components.html(
        intro_html,
        height=440,
        scrolling=False,
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================


with st.sidebar:
    if LOGO_PATH.exists():
        logo_data = base64.b64encode(
            LOGO_PATH.read_bytes()
        ).decode("utf-8")

        st.markdown(
            f"""
            <div class="tn-sidebar-brand">
                <img
                    src="data:image/png;base64,{logo_data}"
                    alt="TrustNet duck logo"
                >
                <div>
                    <div class="tn-sidebar-name">
                        TRUST<span>NET</span>
                    </div>
                    <div class="tn-sidebar-caption">
                        TRANSACTION INTELLIGENCE
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## ◈ TRUSTNET")
        st.caption("TRANSACTION INTELLIGENCE")

    st.divider()

    # Keep your existing navigation, dataset information,
    # and reset button below this line.


    st.divider()

    page = st.radio(
    "INVESTIGATION WORKFLOW",
    [
        "1. Transaction Data",
        "2. Transaction Matrix",
        "3. Matrix Calculations",
        "4. Risk Score",
        "5. Network & Rings",
        
    ],
    label_visibility="visible",
    )

    st.divider()
   
    if st.button(
        "Reset to demo dataset",
        use_container_width=True,
    ):
        try:
            demo_df, truth = make_demo_transactions(
                scenario="easy",
                seed=42,
                n_accounts=100,
                ring_sizes=[5],
            )

            apply_dataset(
                demo_df,
                "Synthetic demo",
                truth,
            )

            st.rerun()

        except Exception as exc:
            st.error(str(exc))


analysis = st.session_state.analysis
accounts = analysis["accounts"]
transactions = analysis["transactions"]
rings = analysis["rings"]
metrics = analysis["metrics"]


# ============================================================
# GLOBAL HEADER
# ============================================================




st.title("TrustNet Dashboard")




# ============================================================
# 1. TRANSACTION DATA
# ============================================================

if page == "1. Transaction Data":
    st.header("Transaction Data")
   
   
    st.divider()
  

    st.dataframe(
        transactions,
        use_container_width=True,
        hide_index=True,
        height=450,
    )

   

# ============================================================
# 2. TRANSACTION MATRIX
# ============================================================

elif page == "2. Transaction Matrix":
    st.header("Transaction Matrix")

   

    engine = analysis["engine_result"]
    A = np.asarray(engine["A"])
    account_ids = list(analysis["account_ids"])

   

    

    display_limit = st.slider(
        "Number of accounts to display",
        min_value=5,
        max_value=min(50, len(account_ids)),
        value=min(12, len(account_ids)),
    )

    matrix_df = pd.DataFrame(
        A[:display_limit, :display_limit],
        index=account_ids[:display_limit],
        columns=account_ids[:display_limit],
    )

    st.subheader("Matrix Preview")
    st.dataframe(
        matrix_df.style.format("{:.2f}"),
        use_container_width=True,
        height=450,
    )

   



# ============================================================
# 3. MATRIX CALCULATIONS
# ============================================================



elif page == "3. Matrix Calculations":
    from pathlib import Path

    st.header("Matrix Calculations")
    
    image_path = Path("assets/matrix_calculations.png")

    if image_path.exists():
        st.image(
            str(image_path),
            use_container_width=True,
        )
    else:
        st.warning(
            "Matrix calculation diagram not found. "
            "Place matrix_calculations.png inside the assets folder."
        )

    with st.expander("View the mathematical example"):
        import numpy as np

        A = np.array([
            [0, 100, 0],
            [0, 0, 80],
            [60, 0, 0],
        ])

        st.write("**Transaction matrix A**")
        st.latex(
            r"A = \begin{bmatrix}"
            r"0 & 100 & 0 \\"
            r"0 & 0 & 80 \\"
            r"60 & 0 & 0"
            r"\end{bmatrix}"
        )

        st.write("**Two-step paths: A²**")
        st.dataframe(
            np.linalg.matrix_power(A, 2),
            hide_index=True,
            use_container_width=True,
        )

        st.write("**Three-step paths: A³**")
        st.dataframe(
            np.linalg.matrix_power(A, 3),
            hide_index=True,
            use_container_width=True,
        )

       

# ============================================================
# 4. RISK SCORE
# ============================================================

elif page == "4. Risk Score":
    st.header("Risk Score Calculation")

    st.write(
        "TrustNet combines six normalised mathematical signals into "
        "a single score to help prioritise accounts for investigation."
    )

    weights = analysis["weights"]
    components = analysis["components"].copy()

    weight_rows = [
        {
            "Component": COMPONENT_LABELS[key],
            "Weight": value * 100,
        }
        for key, value in weights.items()
    ]

    weight_df = pd.DataFrame(weight_rows)

    st.subheader("How the score is weighted")

    st.dataframe(
        weight_df.style.format({"Weight": "{:.0f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        weight_df.set_index("Component")["Weight"]
    )

    st.subheader("Account risk ranking")

    risk_table = accounts.sort_values(
        "risk_score",
        ascending=False,
    ).copy()

    columns = [
        col for col in [
            "account_id",
            "risk_score",
            "risk_band",
            "flagged",
            "ring_ids",
        ]
        if col in risk_table.columns
    ]

    st.dataframe(
        risk_table[columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download risk scores",
        data=risk_table.to_csv(index=False),
        file_name="trustnet_risk_scores.csv",
        mime="text/csv",
    )

    st.info(
        "Risk scores are investigation-priority indicators, not "
        "calibrated probabilities of fraud or proof of wrongdoing."
    )


# ============================================================
# 5. NETWORK AND DETECTED RINGS
# ============================================================

elif page == "5. Network & Rings":
    st.header("Transaction Network & Suspicious Rings")

    st.write(
        "Each node represents an account. Directed edges represent "
        "transactions. Red nodes and red connections identify accounts "
        "and links associated with detected ring candidates."
    )

    fig = build_network_figure(analysis)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No network is available for the current dataset.")

    st.subheader("Accounts flagged for investigation")

    flagged = accounts.loc[accounts["flagged"]].sort_values(
        "risk_score",
        ascending=False,
    )

    if flagged.empty:
        st.success("No accounts were flagged by the current configuration.")
    else:
        st.dataframe(
            flagged,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Detected ring candidates")

    if rings.empty:
        st.info("No ring candidates were returned by the current analysis.")
    else:
        ring_view = rings.drop(
            columns=["accounts"],
            errors="ignore",
        )

        st.dataframe(
            ring_view,
            use_container_width=True,
            hide_index=True,
        )

        selected_ring = st.selectbox(
            "Inspect a ring",
            rings["ring_id"].tolist(),
        )

     
