"""Scientific aesthetic theme for the Streamlit GUI."""

from __future__ import annotations

SCI_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
  --ink: #0F2744;
  --slate: #4A6075;
  --teal: #1F6F7A;
  --teal-dark: #15545C;
  --sand: #F3F6F9;
  --panel: #FFFFFF;
  --line: #D7E0EA;
  --accent: #C45C26;
  --good: #2F6B4F;
  --warn: #A67C00;
  --bad: #8B2E2E;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
  color: var(--ink);
  font-size: 17px;
}

.stApp {
  background: linear-gradient(180deg, #EEF3F7 0%, #F7FAFC 40%, #FFFFFF 100%);
  overflow-x: hidden;
}

.block-container {
  max-width: 1180px;
  padding-top: 1.25rem;
  padding-bottom: 2.5rem;
  overflow-x: hidden;
}

[data-testid="column"] {
  min-width: 0 !important;
}

[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.55rem 0.7rem 0.45rem 0.7rem;
  overflow: hidden;
  min-height: 5.4rem;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
  overflow-wrap: anywhere;
  word-break: break-word;
}
[data-testid="stMetricValue"] {
  font-size: 1.35rem !important;
  line-height: 1.25 !important;
}

h1, h2, h3 {
  font-family: 'IBM Plex Serif', Georgia, serif !important;
  color: var(--ink) !important;
  letter-spacing: 0.01em;
}

h1 { font-size: 2.15rem !important; }
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.25rem !important; }

p, label, .stMarkdown, .stText, .stSelectbox, .stTextInput, .stNumberInput {
  font-size: 1.05rem !important;
  color: var(--ink) !important;
}

section[data-testid="stSidebar"] {
  background: #0F2744;
  min-width: 270px;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
  color: #F4F8FB !important;
}
section[data-testid="stSidebar"] .stRadio label {
  font-size: 1.05rem !important;
  color: #F4F8FB !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-baseweb="popover"] *,
section[data-testid="stSidebar"] [data-baseweb="menu"] * {
  color: #0F2744 !important;
}

.hero-banner {
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 6px solid var(--teal);
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  margin-bottom: 1.1rem;
}
.hero-banner h1 {
  margin: 0 0 0.35rem 0;
}
.hero-banner p {
  margin: 0;
  color: var(--slate) !important;
  font-size: 1.08rem !important;
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.overview-frame {
  width: 100%;
  max-width: 100%;
  overflow: hidden;
  background: #FFFFFF;
  border: 1px solid var(--line);
  border-radius: 10px;
  margin: 0.2rem 0 1rem 0;
}
.overview-frame img,
[data-testid="stImage"] img {
  display: block;
  width: 100% !important;
  max-width: 100% !important;
  height: auto !important;
  object-fit: contain;
}

a.github-link {
  display: inline-block;
  background: var(--ink);
  color: #F4F8FB !important;
  text-decoration: none;
  font-weight: 600;
  border-radius: 8px;
  padding: 0.45rem 0.9rem;
  margin: 0.2rem 0 0.8rem 0;
}
a.github-link:hover {
  background: var(--teal-dark);
  color: #FFFFFF !important;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  align-items: flex-start;
  margin: 0.35rem 0 0.9rem 0;
}

.panel-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.9rem;
}

.metric-chip {
  display: inline-block;
  max-width: 100%;
  background: var(--sand);
  border: 1px solid var(--line);
  color: var(--ink);
  border-radius: 999px;
  padding: 0.28rem 0.75rem;
  font-size: 0.98rem;
  font-weight: 600;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.rationale-box {
  background: #F7FBFC;
  border: 1px solid var(--line);
  border-left: 5px solid var(--teal);
  border-radius: 8px;
  padding: 1rem 1.1rem;
  font-size: 1.08rem;
  line-height: 1.55;
  color: var(--ink);
  overflow-wrap: anywhere;
}

.lit-card {
  background: #FFFFFF;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.7rem;
  overflow-wrap: anywhere;
}
.lit-card a {
  color: var(--teal-dark);
  font-weight: 600;
  text-decoration: none;
}
.lit-card a:hover {
  color: var(--accent);
  text-decoration: underline;
}
.quote {
  font-style: italic;
  color: var(--slate);
  margin-top: 0.45rem;
}

div.stButton > button {
  background: var(--teal);
  color: white;
  border: 1px solid var(--teal-dark);
  border-radius: 8px;
  font-size: 1.05rem;
  font-weight: 600;
  padding: 0.55rem 1rem;
  transition: background 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover {
  background: var(--teal-dark);
  color: white;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(15, 39, 68, 0.14);
}
div.stButton > button:focus {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: auto;
  width: 100%;
}

[data-testid="stPlotlyChart"] {
  background: #FFFFFF;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.35rem 0.35rem 0.6rem 0.35rem;
  overflow: visible;
  margin-bottom: 0.8rem;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.35rem;
  flex-wrap: wrap;
}

.statusbar {
  color: var(--slate) !important;
  font-size: 0.98rem !important;
}
section[data-testid="stSidebar"] .statusbar {
  color: #D7E0EA !important;
}
</style>
"""


def inject_theme() -> None:
    import streamlit as st

    st.markdown(SCI_CSS, unsafe_allow_html=True)
