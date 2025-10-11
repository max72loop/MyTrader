# app/ui/styles.py
import streamlit as st


PAGE_CFG = dict(
page_title="AI Invest Assistant",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded",
)


GLOBAL_CSS = r"""
<style>
:root {
--primary:#667eea; --secondary:#764ba2; --success:#43e97b;
--warning:#ffa502; --danger:#fa709a; --info:#4facfe;
}
.main { background:#f8f9fa; }
.info-card{background:white;border-radius:16px;padding:1.5rem;box-shadow:0 2px 8px rgba(0,0,0,.08);border-left:4px solid var(--primary);margin:1rem 0;transition:.3s}
.info-card:hover{box-shadow:0 4px 16px rgba(0,0,0,.12);transform:translateY(-2px)}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border-radius:16px;padding:1.5rem;text-align:center;box-shadow:0 4px 12px rgba(102,126,234,.3)}
.metric-value{font-size:2.5rem;font-weight:700;margin:.5rem 0}
.metric-label{font-size:.9rem;opacity:.9;text-transform:uppercase;letter-spacing:1px}
.section-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:1rem 1.5rem;border-radius:12px;margin:2rem 0 1rem;display:flex;align-items:center;box-shadow:0 4px 12px rgba(102,126,234,.3)}
.section-header h2{margin:0;font-size:1.5rem}
.category-badge{display:inline-block;padding:.3rem .8rem;border-radius:20px;font-size:.85rem;font-weight:600;margin:.2rem}
.badge-momentum{background:#667eea;color:white}.badge-valuation{background:#f093fb;color:white}.badge-growth{background:#4facfe;color:white}.badge-quality{background:#43e97b;color:white}.badge-risk{background:#fa709a;color:white}
.dataframe{font-size:.9rem}
.dataframe thead th{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white !important;font-weight:600;padding:1rem !important}
.dataframe tbody tr:hover{background:#f0f2ff !important}
.performance-indicator{display:inline-flex;align-items:center;padding:.5rem 1rem;border-radius:8px;font-weight:600;margin:.25rem}
.perf-positive{background:#d4edda;color:#155724}.perf-negative{background:#f8d7da;color:#721c24}.perf-neutral{background:#e2e3e5;color:#383d41}
[data-testid="stSidebar"]{background:white;border-right:2px solid #e9ecef}
.stButton>button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border:none;border-radius:10px;padding:.6rem 1.5rem;font-weight:600;transition:.3s}
.stButton>button:hover{transform:scale(1.05);box-shadow:0 6px 20px rgba(102,126,234,.4)}
[data-testid="stPopover"] button{background:white;border:2px solid #667eea;color:#667eea;border-radius:8px;padding:.5rem 1rem;font-weight:600;font-size:.85rem}
[data-testid="stPopover"] button:hover{background:#667eea;color:white}
.streamlit-expanderHeader{background:#f8f9fa;border-radius:8px;font-weight:600}
</style>
"""


def set_page_config():
    st.set_page_config(**PAGE_CFG)


def inject_global_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)