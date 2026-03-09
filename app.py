import math
from datetime import date
from typing import Iterable

import streamlit as st

from config import TOP_K_DEFAULT
from rag.generator import generate_answer
from rag.retriever import retrieve_tweets

from pathlib import Path
import subprocess

MIN_DATE = date(2009, 5, 4)
MAX_DATE = date(2026, 1, 8)

INDEX_PATH = Path("vectorstore/hybrid_index")

if not INDEX_PATH.exists():
    print("Index not found. Building index...")
    subprocess.run(["python", "build_index.py"], check=True)

def fmt_date(d: date | None) -> str:
    if d is None:
        return ""
    return d.strftime("%d/%m/%Y")


def normalize_for_progress(values: Iterable[float | int | None]) -> list[int]:
    vals = []
    for v in values:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(0.0)

    if not vals:
        return []

    finite_vals = [v for v in vals if math.isfinite(v)]
    if not finite_vals:
        return [0 for _ in vals]

    lo = min(finite_vals)
    hi = max(finite_vals)

    if math.isclose(lo, hi):
        return [100 if math.isfinite(v) else 0 for v in vals]

    out = []
    for v in vals:
        if not math.isfinite(v):
            out.append(0)
        else:
            scaled = int(round(((v - lo) / (hi - lo)) * 100))
            out.append(max(0, min(100, scaled)))
    return out


def preview_text(text: str, limit: int = 110) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def render_loader(target, visible: bool, title: str = "Searching posts and generating answer...") -> None:
    if not visible:
        target.empty()
        return

    target.markdown(
        f"""
        <div class="loader-anchor">
            <div class="loader-wrap">
                <div class="loader-spinner"></div>
                <div class="loader-title">{title}</div>
                <div class="loader-sub">
                    Running hybrid retrieval, reranking, and answer generation.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>

        header[data-testid="stHeader"] {
            background: transparent;
        }

        :root {
            --rosewater: #f5e0dc;
            --flamingo: #f2cdcd;
            --pink: #f5c2e7;
            --mauve: #cba6f7;
            --red: #f38ba8;
            --maroon: #eba0ac;
            --peach: #fab387;
            --yellow: #f9e2af;
            --green: #a6e3a1;
            --teal: #94e2d5;
            --sky: #89dceb;
            --sapphire: #74c7ec;
            --blue: #89b4fa;
            --lavender: #b4befe;
            --text: #cdd6f4;
            --subtext1: #bac2de;
            --subtext0: #a6adc8;
            --overlay0: #6c7086;
            --surface0: #313244;
            --surface1: #45475a;
            --base: #1e1e2e;
            --mantle: #181825;
            --crust: #11111b;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(137,180,250,0.12), transparent 30%),
                radial-gradient(circle at top right, rgba(203,166,247,0.10), transparent 28%),
                linear-gradient(180deg, #11111b 0%, #181825 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 980px;
            margin: 0 auto;
            padding-top: 1.1rem;
            padding-bottom: 1rem;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }

        @media (max-width: 1100px) {
            .block-container {
                max-width: 900px;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                max-width: 100%;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                padding-top: 0.8rem;
                padding-bottom: 0.8rem;
            }
        }

        h1, h2, h3, h4, h5, h6, p, label, div {
            color: var(--text);
        }

        .hero {
            border: 1px solid rgba(186,194,222,0.14);
            background:
                linear-gradient(135deg,
                    rgba(137,180,250,0.18),
                    rgba(203,166,247,0.14)
                );
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border-radius: 20px;
            padding: 1.2rem 1.25rem;
            box-shadow:
                0 20px 45px rgba(0,0,0,0.35),
                inset 0 0 0 1px rgba(255,255,255,0.04);
            text-align: center;
            margin-top: 0.1rem;
            margin-bottom: 0.8rem;
        }

        .hero-title {
            font-size: 2.15rem;
            font-weight: 820;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
            line-height: 1.05;
        }

        .hero-sub {
            color: var(--subtext1);
            font-size: 0.92rem;
            margin: 0;
        }

        @media (max-width: 768px) {
            .hero {
                padding: 1rem 0.9rem;
                border-radius: 18px;
            }

            .hero-title {
                font-size: 1.7rem;
            }

            .hero-sub {
                font-size: 0.88rem;
            }
        }

        .loader-anchor {
            margin-top: 0.25rem;
            margin-bottom: 0.8rem;
        }

        .loader-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            min-height: 145px;
            border: 1px solid rgba(186,194,222,0.14);
            border-radius: 18px;
            background:
                linear-gradient(135deg,
                    rgba(137,180,250,0.12),
                    rgba(203,166,247,0.12)
                );
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 16px 36px rgba(0,0,0,0.25);
        }

        .loader-spinner {
            width: 82px;
            height: 82px;
            border-radius: 50%;
            border: 8px solid rgba(137,180,250,0.14);
            border-top: 8px solid #f5c2e7;
            border-right: 8px solid #cba6f7;
            border-bottom: 8px solid #89dceb;
            border-left: 8px solid rgba(166,227,161,0.12);
            animation: spinGlow 1s linear infinite;
            box-shadow:
                0 0 24px rgba(203,166,247,0.25),
                0 0 42px rgba(137,180,250,0.18);
        }

        .loader-title {
            font-size: 1rem;
            font-weight: 780;
            color: #cdd6f4;
            text-align: center;
        }

        .loader-sub {
            font-size: 0.88rem;
            color: #a6adc8;
            text-align: center;
            max-width: 500px;
        }

        @keyframes spinGlow {
            0% {
                transform: rotate(0deg) scale(1);
                filter: hue-rotate(0deg);
            }
            50% {
                transform: rotate(180deg) scale(1.03);
                filter: hue-rotate(14deg);
            }
            100% {
                transform: rotate(360deg) scale(1);
                filter: hue-rotate(0deg);
            }
        }

        .section-title {
            font-size: 1.12rem;
            font-weight: 760;
            text-align: center;
            margin-top: 0;
            margin-bottom: 0.05rem;
        }

        .section-sub {
            color: var(--subtext0);
            text-align: center;
            margin-bottom: 0.7rem;
            font-size: 0.9rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            justify-content: center;
            align-items: center;
            margin-top: 0.35rem;
            margin-bottom: 0.65rem;
        }

        .chip {
            background:
                linear-gradient(135deg,
                    rgba(137,180,250,0.12),
                    rgba(203,166,247,0.10)
                );
            color: var(--lavender);
            border: 1px solid rgba(137,180,250,0.16);
            padding: 0.38rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            box-shadow: 0 8px 18px rgba(0,0,0,0.14);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        .stat-card {
            border: 1px solid rgba(186,194,222,0.12);
            border-radius: 16px;
            background:
                linear-gradient(135deg,
                    rgba(49,50,68,0.72),
                    rgba(49,50,68,0.50)
                );
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            padding: 0.85rem 0.95rem;
            min-height: 100px;
            box-shadow:
                0 14px 28px rgba(0,0,0,0.18),
                inset 0 0 0 1px rgba(255,255,255,0.03);
        }

        .stat-label {
            color: var(--subtext0);
            font-size: 0.78rem;
            margin-bottom: 0.26rem;
            letter-spacing: 0.01em;
        }

        .stat-value {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 780;
            line-height: 1.28;
            word-break: break-word;
        }

        .answer-shell {
            border: 1px solid rgba(186,194,222,0.14);
            border-radius: 18px;
            background:
                linear-gradient(135deg,
                    rgba(30,30,46,0.82),
                    rgba(49,50,68,0.58)
                );
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow:
                0 14px 30px rgba(0,0,0,0.20),
                inset 0 0 0 1px rgba(255,255,255,0.03);
            margin-top: 0.3rem;
            margin-bottom: 0.8rem;
        }

        .source-meta {
            color: var(--subtext0);
            font-size: 0.8rem;
            margin-bottom: 0.55rem;
        }

        .source-text {
            line-height: 1.65;
            color: var(--text);
        }

        .small-center {
            text-align: center;
            color: var(--subtext0);
            font-size: 0.86rem;
            margin-top: 0.8rem;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input,
        div[data-testid="stDateInput"] input {
            background: rgba(49,50,68,0.82) !important;
            border: 1px solid rgba(186,194,222,0.16) !important;
            border-radius: 14px !important;
            color: #cdd6f4 !important;
            padding-top: 0.7rem !important;
            padding-bottom: 0.7rem !important;
            font-size: 0.98rem !important;
        }

        div[data-testid="stTextArea"] textarea {
            min-height: 92px !important;
        }

        div[data-testid="stDateInput"] > div {
            background: transparent !important;
        }

        div[data-testid="stSlider"] {
            padding-left: 0.2rem;
            padding-right: 0.2rem;
            margin-top: 0 !important;
            margin-bottom: 0.15rem;
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background-color: var(--mauve) !important;
            border-color: var(--mauve) !important;
            box-shadow: 0 0 0 5px rgba(203,166,247,0.18) !important;
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
            background: linear-gradient(90deg,var(--blue),var(--mauve)) !important;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg,var(--blue),var(--mauve)) !important;
            color: #11111b !important;
            border-radius: 14px !important;
            font-weight: 800 !important;
            height: 2.55rem !important;
            border: none !important;
            box-shadow: 0 10px 24px rgba(137,180,250,0.25);
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            filter: brightness(1.05);
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(186,194,222,0.12);
            border-radius: 14px;
            background:
                linear-gradient(135deg,
                    rgba(49,50,68,0.48),
                    rgba(49,50,68,0.34)
                );
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            overflow: hidden;
            box-shadow: 0 8px 18px rgba(0,0,0,0.14);
        }

        div[data-testid="stExpander"] summary {
            font-weight: 650;
        }

        div[data-testid="stTabs"] button {
            color: var(--subtext1) !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--lavender) !important;
        }

        div[data-testid="stProgressBar"] > div > div {
            background-color: rgba(69,71,90,0.85) !important;
            border-radius: 999px !important;
        }

        div[data-testid="stProgressBar"] div[role="progressbar"] {
            background: linear-gradient(90deg,var(--blue),var(--mauve)) !important;
            border-radius: 999px !important;
        }

        .stCodeBlock, pre, code {
            border-radius: 14px !important;
        }

        @media (max-width: 768px) {
            .chip-row {
                justify-content: flex-start;
            }

            .chip {
                font-size: 0.76rem;
                padding: 0.36rem 0.62rem;
            }

            .stat-card {
                min-height: auto;
            }

            .stat-value {
                font-size: 1.18rem;
            }

            .answer-shell {
                padding: 0.9rem;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Trump RAG", page_icon="🇺🇸", layout="wide")
inject_css()

if "result" not in st.session_state:
    st.session_state.result = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "error" not in st.session_state:
    st.session_state.error = None

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧢🇺🇸 Trump Tweets Retrieval System</div>
        <p class="hero-sub">
            Hybrid Retrieval + Cross-Encoder Reranking Architecture
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

loader_slot = st.empty()

st.markdown('<div class="section-title">Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Ask a question, choose a date window, and inspect sources only when you need them.</div>',
    unsafe_allow_html=True,
)

with st.form("search_form", clear_on_submit=False):
    query = st.text_area(
        "Question",
        placeholder="What did Trump say about NATO contributions?",
        height=120,
    )

    date_col_1, date_col_2 = st.columns(2)
    with date_col_1:
        date_from = st.date_input(
            "From",
            value=MIN_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            format="DD/MM/YYYY",
        )
    with date_col_2:
        date_to = st.date_input(
            "To",
            value=MAX_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            format="DD/MM/YYYY",
        )

    top_k = st.slider("Top-k", min_value=3, max_value=20, value=TOP_K_DEFAULT)
    run = st.form_submit_button("Run search", use_container_width=True)

st.markdown(
    f"""
    <div class="chip-row">
        <div class="chip">Allowed range: {fmt_date(MIN_DATE)} to {fmt_date(MAX_DATE)}</div>
        <div class="chip">Selected: {fmt_date(date_from)} to {fmt_date(date_to)}</div>
        <div class="chip">Top-k: {top_k}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if run:
    st.session_state.error = None
    st.session_state.result = None
    st.session_state.answer = None

    cleaned_query = (query or "").strip()
    if not cleaned_query:
        st.session_state.error = "Enter a query first."
    elif date_from > date_to:
        st.session_state.error = "Start date cannot be later than end date."
    else:
        render_loader(loader_slot, True)

        try:
            result = retrieve_tweets(
                query=cleaned_query,
                k=top_k,
                date_from=date_from.isoformat(),
                date_to=date_to.isoformat(),
            )
            docs = result["docs"]

            answer = None
            if docs:
                answer = generate_answer(
                    cleaned_query,
                    docs,
                    time_range=f"{fmt_date(date_from)} - {fmt_date(date_to)}",
                )

            st.session_state.result = {
                "query": cleaned_query,
                "date_from": date_from,
                "date_to": date_to,
                "top_k": top_k,
                **result,
            }
            st.session_state.answer = answer
        except Exception as e:
            st.session_state.error = f"{type(e).__name__}: {e}"
        finally:
            render_loader(loader_slot, False)

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.result is not None and not st.session_state.error:
    result = st.session_state.result
    docs = result["docs"]

    stats_col_1, stats_col_2, stats_col_3 = st.columns(3)
    with stats_col_1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Retrieved docs</div>
                <div class="stat-value">{len(docs)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stats_col_2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Date window</div>
                <div class="stat-value">{fmt_date(result["date_from"])}<br>{fmt_date(result["date_to"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stats_col_3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Top-k</div>
                <div class="stat-value">{result["top_k"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Primary output first. Evidence and scoring live below.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="answer-shell">', unsafe_allow_html=True)
    st.markdown("**Retrieval query**")
    st.code(result["retrieval_query"], language=None)

    if docs and st.session_state.answer:
        st.markdown("**Generated answer**")
        st.write(st.session_state.answer)
    elif docs:
        st.warning("Documents were found, but answer generation returned nothing.")
    else:
        st.warning("No documents found in this date range.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sources & retrieval details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Collapsed by default so the main answer stays clean.</div>',
        unsafe_allow_html=True,
    )

    tab_sources, tab_debug = st.tabs(["Sources", "Retrieval debug"])

    with tab_sources:
        if not docs:
            st.info("No sources to show.")
        else:
            rerank_vals = normalize_for_progress([doc.metadata.get("rerank_score", 0.0) for doc in docs])
            fused_vals = normalize_for_progress([doc.metadata.get("fused_score", 0.0) for doc in docs])
            dense_vals = normalize_for_progress([doc.metadata.get("dense_score", 0.0) for doc in docs])
            bm25_vals = normalize_for_progress([doc.metadata.get("bm25_score", 0.0) for doc in docs])

            for i, doc in enumerate(docs, start=1):
                date_str = doc.metadata.get("date", "")
                url = doc.metadata.get("post_url", "")
                title = f"[{i}] {date_str} • {preview_text(doc.page_content, 85)}"

                with st.expander(title, expanded=False):
                    st.markdown(
                        f"""
                        <div class="source-meta">
                            rerank={float(doc.metadata.get("rerank_score", 0.0)):.4f}
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            fused={float(doc.metadata.get("fused_score", 0.0)):.4f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("**Post text**")
                    st.markdown(f'<div class="source-text">{doc.page_content}</div>', unsafe_allow_html=True)

                    if url:
                        st.link_button("Open source post", url, use_container_width=True)

                    with st.expander("Scoring details", expanded=False):
                        score_col_1, score_col_2 = st.columns(2)

                        with score_col_1:
                            st.caption(f"Rerank: {float(doc.metadata.get('rerank_score', 0.0)):.4f}")
                            st.progress(rerank_vals[i - 1])

                            st.caption(f"Dense: {float(doc.metadata.get('dense_score', 0.0)):.4f}")
                            st.progress(dense_vals[i - 1])

                        with score_col_2:
                            st.caption(f"Fusion: {float(doc.metadata.get('fused_score', 0.0)):.4f}")
                            st.progress(fused_vals[i - 1])

                            st.caption(f"BM25: {float(doc.metadata.get('bm25_score', 0.0)):.4f}")
                            st.progress(bm25_vals[i - 1])

    with tab_debug:
        st.markdown("**Original query**")
        st.code(result["query"], language=None)

        st.markdown("**Retrieval query**")
        st.code(result["retrieval_query"], language=None)

        st.markdown("**Query variants**")
        for variant in result["query_variants"]:
            st.write(f"- {variant}")

        st.markdown("**Selected range**")
        st.write(f"{fmt_date(result['date_from'])} to {fmt_date(result['date_to'])}")

elif not st.session_state.error:
    st.markdown(
        """
        <div class="small-center">
            Run a query to see the answer and sources.
        </div>
        """,
        unsafe_allow_html=True,
    )