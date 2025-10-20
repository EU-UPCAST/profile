"""Standalone Streamlit UI for exploring ACM CCS trends."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
ARXIV_DATA_PATH = REPO_ROOT / "llmdap" / "profiler" / "arxiv_trends.csv.bz2"
HF_DATA_PATH = REPO_ROOT / "llmdap" / "profiler" / "hf_trends.csv.bz2"
ROOT_NODE = "Computing methodologies"
TopicPath = Tuple[str, ...]

try:
    from llmdap.profiler.metadata_schemas.acm_ccs import CCS_HIERARCHY
except ModuleNotFoundError:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from llmdap.profiler.metadata_schemas.acm_ccs import CCS_HIERARCHY


@dataclass(frozen=True)
class TopicSummary:
    path: TopicPath
    paper_count: int

    @property
    def label(self) -> str:
        return " / ".join(self.path)


def _iter_paths(tree: Dict | List | str, prefix: TopicPath = ()) -> Iterable[TopicPath]:
    if isinstance(tree, dict):
        for key, child in tree.items():
            new_prefix = prefix + (key,)
            yield new_prefix
            yield from _iter_paths(child, new_prefix)
    elif isinstance(tree, list):
        for item in tree:
            new_prefix = prefix + (item,)
            yield new_prefix
    else:
        raise TypeError(f"Unsupported tree node type: {type(tree)}")


def _get_subtree(tree: Dict | List, path: TopicPath) -> Dict | List:
    subtree = tree
    for key in path:
        if isinstance(subtree, dict):
            subtree = subtree[key]
        elif isinstance(subtree, list):
            if key not in subtree:
                raise KeyError(f"Leaf '{key}' not found under path {' / '.join(path[:-1])}")
            subtree = key
        else:
            raise KeyError(f"Cannot descend into node at {' / '.join(path)}")
    return subtree


def list_all_topic_paths() -> List[TopicPath]:
    return [path for path in _iter_paths(CCS_HIERARCHY)]


@st.cache_data(show_spinner=False)
def load_arxiv_trend_data(csv_path: Path = ARXIV_DATA_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find trend data at {csv_path}")

    df = pd.read_csv(csv_path, compression="bz2", parse_dates=["submission_date"])
    df["predicted_tag"] = df["predicted_tag"].apply(ast.literal_eval).apply(tuple)
    df["topic_path"] = df["predicted_tag"]
    return df


@st.cache_data(show_spinner=False)
def load_hf_trend_data(csv_path: Path = HF_DATA_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find trend data at {csv_path}")

    df = pd.read_csv(csv_path, compression="bz2", parse_dates=["createdAt", "last_modified"])
    df["predicted_tag"] = df["predicted_tag"].apply(ast.literal_eval).apply(tuple)
    df["topic_path"] = df["predicted_tag"]
    return df


@st.cache_data(show_spinner=False)
def compute_topic_counts(df: pd.DataFrame) -> Dict[TopicPath, int]:
    counts: Dict[TopicPath, int] = defaultdict(int)
    for path in df["topic_path"]:
        if not path:
            continue
        for depth in range(1, len(path) + 1):
            counts[path[:depth]] += 1
    return dict(counts)


def compute_monthly_counts(df: pd.DataFrame, date_column: str, id_column: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "count"])

    monthly_counts = (
        df.set_index(date_column)[id_column]
        .resample("MS")
        .count()
        .rename("count")
        .reset_index()
        .rename(columns={date_column: "month"})
    )
    return monthly_counts


def build_combined_monthly_trends(arxiv_monthly: pd.DataFrame, hf_monthly: pd.DataFrame) -> pd.DataFrame:
    series: List[pd.DataFrame] = []
    if not arxiv_monthly.empty:
        series.append(arxiv_monthly.assign(dataset="arXiv"))
    if not hf_monthly.empty:
        series.append(hf_monthly.assign(dataset="Hugging Face"))

    if not series:
        return pd.DataFrame(columns=["month", "count", "dataset"])

    combined = pd.concat(series, ignore_index=True)
    combined.sort_values("month", inplace=True)
    return combined


def filter_by_path(df: pd.DataFrame, path: TopicPath) -> pd.DataFrame:
    if not path:
        return df
    prefix_len = len(path)
    mask = df["topic_path"].apply(lambda topic: topic[:prefix_len] == path)
    return df[mask].copy()


def child_topic_summaries(path: TopicPath, counts: Dict[TopicPath, int]) -> List[TopicSummary]:
    try:
        subtree = _get_subtree(CCS_HIERARCHY, path)
    except KeyError:
        return []

    if isinstance(subtree, dict):
        child_names = subtree.keys()
    elif isinstance(subtree, list):
        child_names = subtree
    else:
        return []

    summaries = []
    for child in child_names:
        child_path = path + (child,)
        summaries.append(TopicSummary(child_path, counts.get(child_path, 0)))
    summaries.sort(key=lambda summary: summary.paper_count, reverse=True)
    return summaries


@lru_cache(maxsize=1)
def all_topic_paths() -> List[TopicPath]:
    return list_all_topic_paths()


def build_hierarchy_dataframe(counts: Dict[TopicPath, int]) -> pd.DataFrame:
    if not counts:
        return pd.DataFrame()

    rows = []
    for path, value in counts.items():
        node_id = " / ".join(path)
        parent_id = " / ".join(path[:-1]) if len(path) > 1 else ""
        label = path[-1]
        rows.append({"id": node_id, "parent": parent_id, "label": label, "value": value})
    return pd.DataFrame(rows)


def format_path(path: TopicPath) -> str:
    return " / ".join(path)


def topic_selector() -> TopicPath:
    st.sidebar.subheader("Topic Navigator")

    path: List[str] = [ROOT_NODE]
    current_node = CCS_HIERARCHY[ROOT_NODE]
    depth = 0

    while True:
        label = f"Level {depth + 1}"
        if isinstance(current_node, dict):
            options: List[Optional[str]] = [None] + sorted(current_node.keys())
            selection = st.sidebar.selectbox(
                label,
                options,
                format_func=lambda option: f"Stay at {path[-1]}" if option is None else option,
                key=f"selector_{depth}_{'_'.join(path).replace(' ', '_')}"
            )
            if selection is None:
                return tuple(path)
            path.append(selection)
            current_node = current_node[selection]
            depth += 1
        elif isinstance(current_node, list):
            leaf_options: List[Optional[str]] = [None] + sorted(current_node)
            selection = st.sidebar.selectbox(
                label,
                leaf_options,
                format_func=lambda option: f"Stay at {path[-1]}" if option is None else option,
                key=f"selector_leaf_{depth}_{'_'.join(path).replace(' ', '_')}"
            )
            if selection is None:
                return tuple(path)
            path.append(selection)
            return tuple(path)
        else:
            return tuple(path)


def render_topic_overview(
    selected_path: TopicPath,
    arxiv_df: pd.DataFrame,
    arxiv_counts: Dict[TopicPath, int],
    hf_df: pd.DataFrame,
    hf_counts: Dict[TopicPath, int],
) -> None:
    st.header(format_path(selected_path))

    arxiv_papers = filter_by_path(arxiv_df, selected_path)
    hf_models = filter_by_path(hf_df, selected_path)
    arxiv_monthly = compute_monthly_counts(arxiv_papers, "submission_date", "id")
    hf_monthly = compute_monthly_counts(hf_models, "createdAt", "modelId")
    combined_monthly = build_combined_monthly_trends(arxiv_monthly, hf_monthly)

    if not combined_monthly.empty:
        st.subheader("Monthly activity trend")
        trend_fig = px.line(
            combined_monthly,
            x="month",
            y="count",
            color="dataset",
            markers=True,
            labels={"month": "Month", "count": "Items per month", "dataset": "Dataset"},
        )
        trend_fig.update_layout(hovermode="x unified")
        trend_fig.update_xaxes(dtick="M1", tickformat="%b %Y")
        st.plotly_chart(trend_fig, use_container_width=True)

    st.subheader("arXiv Papers")
    total_papers = len(arxiv_papers)
    st.metric("Papers", total_papers)

    if total_papers:
        col1, col2 = st.columns(2)
        with col1:
            first_date = arxiv_papers["submission_date"].min()
            last_date = arxiv_papers["submission_date"].max()
            st.caption(f"Time span: {first_date.date()} → {last_date.date()}")

            arxiv_breakdown = arxiv_papers["categories"].value_counts().head(5)
            st.write("Top arXiv categories")
            st.bar_chart(arxiv_breakdown)

        with col2:
            if arxiv_monthly.empty:
                st.caption("No monthly submission activity to display.")
            else:
                st.write("Monthly submissions")
                monthly_fig = px.line(
                    arxiv_monthly,
                    x="month",
                    y="count",
                    markers=True,
                    labels={"month": "Month", "count": "Papers per month"},
                )
                monthly_fig.update_layout(hovermode="x unified")
                monthly_fig.update_xaxes(dtick="M1", tickformat="%b %Y")
                st.plotly_chart(monthly_fig, use_container_width=True)

        child_summaries = child_topic_summaries(selected_path, arxiv_counts)
        if child_summaries:
            child_df = pd.DataFrame(
                {
                    "Topic": [summary.label for summary in child_summaries],
                    "Papers": [summary.paper_count for summary in child_summaries],
                }
            )
            st.write("Immediate subtopics (papers)")
            st.table(child_df)

        display_df = arxiv_papers[["id", "submission_date", "categories", "predicted_tag"]].copy()
        display_df.sort_values("submission_date", ascending=False, inplace=True)
        display_df["predicted_tag"] = display_df["predicted_tag"].apply(lambda tags: " / ".join(tags))
        display_df["submission_date"] = display_df["submission_date"].dt.date
        display_df["arxiv_link"] = display_df["id"].apply(lambda paper_id: f"https://arxiv.org/abs/{paper_id}")
        display_df = display_df[["id", "submission_date", "categories", "predicted_tag", "arxiv_link"]]
        st.write("Matching papers")
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "arxiv_link": st.column_config.LinkColumn("arXiv", display_text="Open"),
            },
        )
    else:
        st.info("No papers mapped to this topic in the dataset.")

    st.divider()

    st.subheader("Hugging Face Models")
    total_models = len(hf_models)
    st.metric("Models", total_models)

    if total_models:
        col1, col2 = st.columns(2)
        with col1:
            first_created = hf_models["createdAt"].min()
            last_created = hf_models["createdAt"].max()
            st.caption(f"Creation span: {first_created.date()} → {last_created.date()}")

            if hf_models["last_modified"].notna().any():
                latest_update = hf_models["last_modified"].max()
                st.caption(f"Latest modification: {latest_update.date()}")

        with col2:
            if hf_monthly.empty:
                st.caption("No monthly model activity to display.")
            else:
                st.write("Monthly model additions")
                hf_trend_fig = px.line(
                    hf_monthly,
                    x="month",
                    y="count",
                    markers=True,
                    labels={"month": "Month", "count": "Models per month"},
                )
                hf_trend_fig.update_layout(hovermode="x unified")
                hf_trend_fig.update_xaxes(dtick="M1", tickformat="%b %Y")
                st.plotly_chart(hf_trend_fig, use_container_width=True)

        model_child_summaries = child_topic_summaries(selected_path, hf_counts)
        if model_child_summaries:
            child_df = pd.DataFrame(
                {
                    "Topic": [summary.label for summary in model_child_summaries],
                    "Models": [summary.paper_count for summary in model_child_summaries],
                }
            )
            st.write("Immediate subtopics (models)")
            st.table(child_df)

        display_df = hf_models[["modelId", "createdAt", "last_modified", "predicted_tag"]].copy()
        display_df.sort_values("last_modified", ascending=False, inplace=True)
        display_df["predicted_tag"] = display_df["predicted_tag"].apply(lambda tags: " / ".join(tags))
        display_df["createdAt"] = display_df["createdAt"].dt.date
        display_df["last_modified"] = display_df["last_modified"].dt.date
        display_df["hf_link"] = display_df["modelId"].apply(lambda model_id: f"https://huggingface.co/{model_id}")
        display_df = display_df[["modelId", "createdAt", "last_modified", "predicted_tag", "hf_link"]]
        st.write("Matching models")
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "hf_link": st.column_config.LinkColumn("Hugging Face", display_text="Open"),
            },
        )
    else:
        st.info("No models mapped to this topic in the dataset.")


def render_sunburst(
    counts: Dict[TopicPath, int],
    highlight_path: Optional[TopicPath] = None,
    dataset_label: str = "arXiv",
) -> None:
    hierarchy_df = build_hierarchy_dataframe(counts)
    if hierarchy_df.empty:
        st.warning(f"Hierarchy overview unavailable for {dataset_label}. No topic counts computed.")
        return

    fig = px.sunburst(
        hierarchy_df,
        names="label",
        parents="parent",
        values="value",
        ids="id",
        branchvalues="total"
    )

    if highlight_path:
        highlight_id = " / ".join(highlight_path)
        ids = list(fig.data[0].ids)
        marker = fig.data[0].marker
        base_colors = list(marker.colors) if marker.colors is not None else [None] * len(ids)
        if highlight_id in ids:
            updated_colors = base_colors.copy()
            for idx, node_id in enumerate(ids):
                if base_colors[idx] is None:
                    updated_colors[idx] = px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]
            for depth in range(1, len(highlight_path) + 1):
                ancestor_id = " / ".join(highlight_path[:depth])
                if ancestor_id in ids:
                    ancestor_idx = ids.index(ancestor_id)
                    updated_colors[ancestor_idx] = "#1f77b4"
            marker.colors = updated_colors
            marker.line.width = 1
            marker.line.color = "white"

    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
    st.subheader(f"Hierarchy overview ({dataset_label})")
    st.plotly_chart(fig, config={"responsive": True})


def main() -> None:
    st.set_page_config(page_title="ACM AI Trend Explorer", layout="wide")
    st.title("ACM AI Trend Explorer")
    st.caption("Explore arXiv papers and Hugging Face models mapped to the customized ACM CCS hierarchy.")

    arxiv_df = load_arxiv_trend_data()
    arxiv_counts = compute_topic_counts(arxiv_df)
    hf_df = load_hf_trend_data()
    hf_counts = compute_topic_counts(hf_df)

    selector_col, overview_col = st.columns([1, 2])
    with selector_col:
        selected_path = topic_selector()
        render_sunburst(arxiv_counts, selected_path, dataset_label="arXiv")
        render_sunburst(hf_counts, selected_path, dataset_label="Hugging Face")

    with overview_col:
        render_topic_overview(selected_path, arxiv_df, arxiv_counts, hf_df, hf_counts)


if __name__ == "__main__":
    main()
