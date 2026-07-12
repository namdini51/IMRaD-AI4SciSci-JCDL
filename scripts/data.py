from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


LABELS = ["I", "M", "R", "D", "NoIMRaD"]
LABEL_TO_ID = {
    "I": 0,
    "M": 1,
    "R": 2,
    "D": 3,
    "NoIMRaD": 4,
}
ID_TO_LABEL = {
    0: "I",
    1: "M",
    2: "R",
    3: "D",
    4: "NoIMRaD",
}


@dataclass
class DatasetConfig:
    master_path: str
    metadata_path: Optional[str] = None
    paper_id_col: str = "pmcid"
    section_id_col: str = "section_id"
    label_col: str = "IMRaD"
    split_col: str = "split"
    text_col: str = "citation"
    header_col: str = "section_header"
    position_col: str = "section_position"
    sentence_id_col: str = "sentence_id"
    progression_col: str = "progression"
    normalize_progression: bool = True


def load_master(config: DatasetConfig) -> pd.DataFrame:
    """Load the master TSV and apply basic cleaning/validation."""
    master_path = Path(config.master_path)
    if not master_path.exists():
        raise FileNotFoundError(f"Master file not found: {master_path}")

    df = pd.read_csv(master_path, sep="\t")
    df = standardize_columns(df, config)
    validate_required_columns(df, config)
    df = clean_basic_fields(df, config)
    df = normalize_labels(df, config)

    if config.normalize_progression:
        df = normalize_progression_if_needed(df, config)

    return df


def standardize_columns(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Return a copy with stripped column names."""
    out = df.copy()
    new_columns = []
    for col in out.columns:
        new_columns.append(str(col).strip())
    out.columns = new_columns
    return out


def validate_required_columns(df: pd.DataFrame, config: DatasetConfig) -> None:
    """Raise a clear error if the master file is missing required columns."""
    required = [
        config.paper_id_col,
        config.section_id_col,
        config.label_col,
        config.split_col,
        config.text_col,
        config.header_col,
        config.position_col,
        config.sentence_id_col,
        config.progression_col,
    ]

    missing = []
    for col in required:
        if col not in df.columns:
            missing.append(col)

    if len(missing) > 0:
        raise ValueError(f"Missing required columns: {missing}")


def clean_basic_fields(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Clean text/header/split fields and coerce numeric tracking columns."""
    out = df.copy()

    out[config.text_col] = out[config.text_col].fillna("").astype(str).str.strip()
    out[config.header_col] = out[config.header_col].fillna("").astype(str).str.strip()
    out[config.label_col] = out[config.label_col].fillna("").astype(str).str.strip()
    out[config.split_col] = out[config.split_col].fillna("").astype(str).str.strip().str.lower()

    numeric_cols = [config.position_col, config.sentence_id_col, config.progression_col]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    bad_numeric_rows = out[numeric_cols].isna().any(axis=1).sum()
    if bad_numeric_rows > 0:
        raise ValueError(f"Found {bad_numeric_rows} rows with missing/non-numeric position fields.")

    out[config.position_col] = out[config.position_col].astype(int)
    out[config.sentence_id_col] = out[config.sentence_id_col].astype(int)

    return out


def normalize_labels(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    out = df.copy()

    invalid = []
    unique_labels = sorted(out[config.label_col].dropna().unique().tolist())

    for label in unique_labels:
        if label not in LABELS:
            invalid.append(label)

    if len(invalid) > 0:
        raise ValueError(
            f"Unexpected labels found: {invalid}. Expected labels: {LABELS}"
        )

    return out


def normalize_progression_if_needed(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    out = df.copy()
    max_value = out[config.progression_col].max()

    if max_value > 1.5:
        out[config.progression_col] = out[config.progression_col] / 100.0

    return out


def validate_grouped_split(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """
    Check that each paper appears in exactly one split.

    Returns a per-paper split table. Raises an error if leakage is detected.
    """
    grouped = df.groupby(config.paper_id_col)[config.split_col].nunique().reset_index()
    grouped = grouped.rename(columns={config.split_col: "num_splits"})

    leaked = grouped[grouped["num_splits"] > 1]
    if len(leaked) > 0:
        example_ids = leaked[config.paper_id_col].head(10).tolist()
        raise ValueError(
            "Split leakage detected: at least one paper appears in multiple splits. "
            f"Examples: {example_ids}"
        )

    paper_splits = df[[config.paper_id_col, config.split_col]].drop_duplicates().copy()
    return paper_splits


def build_blocks(df: pd.DataFrame, config: DatasetConfig, k: int, fusion: str = "early", separator: str = " ",) -> pd.DataFrame:
    """
    Build one row per candidate section block using the first k sentences.
    params:
    - k:
        - Number of initial sentences to use. For the 01234 master file, valid values are 1-5.
    - fusion:
        - "early": concatenate first-k sentences into block_text.
        - "late": keep ordered sentences in sentence_1 - sentence_k columns for late fusion
    - separator:
        - String used to concatenate sentences for early fusion.

    return: A dataframe with one row per (paper, section_id) candidate block.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    if fusion not in ["early", "late"]:
        raise ValueError("fusion must be either 'early' or 'late'")

    sort_cols = [
        config.paper_id_col,
        config.section_id_col,
        config.position_col,
        config.sentence_id_col,
    ]

    work = df.sort_values(sort_cols).copy()

    # Take the first k available sentence rows per section block.
    work["_rank_within_section"] = (
        work.groupby([config.paper_id_col, config.section_id_col]).cumcount()
    )

    work = work[work["_rank_within_section"] < k].copy()
    work = work.drop(columns=["_rank_within_section"])

    if len(work) == 0:
        raise ValueError(f"No rows left after selecting first k={k} sentences.")

    group_cols = [config.paper_id_col, config.section_id_col]

    records = []
    grouped = work.groupby(group_cols, sort=False)

    for group_key, group in grouped:
        group = group.sort_values([config.position_col, config.sentence_id_col]).copy()

        paper_id = group.iloc[0][config.paper_id_col]
        section_id = group.iloc[0][config.section_id_col]
        label = group.iloc[0][config.label_col]
        split = group.iloc[0][config.split_col]
        header = group.iloc[0][config.header_col]

        if "pmid" in group.columns:
            pmid = group.iloc[0]["pmid"]
        else:
            pmid = None

        if "xml_section_order" in group.columns:
            section_order = group.iloc[0]["xml_section_order"]
        else:
            section_order = None

        sentences = group[config.text_col].tolist()
        sentence_ids = group[config.sentence_id_col].tolist()
        positions = group[config.position_col].tolist()
        progressions = group[config.progression_col].tolist()

        start_progression = min(progressions)
        end_progression = max(progressions)
        num_sentences_available = len(sentences)

        record = {
            "paper_id": paper_id,
            "pmcid": paper_id,
            "pmid": pmid,
            "section_id": section_id,
            "section_order": section_order,
            "label": label,
            "label_id": LABEL_TO_ID[label],
            "split": split,
            "section_header": header,
            "k_requested": k,
            "num_sentences_available": num_sentences_available,
            "start_progression": start_progression,
            "end_progression": end_progression,
            "sentence_ids": "|".join(str(x) for x in sentence_ids),
            "section_positions": "|".join(str(x) for x in positions),
        }

        if fusion == "early":
            clean_sentences = []
            for sent in sentences:
                sent = str(sent).strip()
                if sent != "":
                    clean_sentences.append(sent)
            record["block_text"] = separator.join(clean_sentences)
        else:
            sentence_number = 1
            for sent in sentences:
                record[f"sentence_{sentence_number}"] = sent
                sentence_number += 1

            while sentence_number <= k:
                record[f"sentence_{sentence_number}"] = ""
                sentence_number += 1

        records.append(record)

    blocks = pd.DataFrame(records)
    blocks = sort_blocks_for_sequence_modeling(blocks)
    return blocks


def sort_blocks_for_sequence_modeling(blocks: pd.DataFrame) -> pd.DataFrame:
    out = blocks.copy()

    if "section_order" in out.columns and out["section_order"].notna().any():
        out = out.sort_values(["paper_id", "section_order", "start_progression", "section_id"])
    else:
        out = out.sort_values(["paper_id", "start_progression", "section_id"])

    out = out.reset_index(drop=True)
    return out


def split_blocks(blocks: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = blocks[blocks["split"] == "train"].copy()
    val_df = blocks[blocks["split"].isin(["val", "valid", "validation", "dev"])].copy()
    test_df = blocks[blocks["split"] == "test"].copy()

    return train_df, val_df, test_df


def make_layout_table(blocks: pd.DataFrame) -> pd.DataFrame:
    """
    Convert block-level rows into one row per paper with the true layout sequence.
    """
    records = []
    grouped = blocks.groupby("paper_id", sort=False)

    for paper_id, group in grouped:
        group = sort_blocks_for_sequence_modeling(group)
        labels = group["label"].tolist()
        section_ids = group["section_id"].tolist()
        split_values = group["split"].unique().tolist()

        if len(split_values) != 1:
            raise ValueError(f"Paper {paper_id} appears in multiple splits: {split_values}")

        record = {
            "paper_id": paper_id,
            "split": split_values[0],
            "layout": "->".join(labels),
            "num_blocks": len(labels),
            "section_ids": "|".join(str(x) for x in section_ids),
        }
        records.append(record)

    return pd.DataFrame(records)


def attach_metadata(blocks: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    if config.metadata_path is None:
        return blocks.copy()

    metadata_path = Path(config.metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta = pd.read_csv(metadata_path)
    if "pmid" not in meta.columns:
        raise ValueError("Metadata file must contain a 'pmid' column.")

    out = blocks.merge(meta, on="pmid", how="left", suffixes=("", "_meta"))
    return out


def summarize_blocks(blocks: pd.DataFrame) -> None:
    """Print compact sanity-check summaries."""
    print("Rows / candidate blocks:", len(blocks))
    print("Unique papers:", blocks["paper_id"].nunique())
    print("\nSplit counts by block:")
    print(blocks["split"].value_counts(dropna=False))
    print("\nLabel counts by block:")
    print(blocks["label"].value_counts(dropna=False).reindex(LABELS))
    print("\nRequested k distribution:")
    print(blocks["k_requested"].value_counts(dropna=False).sort_index())
    print("\nAvailable sentence count distribution:")
    print(blocks["num_sentences_available"].value_counts(dropna=False).sort_index())


def load_blocks(config: DatasetConfig, k: int, fusion: str = "early", attach_pubmed_metadata: bool = False) -> pd.DataFrame:
    df = load_master(config)
    validate_grouped_split(df, config)
    blocks = build_blocks(df, config, k=k, fusion=fusion)

    if attach_pubmed_metadata:
        blocks = attach_metadata(blocks, config)

    return blocks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build first-k IMRaD candidate blocks from the master TSV.")
    parser.add_argument("--master_path", required=True)
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--fusion", choices=["early", "late"], default="early")
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    cfg = DatasetConfig(
        master_path=args.master_path,
        metadata_path=args.metadata_path,
    )

    block_df = load_blocks(cfg, k=args.k, fusion=args.fusion)
    summarize_blocks(block_df)

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        block_df.to_csv(output_path, sep="\t", index=False)
        print(f"\nSaved: {output_path}")
