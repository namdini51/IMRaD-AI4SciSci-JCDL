import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


LABEL_ORDER = ["I", "M", "R", "D", "NoIMRaD"]


def evaluate_predictions(y_true, y_pred, y_prob=None, label_order=None):

    if label_order is None:
        label_order = LABEL_ORDER

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=label_order,
        average="macro",
        zero_division=0
    )
    
    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=label_order,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=label_order,
        average="macro",
        zero_division=0
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=label_order,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_order
    )

    confusion_df = pd.DataFrame(
        cm,
        index=label_order,
        columns=label_order
    )

    metrics = {}
    metrics["accuracy"] = float(accuracy)
    metrics["macro_precision"] = float(macro_precision)
    metrics["macro_recall"] = float(macro_recall)
    metrics["macro_f1"] = float(macro_f1)
    metrics["num_samples"] = int(len(y_true))

    if y_prob is not None:
        y_prob = np.array(y_prob)
        metrics["probability_shape"] = list(y_prob.shape)

    return metrics, report_df, confusion_df


def build_article_layout_table(df, paper_id_col="paper_id", true_col="y_true", pred_col="y_pred", order_cols=None, true_layout_col="true_layout", pred_layout_col="pred_layout"):
    if order_cols is None:
        candidate_order_cols = [
            "section_order",
            "start_progression",
            "section_id"
        ]

        order_cols = []
        for col in candidate_order_cols:
            if col in df.columns:
                order_cols.append(col)

    required_cols = [paper_id_col, true_col, pred_col]

    missing_cols = []
    for col in required_cols:
        if col not in df.columns:
            missing_cols.append(col)

    if len(missing_cols) > 0:
        raise ValueError("Missing required columns: " + str(missing_cols))

    work = df.copy()

    sort_cols = [paper_id_col]
    for col in order_cols:
        sort_cols.append(col)

    work = work.sort_values(sort_cols).copy()

    records = []

    grouped = work.groupby(paper_id_col, sort=False)

    for paper_id, group in grouped:
        true_labels = group[true_col].astype(str).tolist()
        pred_labels = group[pred_col].astype(str).tolist()

        record = {
            paper_id_col: paper_id,
            true_layout_col: "->".join(true_labels),
            pred_layout_col: "->".join(pred_labels),
            "num_blocks": int(len(group)),
            "layout_exact_match": "->".join(true_labels) == "->".join(pred_labels),
        }

        records.append(record)

    layout_df = pd.DataFrame(records)

    return layout_df


def levenshtein_distance(seq1, seq2):
    """
    Compute Levenshtein edit distance between two sequences.
    """

    len1 = len(seq1)
    len2 = len(seq2)

    dp = []

    for i in range(len1 + 1):
        row = []
        for j in range(len2 + 1):
            row.append(0)
        dp.append(row)

    for i in range(len1 + 1):
        dp[i][0] = i

    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1

            delete_cost = dp[i - 1][j] + 1
            insert_cost = dp[i][j - 1] + 1
            substitute_cost = dp[i - 1][j - 1] + cost

            dp[i][j] = min(delete_cost, insert_cost, substitute_cost)

    return dp[len1][len2]


def add_layout_edit_distance(layout_df, true_layout_col="true_layout", pred_layout_col="pred_layout"):
    out = layout_df.copy()

    edit_distances = []
    normalized_distances = []

    for _, row in out.iterrows():
        true_seq = str(row[true_layout_col]).split("->")
        pred_seq = str(row[pred_layout_col]).split("->")

        dist = levenshtein_distance(true_seq, pred_seq)

        denom = max(len(true_seq), len(pred_seq))
        if denom == 0:
            norm_dist = 0.0
        else:
            norm_dist = dist / denom

        edit_distances.append(dist)
        normalized_distances.append(norm_dist)

    out["layout_edit_distance"] = edit_distances
    out["layout_normalized_edit_distance"] = normalized_distances

    return out


def evaluate_layout_predictions(df, paper_id_col="paper_id", true_col="y_true", pred_col="y_pred", order_cols=None):
    layout_df = build_article_layout_table(
        df=df,
        paper_id_col=paper_id_col,
        true_col=true_col,
        pred_col=pred_col,
        order_cols=order_cols
    )

    layout_df = add_layout_edit_distance(layout_df)

    exact_match_accuracy = layout_df["layout_exact_match"].mean()
    mean_edit_distance = layout_df["layout_edit_distance"].mean()
    mean_normalized_edit_distance = layout_df["layout_normalized_edit_distance"].mean()

    layout_metrics = {
        "num_articles": int(len(layout_df)),
        "layout_exact_match_accuracy": float(exact_match_accuracy),
        "mean_layout_edit_distance": float(mean_edit_distance),
        "mean_layout_normalized_edit_distance": float(mean_normalized_edit_distance),
    }

    if "num_blocks" in layout_df.columns:
        layout_metrics["mean_num_blocks"] = float(layout_df["num_blocks"].mean())
        layout_metrics["min_num_blocks"] = int(layout_df["num_blocks"].min())
        layout_metrics["max_num_blocks"] = int(layout_df["num_blocks"].max())

    return layout_metrics, layout_df


def print_layout_evaluation_summary(layout_metrics):
    print("Article-level layout evaluation")
    print("--------------------------------")
    print("Number of articles:", layout_metrics["num_articles"])
    print(
        "Layout exact-match accuracy:",
        round(layout_metrics["layout_exact_match_accuracy"], 4)
    )
    print(
        "Mean layout edit distance:",
        round(layout_metrics["mean_layout_edit_distance"], 4)
    )
    print(
        "Mean normalized layout edit distance:",
        round(layout_metrics["mean_layout_normalized_edit_distance"], 4)
    )

    if "mean_num_blocks" in layout_metrics:
        print("Mean number of blocks:", round(layout_metrics["mean_num_blocks"], 4))
        print("Min number of blocks:", layout_metrics["min_num_blocks"])
        print("Max number of blocks:", layout_metrics["max_num_blocks"])


def build_prediction_table(df_ids, y_true, y_pred, y_prob=None, label_order=None, prob_prefix="prob"):

    if label_order is None:
        label_order = LABEL_ORDER

    predictions_df = df_ids.copy()

    predictions_df["y_true"] = list(y_true)
    predictions_df["y_pred"] = list(y_pred)

    if y_prob is not None:
        y_prob = np.array(y_prob)

        if y_prob.shape[1] != len(label_order):
            raise ValueError(
                "y_prob column count does not match label_order length. "
                f"y_prob shape: {y_prob.shape}, label_order length: {len(label_order)}"
            )

        for i in range(len(label_order)):
            label = label_order[i]
            col_name = prob_prefix + "_" + label
            predictions_df[col_name] = y_prob[:, i]

    return predictions_df


def save_evaluation_outputs(output_dir, metrics, report_df, confusion_df, predictions_df=None, config=None):
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics.json")
    report_path = os.path.join(output_dir, "classification_report.csv")
    confusion_path = os.path.join(output_dir, "confusion_matrix.csv")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_df.to_csv(report_path)
    confusion_df.to_csv(confusion_path)

    if predictions_df is not None:
        predictions_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)

    if config is not None:
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


def print_evaluation_summary(metrics, confusion_df):

    print("Evaluation summary")
    print("------------------")
    print("Number of samples:", metrics["num_samples"])
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Macro precision:", round(metrics["macro_precision"], 4))
    print("Macro recall:", round(metrics["macro_recall"], 4))
    print("Macro-F1:", round(metrics["macro_f1"], 4))

    print()
    print("Confusion matrix:")
    print(confusion_df)


def reorder_probabilities_by_label_order(model_classes, y_prob, label_order=None):
    if label_order is None:
        label_order = LABEL_ORDER

    model_classes = list(model_classes)
    y_prob = np.array(y_prob)

    reordered = np.zeros((y_prob.shape[0], len(label_order)))

    for target_index in range(len(label_order)):
        label = label_order[target_index]

        if label not in model_classes:
            raise ValueError(
                "Label missing from model_classes: " + str(label)
            )

        source_index = model_classes.index(label)
        reordered[:, target_index] = y_prob[:, source_index]

    return reordered
