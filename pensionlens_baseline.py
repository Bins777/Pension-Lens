from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")                    # headless rendering (server / SSH)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, label_binarize
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 — CONFIGURATION
# ============================================================

# ── Tabular features used for the baseline (NO topology) ──────────────────────
# These columns come directly from Form 5500 or from fee/manager aggregation
# that does NOT require traversing graph edges.
TABULAR_FEATURES: List[str] = [
    "num_participants",           # total active participants (Form 5500 field)
    "num_managers",               # count of distinct hired asset managers
    "total_fees",                 # total compensation paid to all managers
    "top_manager_concentration",  # fraction of fees to single top manager (0-1)
    "fee_percentile",             # fund's fee rank percentile among all funds
    "same_parent_fee_ratio",      # fraction of fees to managers in same parent
                                  #   group (identified by fuzzy name matching,
                                  #   NOT by graph traversal)
]

# Categorical column requiring encoding
CATEGORICAL_FEATURE: str = "fund_type"

# ── Graph-topology columns — MUST NOT appear in any baseline model ────────────
# Listed for explicitness and validated at load time.
TABULAR_FEATURES_BLOCKED: List[str] = [
    "betweenness_score",          # GDS Betweenness Centrality algorithm
    "community_concentration",    # GDS Louvain community partition
    "total_shared_managers",      # edge count in fund-fund graph
    "in_circular_pattern",        # GDS Strongly Connected Components flag
]

# ── Columns excluded for other reasons ────────────────────────────────────────
EXCLUDE_ALWAYS: List[str] = [
    "ein",                        # identifier, not a signal
    "plan_name",                  # free-text identifier
    "risk_score",                 # LEAKAGE: used to generate the label
    "has_manager_data",           # always 1 in the labeled subset (zero variance)
    "label",                      # target variable
]

TARGET_COL: str = "label"

# ── Fund-type vocabulary (from ingest script) ─────────────────────────────────
FUND_TYPE_CATEGORIES: List[str] = [
    "Single Employer",
    "Multi Employer",
    "Multiple Employer",
    "DFE",
    "Unknown",
]

# ── Outlier clipping thresholds (applied before scaling) ─────────────────────
# total_fees can be negative (fee rebates) and has extreme positive outliers.
FEE_CLIP_LOW:  float = 0.01    # 1st percentile
FEE_CLIP_HIGH: float = 0.99    # 99th percentile

# ── Training parameters ───────────────────────────────────────────────────────
RANDOM_STATE:   int = 42
TEST_SIZE:      float = 0.20
VAL_SIZE:       float = 0.15   # fraction of training set held out for XGB curves
CV_FOLDS:       int = 5
XGB_ROUNDS:     int = 300      # max boosting rounds (early stopping enabled)
EARLY_STOP_RND: int = 30

# ── Plot aesthetics ───────────────────────────────────────────────────────────
PALETTE = {
    "Logistic Regression": "#2563EB",   # blue
    "Random Forest":       "#16A34A",   # green
    "XGBoost":             "#DC2626",   # red
}
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8FAFC",
    "axes.edgecolor":    "#CBD5E1",
    "axes.labelcolor":   "#1E293B",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "grid.color":        "#E2E8F0",
    "grid.linewidth":    0.8,
})


# ============================================================
# SECTION 2 — DATA LOADING AND VALIDATION
# ============================================================

def load_labeled_data(data_dir: str | Path) -> pd.DataFrame:
    """
    Load the labeled fund feature matrix produced by pensionlens_labels.py.

    Parameters
    ----------
    data_dir : path containing pensionlens_labeled.csv

    Returns
    -------
    pd.DataFrame  — raw labeled feature matrix, shape (N, 16)

    Raises
    ------
    FileNotFoundError  if the CSV is missing
    ValueError         if required columns are absent
    """
    data_dir = Path(data_dir)
    fpath = data_dir / "pensionlens_labeled.csv"

    if not fpath.exists():
        raise FileNotFoundError(
            f"pensionlens_labeled.csv not found at {fpath}.\n"
            "Run pensionlens_labels.py first to generate labels."
        )

    df = pd.read_csv(fpath)
    print(f"[load] Loaded {len(df):,} labeled funds from {fpath}")

    # ── Validate all required columns are present ─────────────────────────────
    required = TABULAR_FEATURES + [CATEGORICAL_FEATURE, TARGET_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Validate that blocked topology columns are NOT used ───────────────────
    # (They exist in the file but must be filtered out — guard against accidental
    #  inclusion during future refactors.)
    for col in TABULAR_FEATURES_BLOCKED:
        assert col not in TABULAR_FEATURES, (
            f"TOPOLOGY GUARD VIOLATION: '{col}' is blocked but present in "
            f"TABULAR_FEATURES. This baseline must remain non-topological."
        )

    # ── Basic integrity checks ─────────────────────────────────────────────────
    assert df[TARGET_COL].isin([0, 1]).all(), \
        "Label column contains values other than 0 / 1."
    assert (df["has_manager_data"] == 1).all(), \
        "Labeled set should contain only funds with manager data."

    label_counts = df[TARGET_COL].value_counts()
    print(f"[load] Label distribution → "
          f"0 (risky/underperform): {label_counts.get(0, 0):,}  |  "
          f"1 (healthy/outperform): {label_counts.get(1, 0):,}  |  "
          f"imbalance ratio: {label_counts.min()/label_counts.max():.2f}")

    return df


# ============================================================
# SECTION 3 — FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional tabular features derived purely from node attributes.
    No edge traversal or graph algorithm output is introduced here.

    New columns appended
    --------------------
    log1p_total_fees        : log-scale transformation of total_fees
    log1p_num_participants  : log-scale transformation of num_participants
    fee_per_participant     : total_fees / max(num_participants, 1)
    is_single_manager       : binary flag, num_managers == 1
    is_concentrated         : binary flag, top_manager_concentration > 0.7
    """
    df = df.copy()

    # Log-transform heavy-tailed financial columns to reduce skew
    df["log1p_total_fees"] = np.log1p(np.maximum(df["total_fees"], 0))
    df["log1p_num_participants"] = np.log1p(
        pd.to_numeric(df["num_participants"], errors="coerce").fillna(0)
    )

    # Fee burden per plan participant (higher ⟹ potentially worse governance)
    participants = pd.to_numeric(df["num_participants"], errors="coerce").fillna(0)
    df["fee_per_participant"] = df["total_fees"] / np.maximum(participants, 1)
    df["log1p_fee_per_participant"] = np.log1p(
        np.maximum(df["fee_per_participant"], 0)
    )

    # Binary flags capturing threshold-based signals
    df["is_single_manager"]  = (df["num_managers"] == 1).astype(int)
    df["is_concentrated"]    = (df["top_manager_concentration"] > 0.70).astype(int)

    return df


def encode_fund_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode the 'fund_type' categorical column.
    Drops the first dummy to avoid perfect multicollinearity.

    Returns
    -------
    (encoded_df, dummy_column_names)
    """
    dummies = pd.get_dummies(
        df[CATEGORICAL_FEATURE],
        prefix="ftype",
        drop_first=True,    # avoid dummy trap
        dtype=float,
    )
    # Ensure all known categories produce a column even if absent in this split
    for cat in FUND_TYPE_CATEGORIES[1:]:               # skip first (dropped)
        col = f"ftype_{cat}"
        if col not in dummies.columns:
            dummies[col] = 0.0

    df = pd.concat([df, dummies], axis=1)
    return df, dummies.columns.tolist()


def build_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Assemble the final (X, y) arrays used for model training.

    Processing order
    ----------------
    1. Feature engineering (derived columns)
    2. Fund-type one-hot encoding
    3. Outlier clipping on total_fees
    4. Fill remaining NaNs with column medians
    5. Collect final feature names list (for importance plots)

    Returns
    -------
    X : np.ndarray  shape (N, F)
    y : np.ndarray  shape (N,)
    feature_names : List[str]
    """
    df = engineer_features(df)
    df, dummy_cols = encode_fund_type(df)

    # Core tabular features + engineered features + dummies
    engineered_features = [
        "log1p_total_fees",
        "log1p_num_participants",
        "log1p_fee_per_participant",
        "is_single_manager",
        "is_concentrated",
    ]
    feature_names = TABULAR_FEATURES + engineered_features + dummy_cols

    # ── Outlier clipping on total_fees before scaling ─────────────────────────
    lo = df["total_fees"].quantile(FEE_CLIP_LOW)
    hi = df["total_fees"].quantile(FEE_CLIP_HIGH)
    df["total_fees"] = df["total_fees"].clip(lo, hi)

    # ── Numeric coercion + NaN fill ───────────────────────────────────────────
    X_df = df[feature_names].copy()
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df = X_df.fillna(X_df.median())

    X = X_df.values.astype(np.float32)
    y = df[TARGET_COL].values.astype(int)

    print(f"[features] Feature matrix shape : {X.shape}  "
          f"(strictly non-topological, {len(feature_names)} features)")
    print(f"[features] Features used         : {feature_names}")

    return X, y, feature_names


# ============================================================
# SECTION 4 — DATA SPLITTING
# ============================================================

def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Stratified train / val / test split.

    Split proportions  (approximately)
    -----------------------------------
    Train : 68 %
    Val   : 12 %   (used for XGBoost early stopping and loss curves)
    Test  : 20 %   (held out until final evaluation)

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )
    print(
        f"[split] Train: {len(y_train):,}  |  "
        f"Val: {len(y_val):,}  |  "
        f"Test: {len(y_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# SECTION 5 — MODEL DEFINITIONS
# ============================================================

def _class_weight_ratio(y: np.ndarray) -> float:
    """Compute pos_weight = count(negative) / count(positive) for XGBoost."""
    neg, pos = np.bincount(y)
    return neg / max(pos, 1)


def build_logistic_regression(y_train: np.ndarray) -> Pipeline:
    """
    Logistic Regression pipeline with RobustScaler.

    Hyperparameters
    ---------------
    solver      : lbfgs   (supports L2, memory-efficient for small datasets)
    C           : 0.1     (moderate regularisation to handle small N)
    class_weight: balanced (corrects for 75/25 imbalance)
    max_iter    : 2000    (ensure convergence)

    RobustScaler is used instead of StandardScaler because total_fees and
    fee_per_participant have heavy tails even after clipping.
    """
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])


def build_random_forest(y_train: np.ndarray) -> RandomForestClassifier:
    """
    Random Forest classifier.

    Hyperparameters
    ---------------
    n_estimators  : 500   (enough for stable OOB, target AUC ~0.946 from lit.)
    max_depth     : None  (fully grown trees; controlled by min_samples_leaf)
    min_samples_leaf: 2   (prevents memorisation on the small dataset)
    max_features  : 'sqrt' (standard for classification)
    class_weight  : 'balanced_subsample' (per-tree rebalancing for imbalance)
    oob_score     : True  (free cross-validation estimate without extra split)

    Note: Random Forest does NOT require feature scaling.
    """
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_xgboost(y_train: np.ndarray) -> XGBClassifier:
    """
    XGBoost gradient-boosted trees classifier.

    Hyperparameters
    ---------------
    n_estimators    : 300   (upper bound; early stopping will reduce this)
    max_depth       : 4     (shallow trees → lower variance on small dataset)
    learning_rate   : 0.05  (slow learning with more rounds)
    subsample       : 0.8   (row subsampling → regularisation)
    colsample_bytree: 0.8   (feature subsampling per tree)
    scale_pos_weight: neg/pos ratio → corrects class imbalance
    eval_metric     : logloss (enables training / validation loss tracking)
    use_label_encoder: False (silences deprecation warning)
    """
    scale_pos_weight = _class_weight_ratio(y_train)
    return XGBClassifier(
        n_estimators=XGB_ROUNDS,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP_RND,
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


# ============================================================
# SECTION 6 — TRAINING PIPELINE
# ============================================================

def train_all_models(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
    feature_names: List[str],
) -> Dict:
    """
    Fit all three baseline models and return a results bundle.

    Returns
    -------
    dict with keys: 'lr', 'rf', 'xgb'
    Each value is a dict with:
        model       : fitted model / pipeline
        y_pred      : hard predictions on test set
        y_prob      : class-1 probability on test set
        train_losses: list[float] — per-round log-loss on train (XGB only)
        val_losses  : list[float] — per-round log-loss on val   (XGB only)
    """
    results = {}
    scaler_for_lr = RobustScaler()

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("\n[train] Fitting Logistic Regression ...")
    lr_pipe = build_logistic_regression(y_train)
    lr_pipe.fit(X_train, y_train)

    # Collect per-iteration log-loss using warm-start (manual learning curve)
    lr_train_losses, lr_val_losses = _lr_loss_curve(X_train, X_val, y_train, y_val)

    results["lr"] = dict(
        model=lr_pipe,
        y_pred=lr_pipe.predict(X_test),
        y_prob=lr_pipe.predict_proba(X_test)[:, 1],
        train_losses=lr_train_losses,
        val_losses=lr_val_losses,
        label="Logistic Regression",
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("[train] Fitting Random Forest (500 trees) ...")
    rf = build_random_forest(y_train)
    rf.fit(X_train, y_train)
    print(f"        OOB score: {rf.oob_score_:.4f}")

    # Staged predictions over growing n_estimators for loss curve
    rf_train_losses, rf_val_losses = _rf_staged_loss(
        rf, X_train, X_val, y_train, y_val
    )

    results["rf"] = dict(
        model=rf,
        y_pred=rf.predict(X_test),
        y_prob=rf.predict_proba(X_test)[:, 1],
        train_losses=rf_train_losses,
        val_losses=rf_val_losses,
        label="Random Forest",
    )

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("[train] Fitting XGBoost ...")
    xgb = build_xgboost(y_train)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    best_round = xgb.best_iteration
    print(f"        Best boosting round (early stop): {best_round}")

    # XGBoost records exact log-loss per round via evals_result_
    evals = xgb.evals_result_
    results["xgb"] = dict(
        model=xgb,
        y_pred=xgb.predict(X_test),
        y_prob=xgb.predict_proba(X_test)[:, 1],
        train_losses=evals["validation_0"]["logloss"],
        val_losses=evals["validation_1"]["logloss"],
        best_round=best_round,
        label="XGBoost",
    )

    return results


# ── Helper: Logistic Regression staged log-loss ───────────────────────────────
def _lr_loss_curve(
    X_train, X_val, y_train, y_val,
    n_points: int = 50
) -> Tuple[List[float], List[float]]:
    """
    Approximate an LR loss curve by training on progressively larger subsets
    of X_train (i.e. a learning curve framing rather than per-epoch).
    This reveals how log-loss evolves as training data increases.
    """
    sizes = np.linspace(0.10, 1.0, n_points)
    scaler  = RobustScaler().fit(X_train)
    Xs      = scaler.transform(X_train)
    Xv      = scaler.transform(X_val)

    train_losses, val_losses = [], []
    for frac in sizes:
        n   = max(int(len(y_train) * frac), 5)
        idx = np.random.RandomState(RANDOM_STATE).choice(len(y_train), n, replace=False)
        lr  = LogisticRegression(
            solver="lbfgs", C=0.1, class_weight="balanced",
            max_iter=2000, random_state=RANDOM_STATE
        )
        lr.fit(Xs[idx], y_train[idx])

        t_prob = lr.predict_proba(Xs[idx])
        v_prob = lr.predict_proba(Xv)

        # Guard against single-class subsets at very small sizes
        if len(np.unique(y_train[idx])) < 2:
            train_losses.append(np.nan)
            val_losses.append(np.nan)
        else:
            train_losses.append(log_loss(y_train[idx], t_prob))
            val_losses.append(log_loss(y_val, v_prob))

    return train_losses, val_losses


# ── Helper: Random Forest staged log-loss ────────────────────────────────────
def _rf_staged_loss(
    rf: RandomForestClassifier,
    X_train, X_val, y_train, y_val
) -> Tuple[List[float], List[float]]:
    """
    Iterate over staged predictions (1..n_estimators) to produce a
    per-tree-count log-loss curve. Sampled at 50 points for efficiency.
    """
    n      = rf.n_estimators
    steps  = np.linspace(0, n - 1, 50, dtype=int)

    all_train_proba = np.array([e.predict_proba(X_train) for e in rf.estimators_])
    all_val_proba   = np.array([e.predict_proba(X_val)   for e in rf.estimators_])

    train_losses, val_losses = [], []
    for step in steps:
        tp = all_train_proba[:step + 1].mean(axis=0)
        vp = all_val_proba[:step + 1].mean(axis=0)
        train_losses.append(log_loss(y_train, tp))
        val_losses.append(log_loss(y_val, vp))

    return train_losses, val_losses


# ============================================================
# SECTION 7 — CROSS-VALIDATION
# ============================================================

def cross_validate_all(
    X: np.ndarray,
    y: np.ndarray,
    y_train: np.ndarray,   # needed to compute scale_pos_weight for XGB
) -> pd.DataFrame:
    """
    Stratified K-fold cross-validation on the FULL (train+val+test) dataset.
    Returns a tidy DataFrame of per-fold scores for all three models.

    Metrics evaluated
    -----------------
    accuracy   : overall classification accuracy
    f1_macro   : macro-averaged F1 across both classes
    roc_auc    : area under the ROC curve
    neg_log_loss: negative log-loss (higher = better for sklearn CV)
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy":     "accuracy",
        "f1_macro":     "f1_macro",
        "roc_auc":      "roc_auc",
        "neg_log_loss": "neg_log_loss",
    }

    models_for_cv = {
        "Logistic Regression": build_logistic_regression(y_train),
        "Random Forest":       build_random_forest(y_train),
        "XGBoost":             XGBClassifier(   # no early stopping in CV
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=_class_weight_ratio(y_train),
            eval_metric="logloss", use_label_encoder=False,
            random_state=RANDOM_STATE, verbosity=0,
        ),
    }

    rows = []
    for name, model in models_for_cv.items():
        print(f"[cv] {name} ({CV_FOLDS}-fold) ...")
        cv_res = cross_validate(
            model, X, y, cv=skf, scoring=scoring, return_train_score=True
        )
        for fold in range(CV_FOLDS):
            rows.append({
                "model":          name,
                "fold":           fold + 1,
                "accuracy":       cv_res["test_accuracy"][fold],
                "f1_macro":       cv_res["test_f1_macro"][fold],
                "roc_auc":        cv_res["test_roc_auc"][fold],
                "log_loss":      -cv_res["test_neg_log_loss"][fold],
                "train_accuracy": cv_res["train_accuracy"][fold],
                "train_f1_macro": cv_res["train_f1_macro"][fold],
            })

    df = pd.DataFrame(rows)
    return df


# ============================================================
# SECTION 8 — EVALUATION METRICS
# ============================================================

def compute_test_metrics(
    results: Dict,
    y_test:  np.ndarray,
) -> pd.DataFrame:
    """
    Compute all test-set metrics for each model and return a summary DataFrame.

    Metrics
    -------
    accuracy       : correct / total predictions
    f1_macro       : unweighted mean of per-class F1 (fair for imbalanced data)
    f1_weighted    : class-size-weighted F1
    roc_auc        : area under ROC (primary ranking metric from proposal)
    log_loss       : cross-entropy loss on test set
    precision_macro: mean precision across classes
    recall_macro   : mean recall across classes
    """
    rows = []
    for key, res in results.items():
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]

        rows.append({
            "Model":           res["label"],
            "Accuracy":        accuracy_score(y_test, y_pred),
            "F1 (macro)":      f1_score(y_test, y_pred, average="macro",    zero_division=0),
            "F1 (weighted)":   f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "ROC-AUC":         roc_auc_score(y_test, y_prob),
            "Log Loss":        log_loss(y_test, y_prob),
            "Precision (mac)": _precision(y_test, y_pred),
            "Recall (mac)":    _recall(y_test, y_pred),
        })
        print(f"\n[eval] {res['label']}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Risky (0)", "Healthy (1)"],
                                    zero_division=0))

    return pd.DataFrame(rows).set_index("Model")


def _precision(y_true, y_pred) -> float:
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average="macro", zero_division=0)

def _recall(y_true, y_pred) -> float:
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


# ============================================================
# SECTION 9 — VISUALISATION
# ============================================================

def plot_roc_curves(
    results: Dict,
    y_test:  np.ndarray,
    save_path: Path,
) -> None:
    """
    Overlay ROC curves for all three models on one axes.
    Each curve is annotated with its AUC value.
    The diagonal chance line is drawn for reference.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for key, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc_val = roc_auc_score(y_test, res["y_prob"])
        ax.plot(fpr, tpr,
                label=f"{res['label']}  (AUC = {auc_val:.4f})",
                color=PALETTE[res["label"]], linewidth=2.2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.2,
            label="Random chance (AUC = 0.50)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="#94A3B8")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Tabular Baseline Models\n"
                 "(no graph topology used)", pad=12)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True, alpha=0.5)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] ROC curves → {save_path}")


def plot_confusion_matrices(
    results: Dict,
    y_test:  np.ndarray,
    save_path: Path,
) -> None:
    """
    Side-by-side normalised confusion matrices for each model.
    Values show per-class recall (row-normalised).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    class_names = ["Risky\n(0)", "Healthy\n(1)"]

    for ax, (key, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"], normalize="true")
        sns.heatmap(
            cm, annot=True, fmt=".2f", ax=ax,
            cmap="Blues", linewidths=0.5, linecolor="#E2E8F0",
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 13},
            vmin=0, vmax=1,
        )
        ax.set_title(res["label"], pad=10,
                     color=PALETTE[res["label"]])
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

    fig.suptitle("Confusion Matrices (row-normalised recall)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Confusion matrices → {save_path}")


def plot_loss_curves(
    results: Dict,
    save_path: Path,
) -> None:
    """
    Per-model training vs. validation log-loss curves.

    Logistic Regression : log-loss vs. training-set size (learning curve view)
    Random Forest       : log-loss vs. number of trees in ensemble
    XGBoost             : log-loss vs. boosting round (exact, per-round tracking)

    The vertical dashed line on XGBoost marks the early-stopping round.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    titles = {
        "lr":  "Logistic Regression\n(log-loss vs. training size)",
        "rf":  "Random Forest\n(log-loss vs. # trees)",
        "xgb": "XGBoost\n(log-loss vs. boosting round)",
    }
    xlabels = {
        "lr":  "Training set fraction",
        "rf":  "Number of trees",
        "xgb": "Boosting round",
    }

    for ax, (key, res) in zip(axes, results.items()):
        tl = res["train_losses"]
        vl = res["val_losses"]

        if key == "lr":
            x = np.linspace(0.10, 1.0, len(tl))
        elif key == "rf":
            x = np.linspace(1, 500, len(tl))
        else:
            x = np.arange(1, len(tl) + 1)

        color = PALETTE[res["label"]]
        ax.plot(x, tl, label="Train", color=color, linewidth=2)
        ax.plot(x, vl, label="Validation", color=color,
                linewidth=2, linestyle="--", alpha=0.75)

        # Mark early stopping for XGBoost
        if key == "xgb" and "best_round" in res:
            br = res["best_round"]
            ax.axvline(x=br, color="#F59E0B", linewidth=1.5, linestyle=":",
                       label=f"Best round: {br}")

        ax.set_title(titles[key], pad=9)
        ax.set_xlabel(xlabels[key])
        ax.set_ylabel("Log-Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.suptitle("Loss Curves — Tabular Baseline Models",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Loss curves → {save_path}")


def plot_feature_importance(
    results: Dict,
    feature_names: List[str],
    save_path: Path,
    top_n: int = 15,
) -> None:
    """
    Feature importance visualisation.

    Logistic Regression : absolute coefficient magnitude (scaled features)
    Random Forest       : mean decrease in impurity (Gini importance)
    XGBoost             : weight-based feature importance (gain)

    The x-axis is normalised to [0, 1] per model.
    A red dashed vertical line separates the original features from the
    engineered and one-hot-encoded features.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    n_feats = len(feature_names)

    def get_importance(key, res) -> np.ndarray:
        model = res["model"]
        if key == "lr":
            clf   = model.named_steps["clf"]
            coefs = np.abs(clf.coef_[0])
            return coefs / coefs.max()
        elif key == "rf":
            imp = model.feature_importances_
            return imp / imp.max()
        else:  # xgb
            imp = model.feature_importances_
            return imp / imp.max()

    for ax, (key, res) in zip(axes, results.items()):
        imp    = get_importance(key, res)
        n_show = min(top_n, n_feats)
        idx    = np.argsort(imp)[-n_show:]

        colors = [PALETTE[res["label"]] if feature_names[i] in TABULAR_FEATURES
                  else "#7C3AED"          # purple for engineered / dummies
                  for i in idx]

        ax.barh(
            [feature_names[i] for i in idx],
            imp[idx],
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_title(res["label"], color=PALETTE[res["label"]], pad=9)
        ax.set_xlabel("Normalised importance")
        ax.grid(axis="x", alpha=0.4)
        ax.set_xlim(0, 1.08)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PALETTE[res["label"]], label="Original tabular"),
            Patch(facecolor="#7C3AED",             label="Engineered / dummy"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5)

    fig.suptitle("Feature Importances — Non-Topological Baseline",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Feature importance → {save_path}")


def plot_cv_summary(
    cv_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Box plots showing distribution of CV metrics across folds.
    One column per metric, one box per model.
    """
    metrics = ["accuracy", "f1_macro", "roc_auc", "log_loss"]
    labels  = ["Accuracy", "F1 (macro)", "ROC-AUC", "Log-Loss"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    model_order = ["Logistic Regression", "Random Forest", "XGBoost"]
    colors      = [PALETTE[m] for m in model_order]

    for ax, metric, label in zip(axes, metrics, labels):
        data   = [cv_df[cv_df["model"] == m][metric].values for m in model_order]
        bp     = ax.boxplot(data, patch_artist=True, widths=0.45,
                            medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticklabels(["LR", "RF", "XGB"], fontsize=10)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.4)

        # Annotate median
        for i, d in enumerate(data):
            ax.text(i + 1, np.median(d) + 0.002, f"{np.median(d):.3f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle(f"{CV_FOLDS}-Fold Cross-Validation Summary",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] CV summary → {save_path}")


# ============================================================
# SECTION 10 — RESULTS EXPORT
# ============================================================

def print_final_table(test_metrics: pd.DataFrame, cv_df: pd.DataFrame) -> None:
    """Print a clean summary table to stdout."""
    sep = "=" * 78
    print(f"\n{sep}")
    print("  PensionLens — Tabular Baseline Results (Test Set)")
    print(f"{sep}")
    print(test_metrics.round(4).to_string())
    print(f"\n{sep}")
    print(f"  {CV_FOLDS}-Fold CV Summary (mean ± std)")
    print(f"{sep}")
    cv_summary = (
        cv_df.groupby("model")[["accuracy", "f1_macro", "roc_auc", "log_loss"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(cv_summary.to_string())
    print(f"{sep}\n")


def save_results(
    test_metrics: pd.DataFrame,
    cv_df:        pd.DataFrame,
    output_dir:   Path,
) -> None:
    """Save test metrics and CV scores to CSV."""
    test_metrics.reset_index().to_csv(
        output_dir / "baseline_results_summary.csv", index=False
    )
    cv_df.to_csv(output_dir / "baseline_cv_scores.csv", index=False)
    print(f"[save] Results → {output_dir / 'baseline_results_summary.csv'}")
    print(f"[save] CV scores → {output_dir / 'baseline_cv_scores.csv'}")


# ============================================================
# SECTION 11 — MAIN ENTRY POINT
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PensionLens Non-Topological Baseline Classifier"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing pensionlens_labeled.csv "
             "(default: current directory)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_outputs",
        help="Directory to write results CSVs and plots (default: ./baseline_outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args       = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("  PensionLens — Tabular Baseline Training Pipeline")
    print("  Strictly non-topological: no graph structure, no GDS features")
    print("=" * 78 + "\n")

    # ── 1. Load & validate ────────────────────────────────────────────────────
    df = load_labeled_data(data_dir)

    # ── 2. Build feature matrix ───────────────────────────────────────────────
    X, y, feature_names = build_feature_matrix(df)

    # ── 3. Stratified split ───────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    # ── 4. Cross-validation (on full dataset for unbiased estimates) ──────────
    print("\n[cv] Running stratified cross-validation ...")
    cv_df = cross_validate_all(X, y, y_train)

    # ── 5. Train final models ─────────────────────────────────────────────────
    print("\n[train] Training final models on train split ...")
    results = train_all_models(
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    )

    # ── 6. Evaluate on held-out test set ─────────────────────────────────────
    print("\n[eval] Test-set evaluation ...")
    test_metrics = compute_test_metrics(results, y_test)

    # ── 7. Print summary ──────────────────────────────────────────────────────
    print_final_table(test_metrics, cv_df)

    # ── 8. Generate all plots ─────────────────────────────────────────────────
    print("[plots] Generating visualisations ...")
    plot_roc_curves(results, y_test,
                    plots_dir / "roc_curves.png")
    plot_confusion_matrices(results, y_test,
                             plots_dir / "confusion_matrices.png")
    plot_loss_curves(results,
                     plots_dir / "loss_curves.png")
    plot_feature_importance(results, feature_names,
                             plots_dir / "feature_importance.png")
    plot_cv_summary(cv_df,
                    plots_dir / "cv_summary.png")

    # ── 9. Save CSV results ───────────────────────────────────────────────────
    save_results(test_metrics, cv_df, output_dir)

    print("\n[done] Baseline pipeline complete.")
    print(f"       Results: {output_dir}")
    print(f"       Plots:   {plots_dir}\n")


if __name__ == "__main__":
    main()
