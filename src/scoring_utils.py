""".
 
"""
__author__ = "Bernhard Lehner <https://github.com/berni-lehner>"


# ========================================================================
# ANOMALY DETECTION
# ========================================================================
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


def tn_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    return cm[0, 0]


def fp_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    return cm[0, 1]


def fn_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    return cm[1, 0]


def tp_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    return cm[1, 1]


def precision_scorer(y_true, y_pred, pos_label=1):
    result = precision_score(y_true, y_pred, pos_label=pos_label, average="binary")

    return result


def recall_scorer(y_true, y_pred, pos_label=1):
    result = recall_score(y_true, y_pred, pos_label=pos_label, average="binary")

    return result


def f1_scorer(y_true, y_pred, pos_label=1):
    result = f1_score(y_true, y_pred, pos_label=pos_label, average="binary")

    return result


def roc_auc_scorer(y_true, y_pred):
    result = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")

    return result


def get_tn_scorer():
    scorer = make_scorer(tn_scorer, greater_is_better=True)

    return scorer


def get_tp_scorer():
    scorer = make_scorer(tp_scorer, greater_is_better=True)

    return scorer


def get_fn_scorer():
    # avoid negative results with greater_is_better
    scorer = make_scorer(fn_scorer, greater_is_better=True)

    return scorer


def get_fp_scorer():
    # avoid negative results with greater_is_better
    scorer = make_scorer(fp_scorer, greater_is_better=True)

    return scorer


def get_precision_scorer(pos_label=1):
    scorer = make_scorer(precision_scorer, pos_label=pos_label, greater_is_better=True)

    return scorer


def get_recall_scorer(pos_label=1):
    scorer = make_scorer(recall_scorer, pos_label=pos_label, greater_is_better=True)

    return scorer


def get_f1_scorer(pos_label=1):
    scorer = make_scorer(f1_scorer, pos_label=pos_label, greater_is_better=True)

    return scorer


def get_roc_auc_scorer(pos_label=1):
    scorer = make_scorer(roc_auc_scorer, greater_is_better=True)

    return scorer


def get_anomaly_scoring():
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "f1_pos": get_f1_scorer(pos_label=1),
        "f1_neg": get_f1_scorer(pos_label=-1),
        "recall_pos": get_recall_scorer(pos_label=1),
        "recall_neg": get_recall_scorer(pos_label=-1),
        "precision_pos": get_precision_scorer(pos_label=1),
        "precision_neg": get_precision_scorer(pos_label=-1),
        "roc_auc_ovr": get_roc_auc_scorer(),
        "tn": get_tn_scorer(),
        "fp": get_fp_scorer(),
        "fn": get_fn_scorer(),
        "tp": get_tp_scorer(),
    }

    return scoring


# ========================================================================
# REGRESSION
# ========================================================================
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import mean_squared_error


def r2_range_scorer(y_true, y_pred, limits=None, default=2.0):
    y_idx = (y_true >= limits[0]) & (y_true <= limits[1])
    y_true = y_true[y_idx]
    y_pred = y_pred[y_idx]

    if len(y_true) == 0:
        result = default
    else:
        result = r2_score(y_true, y_pred)

    return result


def mse_range_scorer(y_true, y_pred, limits=None, default=-1.0):
    y_idx = (y_true >= limits[0]) & (y_true <= limits[1])
    y_true = y_true[y_idx]
    y_pred = y_pred[y_idx]

    if len(y_true) == 0:
        result = default
    else:
        result = mean_squared_error(y_true, y_pred)

    return result


def r2_scorer(y_true, y_pred, target=None, default=2.0):
    y_idx = y_true == target
    y_true = y_true[y_idx]
    y_pred = y_pred[y_idx]

    if len(y_true) == 0:
        result = default
    else:
        result = r2_score(y_true, y_pred)

    return result


def mse_scorer(y_true, y_pred, target=None, default=-1.0):
    y_idx = y_true == target
    y_true = y_true[y_idx]
    y_pred = y_pred[y_idx]

    if len(y_true) == 0:
        result = default
    else:
        result = mean_squared_error(y_true, y_pred)

    return result


from scipy import stats


def mode_scorer(y_true, y_pred, target=None, default=-1.0):
    y_idx = y_true == target
    y_pred = y_pred[y_idx]

    if len(y_true) == 0:
        result = default
    else:
        result = stats.mode(y_pred.astype(float).round(1)).mode
        if result.shape == tuple():
            result = result.reshape(1)

        result = result[0]

    return result


def mean_scorer(y_true, y_pred, target=None, default=-1.0):
    y_idx = y_true == target
    y_pred = y_pred[y_idx]

    if len(y_true) == 0:
        result = default
    else:
        result = np.mean(y_pred)

    return result


def mode_count_scorer(y_true, y_pred, target=None, default=-1.0):
    y_idx = y_true == target
    y_pred = y_pred[y_idx]

    if len(y_true) == 0:
        result = default
    else:
        result = stats.mode(y_pred.astype(float).round(1)).count
        if result.shape == tuple():
            result = result.reshape(1)

        result = result[0]

    return result


def get_r2_scorer(target=None):
    scorer = make_scorer(r2_scorer, target=target, greater_is_better=True)

    return scorer


def get_mse_scorer(target=None):
    scorer = make_scorer(mse_scorer, target=target, greater_is_better=True)

    return scorer


def get_mse_range_scorer(limits=None):
    scorer = make_scorer(mse_range_scorer, limits=limits, greater_is_better=True)

    return scorer


def get_mode_scorer(target=None):
    scorer = make_scorer(mode_scorer, target=target, greater_is_better=True)

    return scorer


def get_mode_count_scorer(target=None):
    scorer = make_scorer(mode_count_scorer, target=target, greater_is_better=True)

    return scorer


def get_pred_mean_scorer(target=None):
    scorer = make_scorer(mean_scorer, target=target, greater_is_better=True)

    return scorer


def get_real_regression_scoring():
    scoring = {
        "r2 avg": "r2",
        "neg_mean_squared_error avg": "neg_mean_squared_error",
        "r2_0.0": get_r2_scorer(target=0.0),
        "r2_1.5": get_r2_scorer(target=1.5),
        "r2_2.5": get_r2_scorer(target=2.5),
        "r2_3.5": get_r2_scorer(target=3.5),
        "r2_4.5": get_r2_scorer(target=4.5),
        "r2_5.5": get_r2_scorer(target=5.5),
        "r2_7.0": get_r2_scorer(target=7.0),
        "r2_8.0": get_r2_scorer(target=8.0),
        "r2_10.0": get_r2_scorer(target=10.0),
        "r2_12.0": get_r2_scorer(target=12.0),
        "r2_14.0": get_r2_scorer(target=14.0),
        "r2_16.0": get_r2_scorer(target=16.0),
        "r2_17.0": get_r2_scorer(target=17.0),
        "r2_19.0": get_r2_scorer(target=19.0),
        "r2_21.0": get_r2_scorer(target=21.0),
        "r2_23.0": get_r2_scorer(target=23.0),
        "r2_25.0": get_r2_scorer(target=25.0),
        "r2_27.0": get_r2_scorer(target=27.0),
        "r2_29.0": get_r2_scorer(target=29.0),
        "r2_31.0": get_r2_scorer(target=31.0),
        "r2_33.0": get_r2_scorer(target=33.0),
        "r2_35.0": get_r2_scorer(target=35.0),
        "r2_37.0": get_r2_scorer(target=37.0),
        "r2_39.0": get_r2_scorer(target=39.0),
        "r2_40.0": get_r2_scorer(target=40.0),
        "mse_0.0": get_mse_scorer(target=0.0),
        "mse_1.5": get_mse_scorer(target=1.5),
        "mse_2.5": get_mse_scorer(target=2.5),
        "mse_3.5": get_mse_scorer(target=3.5),
        "mse_4.5": get_mse_scorer(target=4.5),
        "mse_5.5": get_mse_scorer(target=5.5),
        "mse_7.0": get_mse_scorer(target=7.0),
        "mse_8.0": get_mse_scorer(target=8.0),
        "mse_10.0": get_mse_scorer(target=10.0),
        "mse_12.0": get_mse_scorer(target=12.0),
        "mse_14.0": get_mse_scorer(target=14.0),
        "mse_16.0": get_mse_scorer(target=16.0),
        "mse_17.0": get_mse_scorer(target=17.0),
        "mse_19.0": get_mse_scorer(target=19.0),
        "mse_21.0": get_mse_scorer(target=21.0),
        "mse_23.0": get_mse_scorer(target=23.0),
        "mse_25.0": get_mse_scorer(target=25.0),
        "mse_27.0": get_mse_scorer(target=27.0),
        "mse_29.0": get_mse_scorer(target=29.0),
        "mse_31.0": get_mse_scorer(target=31.0),
        "mse_33.0": get_mse_scorer(target=33.0),
        "mse_35.0": get_mse_scorer(target=35.0),
        "mse_37.0": get_mse_scorer(target=37.0),
        "mse_39.0": get_mse_scorer(target=39.0),
        "mse_40.0": get_mse_scorer(target=40.0),
        "mse_0.0_40.0": get_mse_range_scorer(limits=[0.0, 40]),
        "mse_1.5_40.0": get_mse_range_scorer(limits=[1.5, 40]),
        "mse_2.5_40.0": get_mse_range_scorer(limits=[2.5, 40]),
        "mse_3.5_40.0": get_mse_range_scorer(limits=[3.5, 40]),
        "mse_5.0_40.0": get_mse_range_scorer(limits=[5.0, 40]),
        "mode_0.0": get_mode_scorer(target=0.0),
        "mode_1.5": get_mode_scorer(target=1.5),
        "mode_2.5": get_mode_scorer(target=2.5),
        "mode_3.5": get_mode_scorer(target=3.5),
        "mode_4.5": get_mode_scorer(target=4.5),
        "mode_5.5": get_mode_scorer(target=5.5),
        "mode_7.0": get_mode_scorer(target=7.0),
        "mode_8.0": get_mode_scorer(target=8.0),
        "mode_10.0": get_mode_scorer(target=10.0),
        "mode_12.0": get_mode_scorer(target=12.0),
        "mode_14.0": get_mode_scorer(target=14.0),
        "mode_16.0": get_mode_scorer(target=16.0),
        "mode_17.0": get_mode_scorer(target=17.0),
        "mode_19.0": get_mode_scorer(target=19.0),
        "mode_21.0": get_mode_scorer(target=21.0),
        "mode_23.0": get_mode_scorer(target=23.0),
        "mode_25.0": get_mode_scorer(target=25.0),
        "mode_27.0": get_mode_scorer(target=27.0),
        "mode_29.0": get_mode_scorer(target=29.0),
        "mode_31.0": get_mode_scorer(target=31.0),
        "mode_33.0": get_mode_scorer(target=33.0),
        "mode_35.0": get_mode_scorer(target=35.0),
        "mode_37.0": get_mode_scorer(target=37.0),
        "mode_39.0": get_mode_scorer(target=39.0),
        "mode_40.0": get_mode_scorer(target=40.0),
        "mode_count_0.0": get_mode_count_scorer(target=0.0),
        "mode_count_1.5": get_mode_count_scorer(target=1.5),
        "mode_count_2.5": get_mode_count_scorer(target=2.5),
        "mode_count_3.5": get_mode_count_scorer(target=3.5),
        "mode_count_4.5": get_mode_count_scorer(target=4.5),
        "mode_count_5.5": get_mode_count_scorer(target=5.5),
        "mode_count_7.0": get_mode_count_scorer(target=7.0),
        "mode_count_8.0": get_mode_count_scorer(target=8.0),
        "mode_count_10.0": get_mode_count_scorer(target=10.0),
        "mode_count_12.0": get_mode_count_scorer(target=12.0),
        "mode_count_14.0": get_mode_count_scorer(target=14.0),
        "mode_count_16.0": get_mode_count_scorer(target=16.0),
        "mode_count_17.0": get_mode_count_scorer(target=17.0),
        "mode_count_19.0": get_mode_count_scorer(target=19.0),
        "mode_count_21.0": get_mode_count_scorer(target=21.0),
        "mode_count_23.0": get_mode_count_scorer(target=23.0),
        "mode_count_25.0": get_mode_count_scorer(target=25.0),
        "mode_count_27.0": get_mode_count_scorer(target=27.0),
        "mode_count_29.0": get_mode_count_scorer(target=29.0),
        "mode_count_31.0": get_mode_count_scorer(target=31.0),
        "mode_count_33.0": get_mode_count_scorer(target=33.0),
        "mode_count_35.0": get_mode_count_scorer(target=35.0),
        "mode_count_37.0": get_mode_count_scorer(target=37.0),
        "mode_count_39.0": get_mode_count_scorer(target=39.0),
        "mode_count_40.0": get_mode_count_scorer(target=40.0),
        "pred_mean_0.0": get_pred_mean_scorer(target=0.0),
        "pred_mean_1.5": get_pred_mean_scorer(target=1.5),
        "pred_mean_2.5": get_pred_mean_scorer(target=2.5),
        "pred_mean_3.5": get_pred_mean_scorer(target=3.5),
        "pred_mean_4.5": get_pred_mean_scorer(target=4.5),
        "pred_mean_5.5": get_pred_mean_scorer(target=5.5),
        "pred_mean_7.0": get_pred_mean_scorer(target=7.0),
        "pred_mean_8.0": get_pred_mean_scorer(target=8.0),
        "pred_mean_10.0": get_pred_mean_scorer(target=10.0),
        "pred_mean_12.0": get_pred_mean_scorer(target=12.0),
        "pred_mean_14.0": get_pred_mean_scorer(target=14.0),
        "pred_mean_16.0": get_pred_mean_scorer(target=16.0),
        "pred_mean_17.0": get_pred_mean_scorer(target=17.0),
        "pred_mean_19.0": get_pred_mean_scorer(target=19.0),
        "pred_mean_21.0": get_pred_mean_scorer(target=21.0),
        "pred_mean_23.0": get_pred_mean_scorer(target=23.0),
        "pred_mean_25.0": get_pred_mean_scorer(target=25.0),
        "pred_mean_27.0": get_pred_mean_scorer(target=27.0),
        "pred_mean_29.0": get_pred_mean_scorer(target=29.0),
        "pred_mean_31.0": get_pred_mean_scorer(target=31.0),
        "pred_mean_33.0": get_pred_mean_scorer(target=33.0),
        "pred_mean_35.0": get_pred_mean_scorer(target=35.0),
        "pred_mean_37.0": get_pred_mean_scorer(target=37.0),
        "pred_mean_39.0": get_pred_mean_scorer(target=39.0),
        "pred_mean_40.0": get_pred_mean_scorer(target=40.0),
    }

    return scoring


def get_synth_regression_scoring():
    scoring = {
        "r2 avg": "r2",
        "neg_mean_squared_error avg": "neg_mean_squared_error",
        "r2_0.0": get_r2_scorer(target=0.0),
        "r2_2.8": get_r2_scorer(target=2.8),
        "r2_3.1": get_r2_scorer(target=3.1),
        "r2_3.4": get_r2_scorer(target=3.4),
        "r2_3.8": get_r2_scorer(target=3.8),
        "r2_4.1": get_r2_scorer(target=4.1),
        "r2_4.4": get_r2_scorer(target=4.4),
        "r2_4.7": get_r2_scorer(target=4.7),
        "r2_5.0": get_r2_scorer(target=5.0),
        "r2_6.7": get_r2_scorer(target=6.7),
        "r2_8.3": get_r2_scorer(target=8.3),
        "r2_10.0": get_r2_scorer(target=10.0),
        "r2_12.0": get_r2_scorer(target=12.0),
        "r2_14.0": get_r2_scorer(target=14.0),
        "r2_16.0": get_r2_scorer(target=16.0),
        "r2_18.0": get_r2_scorer(target=18.0),
        "r2_20.0": get_r2_scorer(target=20.0),
        "r2_22.0": get_r2_scorer(target=22.0),
        "r2_24.0": get_r2_scorer(target=24.0),
        "r2_26.0": get_r2_scorer(target=26.0),
        "r2_28.0": get_r2_scorer(target=28.0),
        "r2_30.0": get_r2_scorer(target=30.0),
        "r2_32.0": get_r2_scorer(target=32.0),
        "r2_34.0": get_r2_scorer(target=34.0),
        "r2_36.0": get_r2_scorer(target=36.0),
        "r2_38.0": get_r2_scorer(target=38.0),
        "r2_40.0": get_r2_scorer(target=40.0),
        "mse_0.0": get_mse_scorer(target=0.0),
        "mse_2.2": get_mse_scorer(target=2.2),
        "mse_2.5": get_mse_scorer(target=2.5),
        "mse_2.8": get_mse_scorer(target=2.8),
        "mse_3.1": get_mse_scorer(target=3.1),
        "mse_3.4": get_mse_scorer(target=3.4),
        "mse_3.8": get_mse_scorer(target=3.8),
        "mse_4.1": get_mse_scorer(target=4.1),
        "mse_4.4": get_mse_scorer(target=4.4),
        "mse_4.7": get_mse_scorer(target=4.7),
        "mse_5.0": get_mse_scorer(target=5.0),
        "mse_6.7": get_mse_scorer(target=6.7),
        "mse_8.3": get_mse_scorer(target=8.3),
        "mse_10.0": get_mse_scorer(target=10.0),
        "mse_12.0": get_mse_scorer(target=12.0),
        "mse_14.0": get_mse_scorer(target=14.0),
        "mse_16.0": get_mse_scorer(target=16.0),
        "mse_18.0": get_mse_scorer(target=18.0),
        "mse_20.0": get_mse_scorer(target=20.0),
        "mse_22.0": get_mse_scorer(target=22.0),
        "mse_24.0": get_mse_scorer(target=24.0),
        "mse_26.0": get_mse_scorer(target=26.0),
        "mse_28.0": get_mse_scorer(target=28.0),
        "mse_30.0": get_mse_scorer(target=30.0),
        "mse_32.0": get_mse_scorer(target=32.0),
        "mse_34.0": get_mse_scorer(target=34.0),
        "mse_36.0": get_mse_scorer(target=36.0),
        "mse_38.0": get_mse_scorer(target=38.0),
        "mse_40.0": get_mse_scorer(target=40.0),
        "mse_0.0_40.0": get_mse_range_scorer(limits=[0.0, 40]),
        "mse_2.8_40.0": get_mse_range_scorer(limits=[2.8, 40]),
        "mse_5.0_40.0": get_mse_range_scorer(limits=[5.0, 40]),
        "mode_0.0": get_mode_scorer(target=0.0),
        "mode_2.2": get_mode_scorer(target=2.2),
        "mode_2.5": get_mode_scorer(target=2.5),
        "mode_2.8": get_mode_scorer(target=2.8),
        "mode_3.1": get_mode_scorer(target=3.1),
        "mode_3.4": get_mode_scorer(target=3.4),
        "mode_3.8": get_mode_scorer(target=3.8),
        "mode_4.1": get_mode_scorer(target=4.1),
        "mode_4.4": get_mode_scorer(target=4.4),
        "mode_4.7": get_mode_scorer(target=4.7),
        "mode_5.0": get_mode_scorer(target=5.0),
        "mode_6.7": get_mode_scorer(target=6.7),
        "mode_8.3": get_mode_scorer(target=8.3),
        "mode_10.0": get_mode_scorer(target=10.0),
        "mode_12.0": get_mode_scorer(target=12.0),
        "mode_14.0": get_mode_scorer(target=14.0),
        "mode_16.0": get_mode_scorer(target=16.0),
        "mode_18.0": get_mode_scorer(target=18.0),
        "mode_20.0": get_mode_scorer(target=20.0),
        "mode_22.0": get_mode_scorer(target=22.0),
        "mode_24.0": get_mode_scorer(target=24.0),
        "mode_26.0": get_mode_scorer(target=26.0),
        "mode_28.0": get_mode_scorer(target=28.0),
        "mode_30.0": get_mode_scorer(target=30.0),
        "mode_32.0": get_mode_scorer(target=32.0),
        "mode_34.0": get_mode_scorer(target=34.0),
        "mode_36.0": get_mode_scorer(target=36.0),
        "mode_38.0": get_mode_scorer(target=38.0),
        "mode_40.0": get_mode_scorer(target=40.0),
        "mode_count_0.0": get_mode_count_scorer(target=0.0),
        "mode_count_2.2": get_mode_count_scorer(target=2.2),
        "mode_count_2.5": get_mode_count_scorer(target=2.5),
        "mode_count_2.8": get_mode_count_scorer(target=2.8),
        "mode_count_3.1": get_mode_count_scorer(target=3.1),
        "mode_count_3.4": get_mode_count_scorer(target=3.4),
        "mode_count_3.8": get_mode_count_scorer(target=3.8),
        "mode_count_4.1": get_mode_count_scorer(target=4.1),
        "mode_count_4.4": get_mode_count_scorer(target=4.4),
        "mode_count_4.7": get_mode_count_scorer(target=4.7),
        "mode_count_5.0": get_mode_count_scorer(target=5.0),
        "mode_count_6.7": get_mode_count_scorer(target=6.7),
        "mode_count_8.3": get_mode_count_scorer(target=8.3),
        "mode_count_10.0": get_mode_count_scorer(target=10.0),
        "mode_count_12.0": get_mode_count_scorer(target=12.0),
        "mode_count_14.0": get_mode_count_scorer(target=14.0),
        "mode_count_16.0": get_mode_count_scorer(target=16.0),
        "mode_count_18.0": get_mode_count_scorer(target=18.0),
        "mode_count_20.0": get_mode_count_scorer(target=20.0),
        "mode_count_22.0": get_mode_count_scorer(target=22.0),
        "mode_count_24.0": get_mode_count_scorer(target=24.0),
        "mode_count_26.0": get_mode_count_scorer(target=26.0),
        "mode_count_28.0": get_mode_count_scorer(target=28.0),
        "mode_count_30.0": get_mode_count_scorer(target=30.0),
        "mode_count_32.0": get_mode_count_scorer(target=32.0),
        "mode_count_34.0": get_mode_count_scorer(target=34.0),
        "mode_count_36.0": get_mode_count_scorer(target=36.0),
        "mode_count_38.0": get_mode_count_scorer(target=38.0),
        "mode_count_40.0": get_mode_count_scorer(target=40.0),
        "pred_mean_0.0": get_pred_mean_scorer(target=0.0),
        "pred_mean_2.2": get_pred_mean_scorer(target=2.2),
        "pred_mean_2.5": get_pred_mean_scorer(target=2.5),
        "pred_mean_2.8": get_pred_mean_scorer(target=2.8),
        "pred_mean_3.1": get_pred_mean_scorer(target=3.1),
        "pred_mean_3.4": get_pred_mean_scorer(target=3.4),
        "pred_mean_3.8": get_pred_mean_scorer(target=3.8),
        "pred_mean_4.1": get_pred_mean_scorer(target=4.1),
        "pred_mean_4.4": get_pred_mean_scorer(target=4.4),
        "pred_mean_4.7": get_pred_mean_scorer(target=4.7),
        "pred_mean_5.0": get_pred_mean_scorer(target=5.0),
        "pred_mean_6.7": get_pred_mean_scorer(target=6.7),
        "pred_mean_8.3": get_pred_mean_scorer(target=8.3),
        "pred_mean_10.0": get_pred_mean_scorer(target=10.0),
        "pred_mean_12.0": get_pred_mean_scorer(target=12.0),
        "pred_mean_14.0": get_pred_mean_scorer(target=14.0),
        "pred_mean_16.0": get_pred_mean_scorer(target=16.0),
        "pred_mean_18.0": get_pred_mean_scorer(target=18.0),
        "pred_mean_20.0": get_pred_mean_scorer(target=20.0),
        "pred_mean_22.0": get_pred_mean_scorer(target=22.0),
        "pred_mean_24.0": get_pred_mean_scorer(target=24.0),
        "pred_mean_26.0": get_pred_mean_scorer(target=26.0),
        "pred_mean_28.0": get_pred_mean_scorer(target=28.0),
        "pred_mean_30.0": get_pred_mean_scorer(target=30.0),
        "pred_mean_32.0": get_pred_mean_scorer(target=32.0),
        "pred_mean_34.0": get_pred_mean_scorer(target=34.0),
        "pred_mean_36.0": get_pred_mean_scorer(target=36.0),
        "pred_mean_38.0": get_pred_mean_scorer(target=38.0),
        "pred_mean_40.0": get_pred_mean_scorer(target=40.0),
    }

    return scoring


# ========================================================================
# ADDITIONAL REGRESSION METRICS
# ========================================================================
class SHM_Scoring:
    __SYNTH_MSE_RESULTS = [  #'test_mse_0.0',
        "test_mse_2.2",
        "test_mse_2.5",
        "test_mse_2.8",
        "test_mse_3.1",
        "test_mse_3.4",
        "test_mse_3.8",
        "test_mse_4.1",
        "test_mse_4.4",
        "test_mse_4.7",
        "test_mse_5.0",
        "test_mse_6.7",
        "test_mse_8.3",
        "test_mse_10.0",
        "test_mse_12.0",
        "test_mse_14.0",
        "test_mse_16.0",
        "test_mse_18.0",
        "test_mse_20.0",
        "test_mse_22.0",
        "test_mse_24.0",
        "test_mse_26.0",
        "test_mse_28.0",
        "test_mse_30.0",
        "test_mse_32.0",
        "test_mse_34.0",
        "test_mse_36.0",
        "test_mse_38.0",
        "test_mse_40.0",
    ]
    __SYNTH_REL_ERROR = []
    __SYNTH_ABS_ERROR = []

    __REAL_MSE_RESULTS = [  #'test_mse_0.0',
        "test_mse_1.5",
        "test_mse_2.5",
        "test_mse_3.5",
        "test_mse_4.5",
        "test_mse_5.5",
        "test_mse_7.0",
        "test_mse_8.0",
        "test_mse_10.0",
        "test_mse_12.0",
        "test_mse_14.0",
        "test_mse_16.0",
        "test_mse_17.0",
        "test_mse_19.0",
        "test_mse_21.0",
        "test_mse_23.0",
        "test_mse_25.0",
        "test_mse_27.0",
        "test_mse_29.0",
        "test_mse_31.0",
        "test_mse_33.0",
        "test_mse_35.0",
        "test_mse_37.0",
        "test_mse_39.0",
        "test_mse_40.0",
    ]

    __REAL_REL_ERROR = []
    __REAL_ABS_ERROR = []

    def __init__(
        self,
    ):
        # corresponds to SYNTH_MSE_RESULTS
        self.__SYNTH_REL_ERROR = [
            item.replace("test_mse_", "test_rel_err_")
            for item in self.__SYNTH_MSE_RESULTS
        ]
        self.__SYNTH_ABS_ERROR = [
            item.replace("test_mse_", "test_abs_err_")
            for item in self.__SYNTH_MSE_RESULTS
        ]

        # corresponds to REAL_MSE_RESULTS
        self.__REAL_REL_ERROR = [
            item.replace("test_mse_", "test_rel_err_")
            for item in self.__REAL_MSE_RESULTS
        ]
        self.__REAL_ABS_ERROR = [
            item.replace("test_mse_", "test_abs_err_")
            for item in self.__REAL_MSE_RESULTS
        ]

    @property
    def SYNTH_MSE_RESULTS(self) -> list:
        return self.__SYNTH_MSE_RESULTS

    @property
    def SYNTH_REL_ERROR(self) -> list:
        return self.__SYNTH_REL_ERROR

    @property
    def SYNTH_ABS_ERROR(self) -> list:
        return self.__SYNTH_ABS_ERROR

    @property
    def REAL_MSE_RESULTS(self) -> list:
        return self.__REAL_MSE_RESULTS

    @property
    def REAL_REL_ERROR(self) -> list:
        return self.__REAL_REL_ERROR

    @property
    def REAL_ABS_ERROR(self) -> list:
        return self.__REAL_ABS_ERROR

    def add_rel_error(self, df, mse_cols):
        for item in mse_cols:
            size = float(item.replace("test_mse_", ""))
            error = np.sqrt(df[item])

            rel_error = error * 100 / size

            result_col = item.replace("test_mse_", "test_rel_err_")
            df[result_col] = rel_error

        return df

    def add_abs_error(self, df, mse_cols):
        for item in mse_cols:
            error = np.sqrt(df[item])

            result_col = item.replace("test_mse_", "test_abs_err_")
            df[result_col] = error

        return df
