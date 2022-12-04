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
    result = precision_score(y_true, y_pred, pos_label=pos_label, average='binary')
    
    return result


def recall_scorer(y_true, y_pred, pos_label=1):
    result = recall_score(y_true, y_pred, pos_label=pos_label, average='binary')
    
    return result


def f1_scorer(y_true, y_pred, pos_label=1):
    result = f1_score(y_true, y_pred, pos_label=pos_label, average='binary')
    
    return result


def roc_auc_scorer(y_true, y_pred):
    result = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
    
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
    scorer = make_scorer(precision_scorer, pos_label=pos_label,
                         greater_is_better=True)

    return scorer


def get_recall_scorer(pos_label=1):
    scorer = make_scorer(recall_scorer, pos_label=pos_label,
                         greater_is_better=True)

    return scorer


def get_f1_scorer(pos_label=1):
    scorer = make_scorer(f1_scorer, pos_label=pos_label,
                         greater_is_better=True)

    return scorer


def get_roc_auc_scorer(pos_label=1):
    scorer = make_scorer(roc_auc_scorer, greater_is_better=True)

    return scorer


def get_anomaly_scoring():
    scoring = {'balanced_accuracy': 'balanced_accuracy',
                   'roc_auc': 'roc_auc',
                   'f1_pos': get_f1_scorer(pos_label=1),
                   'f1_neg': get_f1_scorer(pos_label=-1),
                   'recall_pos': get_recall_scorer(pos_label=1),
                   'recall_neg': get_recall_scorer(pos_label=-1),
                   'precision_pos': get_precision_scorer(pos_label=1),
                   'precision_neg': get_precision_scorer(pos_label=-1),
                   'roc_auc_ovr': get_roc_auc_scorer(),
                   'tn': get_tn_scorer(),
                   'fp': get_fp_scorer(),
                   'fn': get_fn_scorer(),
                   'tp': get_tp_scorer(),
              }
    
    return scoring
