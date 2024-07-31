"""Show model mistakes"""
from copy import deepcopy

import gin
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from explain.actions.utils import get_parse_filter_text, get_rules


def one_mistake(y_true, y_pred, conversation, intro_text):
    """One mistake text"""
    label = y_true[0]
    prediction = y_pred[0]

    label_text = conversation.get_class_name_from_label(label)
    predict_text = conversation.get_class_name_from_label(prediction)

    if label == prediction:
        correct_text = "correct"
    else:
        correct_text = "incorrect"

    return_string = (f"{intro_text} the model predicts {predict_text} and the ground"
                     f" label is {label_text}, so the model is {correct_text}!")
    return return_string


def sample_mistakes(y_true, y_pred, conversation, intro_text, ids):
    """Sample mistakes sub-operation"""
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        incorrect_num = np.sum(y_true != y_pred)
        total_num = len(y_true)
        incorrect_data = ids[y_true != y_pred]

        error_rate = round(incorrect_num / total_num, conversation.rounding_precision)
        return_string = (f"{intro_text} the model is incorrect {incorrect_num} out of {total_num} "
                         f"times (error rate {error_rate}). Here are the ids of instances the model"
                         f" predicts incorrectly:\n\n{incorrect_data}")

    return return_string


def train_tree(data, target, depth: int = 1):
    """Trains a decision tree"""
    dt_string = []
    tries = 0
    # print("about to train the tree")
    while len(dt_string) < 3 and tries < 10:
        tries += 1
        dt = DecisionTreeClassifier(max_depth=depth).fit(data, target)
        dt_string = get_rules(dt,
                              feature_names=list(data.columns),
                              class_names=["correct", "incorrect"])
        depth += 1

    return dt_string


def typical_mistakes(data, y_true, y_pred, conversation, intro_text, ids):
    """Typical mistakes sub-operation"""
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        incorrect_vals = y_true != y_pred
        return_options = train_tree(data, incorrect_vals)
        # print("Trained the tree")
        if len(return_options) == 0:
            return "I couldn't find any patterns for mistakes the model typically makes."

        return_string = f"{intro_text} the model typically predicts incorrect:\n\n"
        for rule in return_options:
            return_string += rule + "\n\n"

    return return_string


@gin.configurable
def show_mistakes_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows the model mistakes."""
    data = conversation.temp_dataset.contents['X']
    y_true_pd = deepcopy(conversation.temp_dataset.contents['y'])

    if isinstance(y_true_pd, pd.Series):
        y_true = y_true_pd.to_numpy()
    elif isinstance(y_true_pd, list):
        y_true = np.array(y_true_pd)

    # Get ids
    ids = np.array(list(data.index))

    model = conversation.get_var('model').contents

    # The filtering text
    intro_text = get_parse_filter_text(conversation)
    # print("Got intro text")
    if len(y_true) == 0:
        return "There are no instances in the data that meet this description.\n\n", 0

    # print("about to make prediction")
    y_pred = model.predict(data)
    # print("made prediction")
    if np.sum(y_true == y_pred) == len(y_true):
        if len(y_true) == 1:
            return f"{intro_text} the model predicts correctly!\n\n", 1
        else:
            return f"{intro_text} the model predicts correctly on all the instances in the data!\n\n", 1


    if parse_text[i+1] == "sample":
        return_string = sample_mistakes(y_true,
                                        y_pred,
                                        conversation,
                                        intro_text,
                                        ids)
    elif parse_text[i+1] == "typical":
        return_string = typical_mistakes(data,
                                         y_true,
                                         y_pred,
                                         conversation,
                                         intro_text,
                                         ids)
    else:
        raise NotImplementedError(f"No mistake type {parse_text[i+1]}")

    return_string += "\n\n"
    return return_string, 1
