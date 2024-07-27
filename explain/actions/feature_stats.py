"""Shows the feature statistics"""
from copy import deepcopy

import gin
import numpy as np

from explain.actions.utils import get_parse_filter_text


def compute_stats(df, labels, f, conversation):
    """Computes the feature stats"""
    if f == "target":
        labels = deepcopy(labels).to_numpy()
        stats = "<ul>"
        for label in conversation.class_names:
            freq = np.count_nonzero(label == labels) / len(labels)
            r_freq = round(freq*100, conversation.rounding_precision)
            name = conversation.get_class_name_from_label(label)
            stats += f"<li>{name}: {r_freq}%</li>"
        stats += "</ul>"
    else:
        feature = df[f]
        mean = round(feature.mean(), conversation.rounding_precision)
        std = round(feature.std(), conversation.rounding_precision)
        min_v = round(feature.min(), conversation.rounding_precision)
        max_v = round(feature.max(), conversation.rounding_precision)
        stats = (f"mean: {mean}\none std: {std}\n"
                 f"min: {min_v}\nmax: {max_v}")
    return stats


@gin.configurable
def feature_stats(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows the feature stats."""
    data = conversation.temp_dataset.contents['X']
    label = conversation.temp_dataset.contents['y']
    intro_text = get_parse_filter_text(conversation)
    feature_name = parse_text[i+1]

    if len(data) == 1:
        value = data[feature_name].item()
        return_text = f"{intro_text} the value of {feature_name} is {value}"
        return_text += "\n\n"
        return return_text, 1

    # Compute feature statistics
    stats = compute_stats(data, label, feature_name, conversation)

    # Concat filtering text description and statistics
    if feature_name == "target":
        feature_name = "the labels"
    return_text = f"{intro_text} the statistics of {feature_name} in the dataset are:\n"
    return_text += stats
    return_text += "\n\n"
    return return_text, 1
