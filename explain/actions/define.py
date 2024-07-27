"""Function to define feature meanings."""
import gin


@gin.configurable
def define_operation(conversation, parse_text, i, **kwargs):
    """Generates text to define feature."""
    feature_name = parse_text[i+1]
    feature_definition = conversation.get_feature_definition(feature_name)
    if feature_definition is None:
        return f"Definition for feature name {feature_name} is not specified.", 1
    return_string = f"The feature named {feature_name} is defined as: "
    return_string += "" + feature_definition + "."
    return return_string, 1
