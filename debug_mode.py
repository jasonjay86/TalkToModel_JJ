"""Tests performed booting up ExplainBot."""
from os import mkdir  # noqa: E402, F401
from os.path import dirname, abspath, join  # noqa: E402, F401
import sys

import gin
import numpy as np
# parent = dirname(dirname(abspath(__file__)))
# sys.path.append(parent)

from explain.logic import ExplainBot  # noqa: E402, F401
from explain.conversation import fork_conversation  # noqa: E402, 
from explain.sample_prompts_by_action import sample_prompt_for_action

def get_bot_response(BOT, user_text,action):
    """Load the box response."""
    prompt = sample_prompt_for_action(action,
                                      BOT.prompts.filename_to_prompt_id,
                                      BOT.prompts.final_prompt_set,
                                      real_ids=BOT.conversation.get_training_data_ids())
    print(prompt)
    try:
        # data = json.loads(request.data)
        # user_text = data["userInput"]
        conversation = BOT.conversation
        # print("retrieved convo")
        response = BOT.update_state(user_text, conversation)
        # print("got response")
    except Exception as ext:
        # print(f"Traceback getting bot response: {traceback.format_exc()}")
        print(f"Exception getting bot response: {ext}")
        response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
    return response

@gin.configurable
class GlobalArgs:
    def __init__(self, config, baseurl):
        self.config = config
        self.baseurl = baseurl


# Parse gin global config
gin.parse_config_file("global_config.gin")

# Get args
args = GlobalArgs()

# Parse application level configs
gin.parse_config_file(args.config)

# Load the explainbot
bot = ExplainBot()
objective = bot.conversation.describe.get_dataset_objective()

print("What records does the model predict incorrectly?")
print(get_bot_response(bot,"What records does the model predict incorrectly?","mistake"))

print("What are the most prominent features?")
print(get_bot_response(bot,"What are the most prominent features?","important"))


