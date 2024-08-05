"""Tests performed booting up ExplainBot."""
from os import mkdir  # noqa: E402, F401
from os.path import dirname, abspath, join  # noqa: E402, F401
import sys
import re

import gin
import numpy as np
# parent = dirname(dirname(abspath(__file__)))
# sys.path.append(parent)

from explain.logic import ExplainBot  # noqa: E402, F401
from explain.conversation import fork_conversation  # noqa: E402, 
from explain.sample_prompts_by_action import sample_prompt_for_action

def get_bot_response(BOT, user_text,action):
    """Load the box response."""
    sample_prompt_for_action(action,
                            BOT.prompts.filename_to_prompt_id,
                            BOT.prompts.final_prompt_set,
                            real_ids=BOT.conversation.get_training_data_ids())
    # print(BOT.conversation.get_training_data_ids())
    try:
        # data = json.loads(request.data)
        # user_text = data["userInput"]
        conversation = BOT.conversation
        # print("retrieved convo")
        response = BOT.update_state(user_text, conversation)
        response = re.sub('<br>|<li>','\n', response)
        response = re.sub('<...>|<..>|<.>','', response)
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


for i in range(2,6):
    file = "./configs/heartRFSet"+str(i)+"-config.gin"
    args.config = file
    # Parse application level configs
    gin.parse_config_file(args.config)

    # Load the explainbot
    bot = ExplainBot()
    objective = bot.conversation.describe.get_dataset_objective()

    ##### This is the section to update per dataset
    sampleInstance = "1371"
    sampleFeature = "woman"
    print("Heart Data Set "+str(i)+" - RF")
    ############################

    print("-"*80)

    # Question 1
    print("What are the most prominent features?\n")
    print(get_bot_response(bot,"What are the most prominent features?","important"))

    # print("-"*80)
    # # Question 2
    # print("What records does the model predict incorrectly?\n")
    # print(get_bot_response(bot,"What records does the model predict incorrectly?","important"))

    # print("-"*80)
    # # Question 3
    # print("Why is instance " + sampleInstance  + " given this prediction?\n")
    # print(get_bot_response(bot,"Why is instance " + sampleInstance  + " given this prediction?","whatif"))

    # print("-"*80)
    # # Question 4
    # print("What should instance " + sampleInstance  + " change for a different result?\n")
    # print(get_bot_response(bot,"What should instance " + sampleInstance  + " change for a different result?","whatif"))

    # print("-"*80)
    # # Question 5
    # print("Is " + sampleFeature + " used for predictions?\n")
    # print(get_bot_response(bot,"Is " + sampleFeature + " used for predictions?","whatif"))