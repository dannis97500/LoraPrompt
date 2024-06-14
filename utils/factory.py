from methods.sprompt import SPrompts
from methods.prefix_prompt_tuning import PrefixPromptTuning
from methods.dual_prompt_tuning import DualPromptTuning
from methods.hide_prompt_tuning import HidePrompt
from methods.prefix_kan_prompt_tuning import PrefixKANPromptTuning
from methods.lora_prompt import LoraPrompt
def get_model(model_name, args):
    name = model_name.lower()
    options = {
        'sprompts': SPrompts,
        'prefix_one_prompt':PrefixPromptTuning,
        'dual_prompt':DualPromptTuning,
        'hide_prompt':HidePrompt,
        'prefix_kan_prompt': PrefixKANPromptTuning,
        "lora_prompt":LoraPrompt
        }
    return options[name](args)

