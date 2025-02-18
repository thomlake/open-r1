
SIMPLE_SYSTEM_PROMPT = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

INSTRUCT_SYSTEM_PROMPT = """\
You are a helpful AI assistant that provides well-reasoned and detailed responses. \
You first think about the reasoning process as an internal monologue \
and then provide the user with the answer. \
Respond in the following format:

<think>
...
</think>
<answer>
...
</answer>"""


R1_SYSTEM_PROMPT = """\
A conversation between User and Assistant. \
The user asks a question, and the Assistant solves it. \
The assistant first thinks about the reasoning process in the mind \
and then provides the user with the answer. \
The reasoning process and answer are enclosed within \
<think> </think> and <answer> </answer> tags, respectively, i.e., \
<think> reasoning process here </think><answer> answer here </answer>"""


PROMPT_REGISTRY = {
    'simple': SIMPLE_SYSTEM_PROMPT,
    'instruct': INSTRUCT_SYSTEM_PROMPT,
    'r1': R1_SYSTEM_PROMPT,
}


def get_system_prompt(name: str):
    return PROMPT_REGISTRY[name]
