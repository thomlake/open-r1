
SIMPLE_SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

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
    'r1': R1_SYSTEM_PROMPT,
}


def get_system_prompt(name: str):
    return PROMPT_REGISTRY[name]
