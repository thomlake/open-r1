"""Reward functions for GRPO training."""

import math
import re
import string
from dataclasses import dataclass
from collections import Counter

# from latex2sympy2_extended import NormalizationConfig
# from math_verify import LatexExtractionConfig, parse, verify


@dataclass
class RewardFunction:
    scale: float = 1.0

    def __call__(
            self,
            completions: list[str],
            **kwargs,
    ) -> list[float]:
        raise NotImplementedError


class text_processing:
    @staticmethod
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    @staticmethod
    def fix_whitespace(text: str) -> str:
        return ' '.join(text.split())

    @staticmethod
    def handle_punctuation(text: str) -> str:
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    @staticmethod
    def lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def replace_underscore(text: str) -> str:
        return text.replace('_', ' ')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.strip()
    s = text_processing.replace_underscore(s)
    s = text_processing.lower(s)
    s = text_processing.handle_punctuation(s)
    s = text_processing.remove_articles(s)
    s = text_processing.fix_whitespace(s)
    s = s.strip()
    return s


def exact_match_score(completion, answer):
    return completion == answer


def f1_score(completion, answer):
    completion_tokens = completion.split()
    answer_tokens = answer.split()
    common = Counter(completion_tokens) & Counter(answer_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = num_same / len(completion_tokens)
    recall = num_same / len(answer_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def contains_score(completion, answer):
    return answer in completion


ANSWER_SCORER_REGISTRY = {
    'exact_match': exact_match_score,
    'f1': f1_score,
    'contains': contains_score,
}


def max_over_answers(score_function, completion, answers):
    scores = []
    for answer in answers:
        scores.append(score_function(completion, answer))

    return max(scores)


def get_short_answer_accuracy_reward(
    scale: float = 2.0,
    score: str = 'exact_match',
    answer_normalization: bool = True,
):
    def short_answer_accuracy_reward(completions, answer, **kwargs) -> list[float]:
        assert len(completions) == len(answer)

        contents = [completion[0]['content'] for completion in completions]
        if isinstance(answer[0], str):
            answer = [[a] for a in answer]

        score_function = ANSWER_SCORER_REGISTRY[score]

        rewards = []
        for content, answer_aliases in zip(contents, answer):
            *_, content = content.rsplit('<answer>', 1)
            content, *_ = content.split('</answer>', 1)

            if answer_normalization:
                content = normalize_answer(content)
                answer_aliases = [normalize_answer(a) for a in answer_aliases]

            reward = max_over_answers(score_function, content, answer_aliases)
            rewards.append(scale * reward)

        return rewards

    return short_answer_accuracy_reward


def get_strict_format_reward(scale: float = 0.5):
    def strict_format_reward(completions, **kwargs):
        pattern = r'^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$'
        contents = [completion[0]['content'] for completion in completions]
        matches = [re.match(pattern, s, re.DOTALL) for s in contents]
        return [scale if match else 0.0 for match in matches]

    return strict_format_reward


def get_soft_format_reward(scale: float = 0.5):
    def soft_format_reward(completions, **kwargs):
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>$'
        contents = [completion[0]['content'] for completion in completions]
        matches = [re.match(pattern, s, re.DOTALL | re.MULTILINE) for s in contents]

        for i, (content, match_result) in enumerate(zip(contents, matches)):
            print(f'[{i=}]\n{content=}\n{match_result=}\n' + 20 * '-')

        raise ValueError
        return [scale if match else 0.0 for match in matches]

    return soft_format_reward


REWARD_FUNCTION_REGISTRY = {
    'short_answer_accuracy': get_short_answer_accuracy_reward,
    'strict_format': get_strict_format_reward,
    'soft_format': get_soft_format_reward,
}


def create_reward_functions(reward_configs: dict[str, dict]):
    return [
        REWARD_FUNCTION_REGISTRY[name](**kwargs)
        for name, kwargs in reward_configs.items()
    ]


# End: TL


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
