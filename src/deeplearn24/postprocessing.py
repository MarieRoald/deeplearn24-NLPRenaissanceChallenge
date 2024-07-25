"""
Irreguaritie 1:
u and v are used interchangeably
- Assume u at beginning of word
- Assume v inside word

Irregularity 2:
f and s are used interchangeably
- Could be either f or s at beginning of a word
- Could be either f or s inside a word
- Sometimes both are present in a word
- Assume s at the beginning/end of a word, f within a word

Irregularity 3:
Tildes (horizontal “cap” – ignore grave/backwards accents)
- When a q is capped, assume ue follows
- When a vowel is capped, assume n follows
- When n is capped, this is always the letter ñ

Irregularity 4
ç old spelling is always modern z
- always interpret ç as z

Irregularity 5:
Some line end hyphens not present
- Leave words split for now

One solution is to create a regex pattern that allows for the specified character switches.
Could also check against dictionary of known words to see if the word is valid.
"""

import json
import re
import string
import unicodedata
from functools import lru_cache, partial
from itertools import chain
from pathlib import Path
from typing import Callable

from Levenshtein import distance
from typing_extensions import Unpack


def is_matching_word(word: str, match_pattern: re.Pattern) -> bool:
    return bool(match_pattern.match(word))


def compare_words(word1: str, word2: str) -> bool:
    return process_word(word1).lower() == process_word(word2).lower()


@lru_cache(2**20)  # ~1M
def lookup_valid_word(word: str, dictionary: frozenset[str]) -> bool:
    matches = [w for w in dictionary if compare_words(word, w)]
    # choose the match with shortest levenstein distance
    if matches:
        best_match = min(matches, key=partial(distance, word))
        return best_match
    return None


def remove_punctuation(word: str) -> str:
    # Use a generator expression to filter out punctuation characters
    return "".join(char for char in word if char not in string.punctuation)


def replace_word_with_punctuation(text: str, text_without_punctuation: str, new_word: str) -> str:
    """Replace the old word in the text with the new word, preserving the case of the original word.

    This will not work if the old word has punctuation inside it. For example, "e.g." will not be
    matched by anything in the text, because the period is removed from the text without punctuation.
    """

    def replace_match(match):
        word = match.group(1)
        punctuation = match.group(2)

        # Preserve the case of the original word
        if word.isupper():
            new_word_case = new_word.upper()
        elif word.istitle():
            new_word_case = new_word.title()
        elif word.islower():
            new_word_case = new_word.lower()
        else:
            new_word_case = new_word

        return new_word_case + punctuation

    # Regular expression to match the word and any trailing punctuation
    pattern = re.compile(
        r"(\b" + re.escape(text_without_punctuation) + r"\b)([\W_]*)", re.IGNORECASE
    )
    return pattern.sub(replace_match, text)


@lru_cache(2**20)  # ~1M
def process_word(word):
    """
    Irreguaritie 1:
    u and v are used interchangeably
    - Assume u at beginning of word
    - Assume v inside word

    Irregularity 2:
    f and s are used interchangeably
    - Could be either f or s at beginning of a word
    - Could be either f or s inside a word
    - Sometimes both are present in a word
    - Assume s at the beginning/end of a word, f within a word

    Irregularity 3:
    Tildes (horizontal "cap" – ignore grave/backwards accents)
    - When a q is capped, assume ue follows
    - When a vowel is capped, assume n follows
    - When n is capped, this is always the letter ñ

    Irregularity 4
    ç old spelling is always modern z
    - always interpret ç as z

    Irregularity 5:
    Some line end hyphens not present
    - Leave words split for now
    """
    word = unicodedata.normalize("NFKD", word)
    vowels = "aeiou"
    new_word = []
    for idx, char in enumerate(word):
        if char == "u" or char == "v":
            if idx == 0:
                new_character = "u"
            else:
                new_character = "v"

        elif char == "s" or char == "f":
            if idx == 0 or idx == len(word) - 1:
                new_character = "s"
            else:
                new_character = "f"

        elif char == "\u0303":
            if word[idx - 1] == "q":
                if word[idx + 1 : idx + 3] == "ue":
                    new_character = ""
                else:
                    new_character = "ue"
            elif word[idx - 1] in vowels:
                if word[idx + 1] == "n":
                    new_character = ""
                else:
                    new_character = "n"

        elif char == "\u0327" and word[idx - 1] == "c":
            new_word[-1] = "z"
            new_character = ""
        else:
            new_character = char
        new_word.append(new_character)
    return "".join(new_word)


def post_process_text(text: str, unique_words: frozenset[str]) -> str:
    processed_words = []
    for word in text.split():
        word_without_punctuation = remove_punctuation(word)

        dictionary_match = lookup_valid_word(
            word_without_punctuation,
            unique_words,
        )
        if dictionary_match:
            new_word = replace_word_with_punctuation(
                word, word_without_punctuation, dictionary_match
            )
        else:
            new_word = process_word(word)
        processed_words.append(new_word)

    return " ".join(processed_words)


def load_dictionaries(*dictionary_paths: Unpack[Path]) -> frozenset[str]:
    output = chain.from_iterable(
        json.loads(dictionary_path.read_text()) for dictionary_path in dictionary_paths
    )

    return frozenset(output)
