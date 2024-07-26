import string

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
"""


def process_word(word):
    capped_wovels = "ãẽõũỹ"
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

        elif char == "q":
            if idx < len(word) - 2:
                if word[idx : idx + 2] == "ue":
                    new_character = "q"
            else:
                new_character = "que"

        elif char in capped_wovels:
            if idx < len(word) - 1:
                if word[idx + 1] == "n":
                    new_character = char
                else:
                    new_character = char + "n"
            else:
                new_character = char + "n"

        elif char == "ç":
            new_character = "z"
        new_character.append(new_character)
    return "".join(new_word)
