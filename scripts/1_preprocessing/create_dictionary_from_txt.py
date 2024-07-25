import json
import unicodedata
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def create_dictionary(corpus_path: Path, num_words: int, num_lines: int) -> frozenset[str]:
    corpus = Counter()

    with open(corpus_path, "r") as file:
        for line in tqdm(file, desc="Processing lines", total=num_lines):
            words = unicodedata.normalize("NFKD", line.strip().lower()).split()
            corpus.update(Counter(words))

    return frozenset(word for word, _ in corpus.most_common(num_words))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("corpus_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--num_words", type=int, default=10_000)
    parser.add_argument("--num_lines", type=int, default=None)

    args = parser.parse_args()

    corpus = create_dictionary(args.corpus_path, args.num_words, args.num_lines)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as file:
        json.dump(list(corpus), file)
