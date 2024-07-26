from pathlib import Path
from deeplearn24.postprocessing import load_dictionaries, post_process_text, remove_punctuation
from time import monotonic_ns

training_data = "This is a test text with various words like save, safe, fave and surf. que Here is more text with more words farris. vann slukker ild. fozef"

# Create a "dictionary" of known unique words
unique_words = frozenset(remove_punctuation(training_data).lower().split())
dictionaries = load_dictionaries(
    Path("data/0_input/sbwce-corpus/dictionary.json"),
    Path("data/0_input/dataset_words/dictionary.json"),
)

example_data = "Lets saue. the world and Surs the web! faffef q̃r uvis que caçõru. Farris slukker tørsten! fof foçes. "

print("STARTING")
t0 = monotonic_ns()
processed_text = post_process_text(
    example_data,
    dictionaries,
)
print("ENDING", 1e-9 * (monotonic_ns() - t0))
print(processed_text)
