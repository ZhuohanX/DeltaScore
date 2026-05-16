# Perturbation utilities

This directory contains the auxiliary perturbation module used by the original
DeltaScore experiments. It was restored from an old project backup after the
public repository was found to be missing `eva/perturb`.

Included here:

- `perturb.py` and `utils.py`
- small resource tables for negation, cause, pronoun, and time perturbations
- `back_trans_data/story_bt.json`

Not included here:

- `conceptnet_triple.csv`
- `kg.txt`
- `word2kg.txt`

Those files are large auxiliary resources, about 300 MB in total in the
archived experiment directory. They are only needed by the commonsense
knowledge-graph perturbation path. The original external mirror links are kept
in `download.sh`, but they may no longer return raw data files.

The original environment used older NLP dependencies, including spaCy,
NLTK/WordNet, and CLiPS `pattern.en`. Some `pattern.en` code paths may need an
older Python environment to reproduce exactly.
