# DeltaScore

Welcome to **DeltaScore: Fine-Grained Story Evaluation with Perturbations**, which has been published in [EMNLP2023](https://2023.emnlp.org).

## Overview

The goal of this project is to propose a new methodology to conduct fine-grained story evaluation by differentiating perturbations.
We have demonstrated that DeltaScore can outperform evaluation methods that use likelihood directly, and you can find the details in our [published paper](https://arxiv.org/abs/2303.08991).

## Perturbation Utilities

The auxiliary `eva/perturb` module used by the original experiments has been
restored from an old project backup. The restored directory includes the source
code and small resource tables needed by several perturbation types.

Some large auxiliary resources are not committed to this repository:
`conceptnet_triple.csv`, `kg.txt`, and `word2kg.txt`. They are about 300 MB in
the archived experiment directory and are only needed for the commonsense
knowledge-graph perturbation path. The original external mirror links are kept
in `eva/perturb/download.sh`, but those mirrors may no longer return raw data
files.

The original code depends on older NLP tooling, including spaCy, NLTK/WordNet,
and CLiPS `pattern.en`. Some perturbation paths may require the original Python
environment for exact reproduction.

## Contact

If you have any questions, feedback, feel free to reach out to Zhuohan Xie at zhuohanx@student.unimelb.edu.au. We welcome any inquiries and look forward to hearing from you!
