
Source code for experiments with the GoldHamster corpus, which is described in the manuscript:

**Automatic classification of experimental models in biomedical literature to support searching for alternative methods to animal experiments**
Mariana Neves, Antonina Klippert, Fanny Knöspel, Juliane Rudeck, Ailine Stolz, Zsofia Ban, Markus Becker, Kai Diederich, Barbara Grune, Pia Kahnau, Nils Ohnesorge, Johannes Pucher, Ines Schadock, Gilbert Schönfelder, Bettina Bert and Daniel Butzke, *Journal of Biomedical Semantics*, (under review).

The [corpus](https://doi.org/10.5281/zenodo.7152295) is available for download. The abstracts can be retrieved using [PubMed E-Utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/) or downloaded from [PubAnnotation GoldHamster project](http://pubannotation.org/projects/GoldHamster).

## Installation

The script works with Python 3.7. The following libraries should be installed:

- pandas
- [transformers](https://huggingface.co/docs/transformers/installation) for CPU: transformers[tf-cpu] and transformers[torch]

## Use

Download the abstracts and the data to their folders, [TRAIN_DEV_TEST_DIR] and [DOCS_DIR], respectively, and update them in the script.


