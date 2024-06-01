*There is no HuggingFace Space description here, as it is created by the GitHub Action which is triggered by the push event on the main branch.*

# BiodivX-XPRIZE-Classifier
Simple repository for running insect classification of the ETH BiodivX for the XPRIZE Rainforest competition finals.

# Installation
The installation will **only** be verified for `Ubuntu 22.04` with `CUDA 12.2` with NVIDIA driver version `535.161.07`. The system must be pre-installed with `Python 3.11.5` and `micromamba`.

```bash
. install.sh
```

# Execution
## Activate environment

```bash
micromamba activate xprize_classifier
cd "$HOME/BiodivX-XPRIZE-Classifier"
```

## Run pipeline
Input should be a path or glob pattern to the image(s) to classify.

```bash
python classify.py [-h] --input <INPUT_IMAGES> [--output <OUTPUT_CSV>] [--weights <WEIGHT_PATH>] [--class_dict <CLASS_DICT_CSV>] [--device <DEVICE>]
```

# Output
The output is a formatted CSV which will be output in `stdout` or in the file specified by the `-o`/`--output` flag.
