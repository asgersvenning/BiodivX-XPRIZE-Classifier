*There is no HuggingFace Space description here, as it is created by the GitHub Action which is triggered by the push event on the main branch.*

# BiodivX-XPRIZE-Localizer
Simple repository for running insect localization of the ETH BiodivX for the XPRIZE Rainforest competition finals.

# Installation
The installation will **only** be verified for `Ubuntu 22.04` with `CUDA 12.2` with NVIDIA driver version `535.161.07`. The system must be pre-installed with `Python 3.11.5` and `micromamba`.

```bash
. install.sh
```

# Execution
## Activate environment

```bash
micromamba activate xprize_localizer
cd "$HOME/BiodivX-XPRIZE-Localizer"
```

## Run pipeline
Input should be a path to a zip-folder containing images.

```bash
python pipeline.py [-h] -i INPUT_ZIP_PATH [-o OUTPUT_ZIP_PATH]
```

# Output
TODO: Describe output
