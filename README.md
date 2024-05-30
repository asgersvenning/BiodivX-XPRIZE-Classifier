# BiodivX-XPRIZE-ML-pipeline
Simple repository for managing the overall CV ML execution pipeline of the ETH BiodivX for the XPRIZE Rainforest competition finals.

# Installation
The installation will **only** be verified for `Ubuntu 20.04` with `CUDA 12.2` (TODO: NVIDIA driver version `450.51`?). The system must be pre-installed with `Python` (TODO: version `3.11`?) and `micromamba`.

```bash
. install.sh
```

# Execution
## Activate environment
```bash
micromamba activate xprize_pipeline
cd "$HOME/BiodivX-XPRIZE-ML-pipeline"
```

## Run pipeline
Input should be a path to a zip-folder containing images.
```bash
python pipeline.py --input <INPUT.ZIP>
```

# Output
TODO: Describe output
