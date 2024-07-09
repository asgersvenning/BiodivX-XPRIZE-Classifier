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
Input should be a path or glob pattern to the image(s) to classify. If it is a .txt file, then it should contain lines corresponding to the former.

```bash
python classify.py [-h] --input/-i [<INPUT_IMAGES> ...] [--output/-o <OUTPUT_CSV>] [--output_type/-t <OUTPUT_TYPE>] [--model_type/-m <MODEL_TYPE>] [--include_embeddings] [--weights <WEIGHT_PATH>] [--class_handles <CLASS_HANDLES_JSON>] [--device <DEVICE>]
```

**Example:**

```bash
python classify.py --input example_image1.jpg example_image2.png
```

# Output
The output is a formatted CSV (delimiter `,`) which will be output in `stdout` or in the file specified by the `-o`/`--output` flag.

The CSV has the columns:

1) `Predicted`: The predicted class of the image.
2) `Confidence`: The confidence of the prediction.
3) `<Class_0>`: The confidence of the prediction for the first class with the name `<Class_0>`.
4) `<Class_1>`: The confidence of the prediction for the second class with the name `<Class_1>`.

    .   
    .   
    .

5) `<Class_N>`: The confidence of the prediction for the last class with the name `<Class_N>`.

All cells follow the following format `<SPACE><CONTENT><SPACE>*` where `CONTENT` is the content of the cell and `<SPACE>` is a space character. The number of spaces afterwards vary, and are determined to ensure that all cells in a column have the same width, when displayed in a monospaced font.