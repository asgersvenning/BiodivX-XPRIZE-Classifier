### Placeholder for orchestrating the execution pipeline ###
# We will run each of the dependency scripts as a subprocess or os.system
# Script will also handle input-output formatting

import os
import gradio as gr

import zipfile

# For now I just want a **VERY** simply dummy script that can take a zip-folder of files and returns a new zip-folder with a single text file containing the text of all the files in the original zip-folder

def pipeline(input_zip_path, output_zip_path):
    # Read the input zip file contents 
    input_content = []
    with zipfile.ZipFile(input_zip_path, 'r') as z:
        for file in z.filelist:
            with z.open(file.filename, 'r') as f:
                input_content.append(f.read().decode('utf-8'))

    # Write the output zip file contents
    with zipfile.ZipFile(output_zip_path, 'w') as z:
        with z.open('output.txt', 'w') as f:
            f.write('\n'.join(input_content).encode('utf-8'))

    return output_zip_path

# Define the input-output format
inputs = gr.inputs.File(label="Input Zip File")
outputs = gr.outputs.File(label="Output Zip File")

# Define the interface
interface = gr.Interface(fn=pipeline, inputs=inputs, outputs=outputs, title="Pipeline")

if __name__ == "__main__":
    interface.launch()

