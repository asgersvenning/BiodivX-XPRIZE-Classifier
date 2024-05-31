import subprocess
import gradio as gr

# For now I just want a **VERY** simply dummy script that can take a zip-folder of files and returns a new zip-folder with a single text file containing the text of all the files in the original zip-folder

def pipeline(input_zip_path, output_zip_path):
    subprocess.Call(["python", "pipeline.py", input_zip_path, output_zip_path])
    return output_zip_path

# Define the input-output format
inputs = gr.inputs.File(label="Input Zip File")
outputs = gr.outputs.File(label="Output Zip File")

# Define the interface
interface = gr.Interface(fn=pipeline, inputs=inputs, outputs=outputs, title="Pipeline")

if __name__ == "__main__":
    interface.launch()