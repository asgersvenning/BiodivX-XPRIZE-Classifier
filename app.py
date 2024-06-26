import os

from typing import List, Tuple, Optional, Union

import numpy as np

import gradio as gr

from classify import Resnet50Classifier, HierarchicalClassifier, get_defaults, dict2csv, cell_formatter

def postprocess(predictions):
    return (
        [f'{c} ({cell_formatter(s, digits=1)})' for c, s in zip(predictions["class"], predictions["score"])], 
        [dict2csv({k : [v[i]] for k, v in predictions["data"].items()}, write=True) for i in range(len(predictions["data"]["Predicted"]))]
    )

with gr.Blocks() as demo:

    class_dict, weights, device, dtype = get_defaults()

    # Create button for enabling, disabling TTA
    tta_button = gr.Button(value="❌ Test-Time Augmentation is off! ❌", variant="secondary", size="sm")

    # Create a function to toggle TTA
    def toggle_tta():
        model.value.tta_enabled = not model.value.tta_enabled
        return "✅ Test-Time Augmentation is on! ✅" if model.value.tta_enabled else "❌ Test-Time Augmentation is off! ❌"

    # Create a model loader
    def get_model():
        # return Resnet50Classifier(weights=weights, category_map=class_dict, device=device, test_time_augmentation=False)
        return HierarchicalClassifier(weights=weights, device=device, test_time_augmentation=False)
    
    # Load the model in the app state
    model = gr.State(get_model)

    tta_button.click(toggle_tta, outputs=[tta_button])

    # Define the localization function
    def classify(images : Optional[Union[np.ndarray, bytes, str, Union[List[Union[np.ndarray, bytes, str]], Tuple[Union[np.ndarray, bytes, str]]]]]) -> Tuple[List[str], List[str]]:
        if not os.path.exists("output"):
            os.makedirs("output")
        return postprocess(model.value.predict(images))

    # Define the input-output format
    file_input = gr.Image(value="example_image1.jpg", label="Input image")
    label_output = gr.Textbox(label="Classification", type="text")
    rich_output = gr.File(label="Rich Output")

    gr.Interface(
        fn=classify,
        inputs=[file_input],
        outputs=[label_output, rich_output],
        title="Perform classification on a single image",
        description="This model classifies images of single bugs",
        batch=True
    )

if __name__ == "__main__":
    demo.launch()