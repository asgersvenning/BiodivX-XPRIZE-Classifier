import os

from typing import List, Tuple, Optional, Union

import numpy as np

import gradio as gr

from classify import HierarchicalClassifier, FastaiClassifier, get_defaults, dict2csv, cell_formatter

def postprocess(predictions):
    return (
        [f'{c} ({cell_formatter(s, digits=1)})' for c, s in zip(predictions["class"], predictions["score"])], 
        [dict2csv({k : [v[i]] for k, v in predictions["data"].items()}, digits=3, write=True) for i in range(len(predictions["data"]["Predicted"]))]
    )

with gr.Blocks() as demo:

    # Create button for enabling, disabling TTA
    tta_button = gr.Button(value="❌ Test-Time Augmentation is off! ❌", variant="secondary", size="sm")
    # Dropdown for choosing between "fastai" and "hierarchical model"
    model_dropdown = gr.Dropdown(["fastai", "hierarchical"], label="Model", value="fastai")
    # dummy_label = gr.Label("Dummy label")

    # Create a function to toggle TTA
    def toggle_tta(state):
        state.tta_enabled = not state.tta_enabled
        return "✅ Test-Time Augmentation is on! ✅" if state.tta_enabled else "❌ Test-Time Augmentation is off! ❌"

    # Create a model loader
    def get_model(mtype : str):
        class_dict, weights, device, dtype = get_defaults(mtype)
        if mtype == "fastai":
            return FastaiClassifier(class_handles=class_dict, weights=weights, device=device, dtype=dtype, test_time_augmentation=False)
        elif mtype == "hierarchical":
            return HierarchicalClassifier(weights=weights, device=device, test_time_augmentation=False)
        else:
            raise RuntimeError("Unexpected model type")

    def update_model(mtype : str, state):
        tta = state.tta_enabled
        model = get_model(mtype)
        model.tta_enabled = tta
        return model
    
    # Load the model in the app state
    model = gr.State(get_model(model_dropdown.value))
    # model.change(update_model, inputs=model_dropdown, outputs=model)

    # # Create callback to update the model when the dropdown changes
    model_dropdown.change(update_model, inputs=[model_dropdown, model], outputs=[model])

    # Create callback to update the button text when TTA is toggled
    tta_button.click(toggle_tta, inputs=[model], outputs=[tta_button])

    # Define the localization function
    def classify(state : gr.State, images : Optional[Union[np.ndarray, bytes, str, Union[List[Union[np.ndarray, bytes, str]], Tuple[Union[np.ndarray, bytes, str]]]]]) -> Tuple[List[str], List[str]]:
        if not os.path.exists("output"):
            os.makedirs("output")
        psout = postprocess(state[0].predict(images))
        return state, psout[0], psout[1]

    # Define the input-output format
    file_input = gr.Image(value="example_image1.jpg", label="Input image")
    label_output = gr.Textbox(label="Classification", type="text")
    rich_output = gr.File(label="Rich Output")

    gr.Interface(
        fn=classify,
        inputs=[model, file_input],
        outputs=[model, label_output, rich_output],
        title="Perform classification on a single image",
        description="This model classifies images of single bugs",
        batch=True
    )

if __name__ == "__main__":
    demo.launch()