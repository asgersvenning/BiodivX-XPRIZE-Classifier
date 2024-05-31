import os, io
from urllib.request import urlretrieve

import gradio as gr

import numpy as np
import torch
from PIL import Image

from flat_bug.predictor import Predictor, TensorPredictions


with gr.Blocks() as app:
    
    # Define the model parameters
    config_file = "config.yml"
    weights = "weights.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # Checks
    if not os.path.exists(config_file):
        raise ValueError(f"Could not find {config_file} file")
    if not os.path.exists(weights):
        urlretrieve("https://anon.erda.au.dk/share_redirect/aRbj0NCBkf/fb_2024-03-18_large_best.pt", weights)
    if not os.path.exists(weights):
        raise ValueError(f"Could not find {weights} file even after downloading?")

    # Create a model loader
    def get_model():
        return Predictor(model=weights, cfg=config_file, device=device, dtype=dtype)
    
    # Load the model in the app state
    model = gr.State(get_model)

    # Define the localization function
    def localize(image : bytes | str | list[bytes] | list[str] | None, do_plot : bool=False) -> dict:
        # Save the image
        if isinstance(image, np.ndarray):
            pass
        elif isinstance(image, list):
            raise NotImplementedError("Batch processing not supported yet")
        elif isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        else:
            raise ValueError(f"Expected image to be a string or bytes, but got {type(image)}")
        
        # Convert the image to a numpy array
        image = np.array(image)

        # Convert the image to a torch tensor
        image = torch.from_numpy(image).permute(2, 0, 1).to(device)
        
        # Run the model
        predictions : TensorPredictions = model.value.pyramid_predictions(image, "temp.123")

        # Plot the image if requested
        if do_plot:
            predict_image = "temp_predict.jpg"
            predictions.plot(outpath=predict_image)
            pil_image = Image.open(predict_image)
        else:
            pil_image = None
        
        # Remove the temporary images
        os.remove(predict_image)

        # Return the predictions as JSON
        return predictions.json_data, pil_image

    # Define the input-output format
    file_input = gr.Image(value="example_image.jpg", label="Input image")
    checkbox = gr.Checkbox(label="Return annotated image")
    output_json = gr.Textbox(label="Output JSON")
    output_image = gr.Image(label="Annotated Image")

    gr.Interface(
        fn=localize,
        inputs=[file_input, checkbox],
        outputs=[output_json, output_image],
        title="Perform localization on a single image",
        description="This model localizes bugs in images. Upload an image and it will return the localization as a JSON file and optionally an annotated image."
    )

if __name__ == "__main__":
    app.launch()