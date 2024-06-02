import os, sys, csv, io, uuid, contextlib, re, glob

from urllib.request import urlretrieve
from typing import List, Tuple, Optional, Union, Callable

import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

# Input parsing
IMG_REGEX = re.compile(r'\.(jp[e]{0,1}g|png)$', re.IGNORECASE)

def is_image(file_path):
    return bool(re.search(IMG_REGEX, file_path)) & os.path.isfile(file_path)

def is_dir(file_path):
    return os.path.isdir(file_path)

def is_glob(file_path):
    return not (is_image(file_path) or is_dir(file_path))

def type_of_path(file_path):
    if is_image(file_path):
        return 'image'
    elif is_dir(file_path):
        return 'dir'
    elif is_glob(file_path):
        return 'glob'
    else:
        return 'unknown'
    
def get_images(input_path_dir_globs : List[str]) -> List[str]:
    images = []
    for path in input_path_dir_globs:
        match type_of_path(path):
            case 'image':
                images.append(path)
            case 'dir':
                images.extend(glob.glob(os.path.join(path, '*')))
            case 'glob':
                images.extend(glob.glob(path))
            case _:
                raise ValueError(f"Unknown path type: {path}")
    if len(images) == 0:
        raise ValueError("No images found")
    return images

# CSV functions
def get_col_width(d : dict, formatter : Callable = str, **kwargs) -> List[int]:
    col_widths = [len(k) for k in d.keys()]
    for i, col in enumerate(d.keys()):
        col_widths[i] = max(col_widths[i], max([len(formatter(v, **kwargs)) for v in d[col]]))
    return col_widths

def cell_formatter(v : Union[int, float, str], digits : int) -> str:
    if isinstance(v, float):
        return f"{v * 100:.{digits}f}%"
    return str(v)

def dict2csv(d : dict, digits : int=1, write : bool=False) -> str:
    csv = []
    columns = list(d.keys())
    col_widths = get_col_width(d, cell_formatter, digits=digits)
    row_spec = ", ".join([f"{{:<{col_widths[i]}}}" for i in range(len(columns))])
    header = row_spec.format(*columns)
    # horiz_rule = "-" * len(header)
    # csv.append(horiz_rule)
    csv.append(header)
    # csv.append(horiz_rule)
    rows = set([len(v) for v in d.values()])
    if len(rows) != 1:
        raise ValueError("All values in the dictionary must have the same length")
    rows = list(rows)[0]
    for row in range(rows):
        csv.append(row_spec.format(*[cell_formatter(d[col][row], digits) for col in columns]))
    # csv.append(horiz_rule)
    csv_content = "\n".join(csv)
    if write:
        unique_prefix = str(uuid.uuid4())[::3]
        csv_path = f"output/{unique_prefix}_rich_classification.csv"
        with open(csv_path, "w") as f:
            f.write(csv_content)
        return csv_path
    else:
        return csv_content

# Model definition
class ResNet50(nn.Module):
    def __init__(self, num_classes : int=20):
        """
        Initialize model.
        """
        super(ResNet50, self).__init__()

        self.expansion = 4
        self.out_channels = 512
        
        self.model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        self.model_ft.fc = nn.Identity() # Do nothing just pass input to output

        self.drop = nn.Dropout(p=0.5)
        self.linear_lvl1 = nn.Linear(self.out_channels*self.expansion, self.out_channels)
        self.relu_lv1 = nn.ReLU(inplace=False)
        self.softmax_reg1 = nn.Linear(self.out_channels, num_classes)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of pretrained ResNet-50.
        """
        x = self.model_ft(x)
        x = self.drop(x) 
        x = self.linear_lvl1(x)
        x = self.relu_lv1(x)
        x = self.softmax_reg1(x)
        return x

def get_defaults():
    # Define the model parameters
    remote_dir = "https://anon.erda.au.dk/share_redirect/aRbj0NCBkf"
    class_dict = "class_dict.csv"
    remote_class_dict = "mcc24/model_order_24.05.10/thresholds.csv"
    weights = "classification_weights.pth"
    remote_weight = "mcc24/model_order_24.05.10/dhc_best_128.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Checks
    if not os.path.exists(class_dict):
        urlretrieve(f"{remote_dir}/{remote_class_dict}", class_dict)
    if not os.path.exists(class_dict):
        raise ValueError(f"Could not find {class_dict} file even after downloading?")
    if not os.path.exists(weights):
        urlretrieve(f"{remote_dir}/{remote_weight}", weights)
    if not os.path.exists(weights):
        raise ValueError(f"Could not find {weights} file even after downloading?")
    
    return class_dict, weights, device, dtype

def parse_image(images : Optional[Union[np.ndarray, bytes, str, Union[List[Union[np.ndarray, bytes, str]], Tuple[Union[np.ndarray, bytes, str]]]]], device : Union[torch.device, str]="cpu"):
    # Cases:
    # List: Recursively parse each image
    if isinstance(images, (list, tuple)):
        return [parse_image(image, device) for image in images]
    # String: Open the image with PIL
    elif isinstance(images, str):
        images = Image.open(images)
    # Bytes: Open the image with PIL using a BytesIO
    elif isinstance(images, bytes):
        images = Image.open(io.BytesIO(images))
    # Numpy array: Do nothing
    elif isinstance(images, np.ndarray):
        pass
    # Other: Raise an error
    else:
        raise ValueError(f"Expected image(s) to be a np.ndarray, string or bytes, or list of these, but got {type(images)}")
    
    # Convert the image to a numpy array
    image = np.array(images)

    # Convert the image to a torch tensor and change from HWC to CHW
    image = torch.from_numpy(image).permute(2, 0, 1).to(device)

    # Check if the image has an alpha channel
    if image.shape[0] == 4:
        alpha = image[3]
        image = image[:3]
        image[:, alpha == 0] = 255

    return image

class toFloat32:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image):
        return (image.float() / 255.0)

class Resnet50Classifier(torch.nn.Module):
    input_size = 300
    batch_size = 16

    def __init__(self, weights, category_map, device):
        super().__init__()
        self.weights = weights
        with open(category_map) as f:
            reader = csv.reader(f)
            self.category_map = {int(rows[0]): rows[1] for i, rows in enumerate(reader) if i != 0}
        self.device = device
        self.model = self.get_model()
        self.transforms = self.get_transforms()

    def get_model(self):
        num_classes = len(self.category_map)
        model = ResNet50(num_classes=num_classes)
        model = model.to(self.device)
        # state_dict = torch.hub.load_state_dict_from_url(weights_url)
        checkpoint = torch.load(self.weights, map_location=self.device)
        # The model state dict is nested in some checkpoints, and not in others
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_transforms(self) -> torchvision.transforms.Compose:
        mean, std = [0, 0, 0], [1, 1, 1] # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.input_size, self.input_size)),
                toFloat32(),
                torchvision.transforms.Normalize(mean, std),
            ]
        )

    def post_process_batch(self, output):
        predictions = torch.nn.functional.softmax(output, dim=1)

        categories = predictions.argmax(dim=1).tolist()
        labels = [self.category_map[cat] for cat in categories]
        scores = predictions.max(dim=1).values.tolist()

        data = dict()
        data["Predicted"] = labels
        data["Confidence"] = scores
        data.update({
            self.category_map[c] : predictions[:, c].tolist() for c in range(predictions.shape[1])
        })

        return {
            "class" : labels,
            "score" : scores,
            "data" : data
        }
    
    def predict(self, images):
        images = parse_image(images, self.device)
        if not isinstance(images, list):
            images = [images] 
        images = [self.transforms(image) for image in images]
        images = torch.stack(images)
        outputs = torch.zeros((len(images), len(self.category_map)), device=self.device, dtype=torch.float32)
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i+self.batch_size]
            with torch.no_grad():
                output = self.model(batch)
            outputs[i:i+self.batch_size] = output
        
        return self.post_process_batch(outputs)

def main(args : dict):
    output_stream = io.StringIO() if not args["output"] else sys.stdout
    # Supress all prints here, to ensure that the output is only the JSON string if the output is not specified
    with contextlib.redirect_stdout(output_stream):
        # Get the defaults
        class_dict, weights, device, dtype = get_defaults()

        input_images = get_images(args["input"])

        # Update the parameters
        if args["class_dict"]:
            class_dict = args["class_dict"]

        if args["weights"]:
            weights = args["weights"]

        if args["device"]:
            device = torch.device(args["device"])

        # Create the model
        model = Resnet50Classifier(weights=weights, category_map=class_dict, device=device)

        # Run the model
        output = dict2csv(model.predict(input_images)["data"])

    # Save the output
    if args["output"]:
        if os.path.isdir(args["output"]):
            raise ValueError("Output cannot be a directory.")
        with open(args["output"], "w") as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    import argparse

    args_parser = argparse.ArgumentParser(description="Classify images of bugs.")
    args_parser.add_argument('-i', '--input', type=str, nargs="+", help='Path(s), director(y/ies) or glob(s) to such. Outputs will be saved in the output directory.', required=True)
    args_parser.add_argument("-o", "--output", type=str, help="The output CSV file. If not specified it will be dumped in stdout.", required=False)
    args_parser.add_argument("--weights", type=str, help="The path to the weights file. Default is 'classification_weights.pth'.")
    args_parser.add_argument("--class_dict", type=str, help="The path to the class_dict CSV. Default is 'class_dict.csv'.")
    args_parser.add_argument("--device", type=str, help="The device to use defaults to 'cuda:0' if available else 'cpu'")
    args = args_parser.parse_args()

    main(vars(args))