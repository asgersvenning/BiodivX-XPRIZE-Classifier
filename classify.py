import os, sys, csv, json, io, uuid, contextlib, re, glob

from io import BytesIO
from urllib.request import urlretrieve
from typing import List, Tuple, Dict, Optional, Union, Callable, Any

import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.transforms import functional as tvf

from fastai.vision.learner import load_learner
from xprize_insectnet.hierarchical.model import model_from_state_file

# Constants
key_to_name_path = os.path.join(os.path.dirname(__file__), "clean_class_to_name.json")
if not os.path.exists(key_to_name_path):
    urlretrieve("https://anon.erda.au.dk/share_redirect/aRbj0NCBkf/clean_class_to_name.json", key_to_name_path)
with open(key_to_name_path) as f:
    KEY_TO_NAME = json.load(f)

# Input parsing 
IMG_REGEX = re.compile(r'\.(jp[e]{0,1}g|png)$', re.IGNORECASE)

def is_image(file_path):
    return bool(re.search(IMG_REGEX, file_path)) and os.path.isfile(file_path)

def is_txt(file_path):
    return file_path.endswith('.txt') and os.path.isfile(file_path)

def is_dir(file_path):
    return os.path.isdir(file_path)

def is_glob(file_path):
    return not (is_image(file_path) or is_dir(file_path))

def type_of_path(file_path):
    if is_image(file_path):
        return 'image'
    elif is_txt(file_path):
        return 'txt'
    elif is_dir(file_path):
        return 'dir'
    elif is_glob(file_path):
        return 'glob'
    else:
        return 'unknown'

def get_images(input_path_dir_globs : Union[str, List[str]]) -> List[str]:
    if isinstance(input_path_dir_globs, str):
        input_path_dir_globs = [input_path_dir_globs]
    images = []
    for path in input_path_dir_globs:
        match type_of_path(path):
            case 'image':
                images.append(path)
            case 'txt':
                with open(path, 'r') as f:
                    paths = [path.strip() for path in f.readlines() if len(path.strip()) > 0]
                images.extend(get_images(paths))
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

def get_defaults():
    # Define the model parameters
    remote_dir = "https://anon.erda.au.dk/share_redirect/aRbj0NCBkf"
    class_dict = "class_dict.csv"
    remote_class_dict = "mcc24/model_order_24.05.10/thresholds.csv"
    weights = "efficientnet_v2_s___hierarchical.state" # "classification_weights.pth"
    remote_weight = "hierarchical/effnetv2s_sgfoc_train_v3_1/efficientnet_v2_s___epoch_8_batch_22000.state" # "mcc24/model_order_24.05.10/dhc_best_128.pth"
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

    # Pad images to square
    h, w = image.shape[1:]
    if h != w:
        size = max(h, w)
        horizontal_error = size - w
        vertical_error = size - h
        left = horizontal_error // 2
        right = horizontal_error - left
        top = vertical_error // 2
        bottom = vertical_error - top
        image = torch.nn.functional.pad(image, (left, right, top, bottom), value=255)

    return image

def plot_images(images : torch.Tensor):
    n_images = len(images)
    nrow = int(n_images ** 0.5)
    torchvision.utils.save_image(images, "output.png", nrow=nrow)

class toFloat32:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image):
        return (image.float() / 255.0)

class BaseClassifier(torch.nn.Module):
    def __init__(self, weights : Any, device : Any, test_time_augmentation : bool=False, *args, **kwargs):
        self.weights = weights
        self.device = device
        self.tta_enabled = test_time_augmentation
        super().__init__(*args, **kwargs)
        self.model = self.get_model()
        self.transforms = self.get_transforms()
        self.augmentations = self.get_augmentations()
    
    def get_model(self):
        raise NotImplementedError("get_model method must be implemented in the subclass")
        # IMPLEMENT HERE ==> Define and load the model to be used for inference.
    
    def get_transforms(self) -> torchvision.transforms.Compose:
        raise NotImplementedError("get_transforms method must be implemented in the subclass")
        # IMPLEMENT HERE ==> Define the transformations to be used during inference; i.e. preprocessing the input.
    
    def get_augmentations(self) -> torchvision.transforms.Compose:
        raise NotImplementedError("get_augmentations method must be implemented in the subclass")
        # IMPLEMENT HERE ==> Define the augmentations to be used during test time augmentation.
    
    def post_process_batch(self, output : Any) -> dict:
        raise NotImplementedError("post_process_batch method must be implemented in the subclass")
        # IMPLEMENT HERE ==> Take whatever output the model gives and convert it to a dictionary of the following format:
        # return {
        #   "class" : List[str], # The predicted class for each image
        #   "score" : List[float], # The confidence score for the predicted class for each image
        #   "data" : dict # The data dictionary containing the scores for all classes for each image
        # }

    def predict(self, images : Optional[Union[str, bytes, np.ndarray, List[Optional[Union[str, bytes, np.ndarray]]], Tuple[Optional[Union[str, bytes, np.ndarray]]]]]) -> dict:
        raise NotImplementedError("predict method must be implemented in the subclass")
        # IMPLEMENT HERE ==> Batched inference of model on `images` -> `output`:
        # output = ...
        # return self.post_process_batch(output)
    
class HierarchicalClassifier(BaseClassifier):
    batch_size = 16

    def __init__(self, dtype=torch.bfloat16, *args, **kwargs):
        self.dtype = dtype
        super().__init__(*args, **kwargs)

    @property
    def n_levels(self):
        return len(self.model.class_handles["n_classes"])

    def get_model(self):
        model = model_from_state_file(self.weights, device=self.device, dtype=self.dtype)
        model.eval()
        return model
    
    def get_transforms(self) -> torchvision.transforms.Compose:
        return self.model.default_transform
    
    def get_augmentations(self) -> torchvision.transforms.Compose:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(90),
                torchvision.transforms.RandomAffine(0, scale=(0.9, 1.1)),
                torchvision.transforms.RandomAffine(0, shear=5),
                # torchvision.transforms.RandomInvert(), 
                # torchvision.transforms.RandomAutocontrast(),
                # torchvision.transforms.RandomErasing(),
            ]
        )
    
    def post_process_batch(self, output : torch.Tensor) -> dict:
        this_batch_len = len(output[0])
        predictions = [torch.nn.functional.softmax(output, dim=1) for output in output]
        categories = [prediction.argmax(dim=1).tolist() for prediction in predictions]
        labels = [[self.model.class_handles["idx_to_class"][level][c] for c in cat] for level, cat in enumerate(categories)]
        scores = [prediction.max(dim=1).values.tolist() for prediction in predictions]

        # Choose the first prediction that is above the threshold
        category = []
        label = []
        score = []

        threshold = 0
        for item in range(len(categories[0])):
            for level in range(self.n_levels):
                if scores[level][item] > threshold:
                    category.append(categories[level][item])
                    label.append(labels[level][item])
                    score.append(scores[level][item])
                    break
            else:
                category.append(-1)
                label.append("Unknown")
                score.append(0.0)

        label = [" | ".join([KEY_TO_NAME[lab] if label != 'Unknown' else 'Unknown' for lab in label]) for i in range(this_batch_len)]

        data = dict()
        data["Predicted"] = label
        data["Confidence"] = score
        data["Unknown"] = [0.0 if c != -1 else 1.0 for c in category]
        for level in range(self.n_levels):
            this_level_predictions = predictions[level].T.tolist()
            data.update({
                KEY_TO_NAME[self.model.class_handles["idx_to_class"][level][c]] : this_level_predictions[c]  for c in range(len(predictions[level][0]))
            })

        return {
            "class" : label,
            "score" : score,
            "data" : data
        }

    def predict(self, images):
        # print(type(images), type(images[0]), type(images[0][0]))
        if not isinstance(images, (list, tuple)):
            images = [images] 
        # ## DEBUG
        # plot_images(images)
        outputs = [torch.zeros((len(images), self.model.class_handles["n_classes"][level]), device=self.device, dtype=self.dtype) for level in range(self.n_levels)]
        for i in range(0, len(images), self.batch_size):
            batch = parse_image(images[i:i+self.batch_size], device=self.device)
            batch = torch.stack([self.transforms(image) for image in batch]).to(self.device, self.dtype)
            with torch.no_grad():
                if self.tta_enabled:
                    tta_output = [self.model(self.augmentations(batch)) for _ in range(10)]
                    output = [torch.stack([to[level] for to in tta_output]).logsumexp(dim=0) for level in range(self.n_levels)]
                else:
                    output = self.model(batch)
            for level in range(self.n_levels):
                outputs[level][i:i+self.batch_size] = output[level]
        
        return self.post_process_batch(outputs)

def gen_level_idx(vocab, taxa):
    """Return a dictionary with "species":[False, True, ....]
    """
    # taxa list of list
    if type(taxa)==dict:
        taxa_lofl = [list(t.keys()) for t in taxa.values()]
    elif type(taxa)==list:
        taxa_lofl = [list(t.keys()) for t in taxa]
    else:
        raise NotImplementedError(f"Unknown type for taxa {type(taxa)}. Please implement it")
    indices = np.ones(len(vocab), dtype=int)*-1
    for i,v in enumerate(vocab):
        for j in range(len(taxa_lofl)):
            if v in taxa_lofl[j]:
                indices[i] = j
                break 
    if -1 in indices:
       print(f"[Warning] Missing values in taxa dictionary: {len(np.array(vocab)[np.array(indices)==-1])}.")
    return indices

def get_last_true(l, reverse=False):
    """Browse boolean list to search for the last True in list. Can be done backwards.
    Return list index. Return -1 if only False in list.
    """
    if reverse: 
        i = len(l)-1
        while i >= 0 and l[i]:
            i -= 1 
        return -1 if i==len(l)-1 else i+1
    else:
        i = 0
        while i < len(l) and l[i]:
            i += 1 
        return i-1

class FastaiClassifier(BaseClassifier):
    def __init__(self, category_map : str, device : Any, *args, **kwargs):
        with open(category_map, "r") as f:
            self.category_map = json.load(f)
        self.class_to_idx = self.category_map["class_to_idx"]
        self.idx_to_class = self.category_map["idx_to_class"]
        super().__init__(device=device, *args, **kwargs)
        
    def get_model(self):
        model = load_learner(self.weights, cpu=True)
        model.model.eval().to(self.device)
        return model
    
    def get_transforms(self) -> torchvision.transforms.Compose:
        return
    
    def get_augmentations(self) -> torchvision.transforms.Compose:
        return
    
    def post_process_batch(self, output : Any) -> dict:
        # Apply activation
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        all_scores = [sigmoid(p) for p in output] # (5,n,ni)
        num_taxa = len(all_scores)
        num_images = len(all_scores[0])

        # Get the best score per taxa for all images
        best_scores = np.array([np.max(a, axis=1) for a in all_scores]) # (5,n)
        

        # Get the corresponding best class per taxa for all images
        ## Generate indices for taxa levels
        vocab = np.array(self.model.dls.vocab)
        indices = gen_level_idx(vocab=vocab, taxa=self.class_to_idx)
        indices = torch.from_numpy(indices)
        ## Generate a vocab for each taxa
        sorted_vocab = [vocab[indices==i] for i in range(num_taxa)]
        ## Best classes
        best_classes = [sorted_vocab[i][np.argmax(a, axis=1)] for i, a in enumerate(all_scores)] # (5,n)

        # Get the scores and classes
        best_scores_taxa = [get_last_true(best_scores[:, i] > 0.5, reverse=True) for i in range(num_images)] # (n)
        scores = [best_scores[i][j] for j, i in enumerate(best_scores_taxa)]
        classes = [best_classes[i][j] for j, i in enumerate(best_scores_taxa)]
        
        # Get data
        flatten_sorted_vocab = np.concatenate(sorted_vocab) # (5*ni)
        flatten_all_scores = [] # (n, 5*ni)
        for i in range(num_images):
            flatten_all_scores += [np.concatenate([s[i,:] for s in all_scores])] 
        flatten_all_scores = np.array(flatten_all_scores).T # (5*ni, n)
        data = {k:v for k,v in zip(flatten_sorted_vocab, flatten_all_scores)}

        data["Predicted"] = classes
        data["Confidence"] = scores

        return {
            "class" : classes,
            "score" : scores,
            "data" : data
        }

    def predict(self, images : Optional[Union[str, bytes, np.ndarray, List[Optional[Union[str, bytes, np.ndarray]]], Tuple[Optional[Union[str, bytes, np.ndarray]]]]]) -> dict:
        if not isinstance(images, (list, tuple)):
            images = [images] 
        
        # Creating test DataLoader
        test_dl = self.model.dls.test_dl(images)

        # Generate indices for taxa levels
        indices = gen_level_idx(vocab=list(self.model.dls.vocab), taxa=self.class_to_idx)
        indices = torch.from_numpy(indices)

        # Running predictions
        num_levels = len(self.class_to_idx)
        preds = [[] for _ in range(num_levels)]

        with torch.no_grad():
            for batch in test_dl:
                logits = self.model.model(batch[0].to(self.device)).detach()
                for i in range(num_levels):
                    preds[i].append(logits[:, (indices == i).nonzero().flatten()].cpu().numpy())
        
        # Concatenating predictions
        for i in range(num_levels):
            preds[i] = np.concatenate(preds[i])

        return self.post_process_batch(preds)

def main(args : dict):

    output_stream = sys.stdout # io.StringIO() if not args["output"] else sys.stdout
    # Supress all prints here, to ensure that the output is only the JSON string if the output is not specified
    with contextlib.redirect_stdout(output_stream):
        # Get the defaults
        class_dict, weights, device, dtype = get_defaults()

        input_images = get_images(args["input"])

        # Update the parameters
        if "class_dict" in args and args["class_dict"] is not None:
            class_dict = args["class_dict"]

        if "weights" in args and args["weights"] is not None:
            weights = args["weights"]

        if "device" in args and args["device"] is not None:
            device = torch.device(args["device"])

        # Create the model
        # model = Resnet50Classifier(weights=weights, category_map=class_dict, device=device, test_time_augmentation=True)
        model = FastaiClassifier(weights=weights, category_map=class_dict, device=device)
        model = HierarchicalClassifier(weights=weights, device=device, test_time_augmentation=False)

        # Run the model
        output = dict2csv(model.predict(input_images)["data"])

    # Save the output
    if "output" in args and args["output"] is not None:
        if os.path.isdir(args["output"]):
            raise ValueError("Output cannot be a directory.")
        with open(args["output"], "w") as f:
            f.write(output)
    else:
        print(output)

    return output

if __name__ == "__main__":
    import argparse

    args_parser = argparse.ArgumentParser(description="Classify images of bugs.")
    args_parser.add_argument('-i', '--input', type=str, nargs="+", help='Path(s), director(y/ies) or glob(s) to such. If a .txt then it each line should correspond to the former. Outputs will be saved in the output directory.', required=True)
    args_parser.add_argument("-o", "--output", type=str, help="The output CSV file. If not specified it will be dumped in stdout.", required=False)
    args_parser.add_argument("--weights", type=str, help="The path to the weights file. Default is 'classification_weights.pth'.")
    args_parser.add_argument("--class_dict", type=str, help="The path to the class_dict CSV. Default is 'class_dict.csv'.")
    args_parser.add_argument("--device", type=str, help="The device to use defaults to 'cuda:0' if available else 'cpu'")
    args = args_parser.parse_args()

    main(vars(args))
