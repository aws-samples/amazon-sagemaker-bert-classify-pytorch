# ***************************************************************************************
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.                    *
#                                                                                       *
# Permission is hereby granted, free of charge, to any person obtaining a copy of this  *
# software and associated documentation files (the "Software"), to deal in the Software *
# without restriction, including without limitation the rights to use, copy, modify,    *
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to    *
# permit persons to whom the Software is furnished to do so.                            *
#                                                                                       *
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,   *
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A         *
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT    *
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION     *
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE        *
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                *
# ***************************************************************************************

import glob
import json
import os
import pickle

import torch

"""
This is the sagemaker inference entry script
"""
CSV_CONTENT_TYPE = 'text/csv'
JSON_CONTENT_TYPE = 'text/json'


def model_fn(model_dir):
    # Load model
    model_files = list(glob.glob("{}/*.pt".format(model_dir)))
    error_msg = "Expected exactly 1 model file (match pattern *.pt)in dir {}, but instead found {} files. Found.. {}".format(
        model_dir, len(model_files), ",".join(model_files))
    assert len(model_files) == 1, error_msg

    model_file = model_files[0]
    device = get_device()
    model = torch.load(model_file, map_location=torch.device(device))

    # Load label mapper
    label_mapper_pickle_file = os.path.join(model_dir, "label_mapper.pkl")
    with open(label_mapper_pickle_file, "rb") as f:
        label_mapper = pickle.load(f)

    # Load preprocessor
    preprocessor_pickle_file = os.path.join(model_dir, "preprocessor.pkl")
    with open(preprocessor_pickle_file, "rb") as f:
        preprocessor_mapper = pickle.load(f)

    return preprocessor_mapper, model, label_mapper


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def input_fn(input, content_type):
    if content_type == CSV_CONTENT_TYPE:
        records = input.split("\n")
        return records
    else:
        raise ValueError(
            "Content type {} not supported. The supported type is {}".format(content_type, CSV_CONTENT_TYPE))


def preprocess(input, preprocessor):
    result = [preprocessor(i).unsqueeze(dim=0) for i in input]
    result = torch.cat(result)
    return result


def predict_fn(input, model_artifacts):
    preprocessor, model, label_mapper = model_artifacts

    # Pre-process
    input_tensor = preprocess(input, preprocessor)

    # Copy input to gpu if available
    device = get_device()
    input_tensor = input_tensor.to(device=device)

    # Invoke
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)[0]
        # Convert to probabilities
        softmax = torch.nn.Softmax()
        output_tensor = softmax(output_tensor)

    # Return the class with the highest prob and the corresponding prob
    prob, class_indices = torch.max(output_tensor, dim=1)
    classes = [label_mapper.reverse_map(i.item()) for i in class_indices]
    result = []
    for c, p in zip(classes, prob):
        result.append({c: p.item()})

    return result


def output_fn(output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        prediction = json.dumps(output)
        return prediction, accept
    else:
        raise ValueError(
            "Content type {} not supported. The only types supported are {}".format(accept, JSON_CONTENT_TYPE))
