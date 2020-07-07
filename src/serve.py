# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
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

    # Preprocess
    input_tensor = preprocess(input, preprocessor)

    # Copy input to gpu if available
    device = get_device()
    input_tensor = input_tensor.to(device=device)

    # Invoke
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)[0]
        # Convert to probablties
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
            "Content type {} not supported. The only types supprted are {}".format(accept, JSON_CONTENT_TYPE))
