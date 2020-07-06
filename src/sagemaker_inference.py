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
import torch

"""
This is the sagemaker inference entry script
"""
CSV_CONTENT_TYPE = 'application/csv'

def model_fn(model_dir):
    model = torch.load(model_dir)
    device = get_device()
    model.to(device=device)
    return model, tokensier


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def input_fn(input, content_type):
    tensor = data
    if content_type == CSV_CONTENT_TYPE:
        tensor()
    else:
        raise ValueError(
            "Content type {} not supported. The supported type is {}".format(content_type, CSV_CONTENT_TYPE))
    return input


def predict_fn(input, model):
    device = get_device()
    input = input.to(device=device)
    return model(input)


def output_fn(output, content_type):
    return output
