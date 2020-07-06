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

from dbpedia_dataset_label_mapper import DbpediaLabelMapper
from preprocessor_bert_tokeniser import PreprocessorBertTokeniser

"""
This is the sagemaker inference entry script
"""
CSV_CONTENT_TYPE = 'application/csv'


def model_fn(model_dir):
    model = torch.load(model_dir)
    device = get_device()
    model.to(device=device)
    return model


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


def preprocess(input):
    tokeniser = None
    p = PreprocessorBertTokeniser(max_feature_len=512, tokeniser=tokeniser)
    result = [p(i) for i in input]
    return result


def predict_fn(input, model):
    # TODO: convert to tensor
    input_tensor = preprocess(input)
    device = get_device()
    input_tensor = input_tensor.to(device=device)
    return model(input_tensor)


def output_fn(output, content_type):
    classes_file = None
    prob, class_indices = torch.max(output, dim=1)
    label_mapper = DbpediaLabelMapper(classes_file=classes_file)
    classes = [label_mapper.reverse_map(i) for i in class_indices]
    formatted_result = []
    for c, p in zip(classes, prob):
        formatted_result.append((c, p))
    return output
