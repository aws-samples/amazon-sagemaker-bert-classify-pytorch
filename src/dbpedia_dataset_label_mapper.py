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
class DbpediaLabelMapper:

    def __init__(self, classes_file):
        with open(classes_file, "r") as f:
            self._raw_labels = f.readlines()

        self._map = {v: i for i, v in enumerate(self._raw_labels)}

        self._reverse_map = {i: v for i, v in enumerate(self._raw_labels)}

    def map(self, item):
        return self._map[item]

    def reverse_map(self, item):
        return self._reverse_map[item]

    @property
    def num_classes(self):
        return len(self._reverse_map)

    @property
    def pos_label(self):
        return self.reverse_map(0)

    @property
    def pos_label_index(self):
        return self.map(self.pos_label)
