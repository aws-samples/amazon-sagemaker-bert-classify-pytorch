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
import os
from unittest import TestCase

from dbpedia_dataset import DbpediaDataset


class TestDbpediaDataset(TestCase):
    def test___getitem__(self):
        file = os.path.join(os.path.dirname(__file__), "sample_dbpedia.csv")
        sut = DbpediaDataset(file)
        expected_y = 1
        expected_x = " Supinfocom (école SUPérieure d'INFOrmatique de COMmunication roughly University of Communication Science) is a computer graphics university with campuses in Valenciennes Arles (France) and Pune (India).Founded in 1988 in Valenciennes the school offers a five-year course leading to a diploma of digital direction (certified Level I)."

        # Act
        actual_x, actual_y = sut.__getitem__(0)

        # Assert
        self.assertEqual(expected_y, actual_y)
        self.assertEqual(expected_x, actual_x)
