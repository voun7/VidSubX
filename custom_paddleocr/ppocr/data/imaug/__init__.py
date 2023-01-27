# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from custom_paddleocr.ppocr.data.imaug.ColorJitter import ColorJitter
from custom_paddleocr.ppocr.data.imaug.copy_paste import CopyPaste
from custom_paddleocr.ppocr.data.imaug.ct_process import *
from custom_paddleocr.ppocr.data.imaug.drrg_targets import DRRGTargets
from custom_paddleocr.ppocr.data.imaug.east_process import *
from custom_paddleocr.ppocr.data.imaug.fce_aug import *
from custom_paddleocr.ppocr.data.imaug.fce_targets import FCENetTargets
from custom_paddleocr.ppocr.data.imaug.iaa_augment import IaaAugment
from custom_paddleocr.ppocr.data.imaug.label_ops import *
from custom_paddleocr.ppocr.data.imaug.make_border_map import MakeBorderMap
from custom_paddleocr.ppocr.data.imaug.make_pse_gt import MakePseGt
from custom_paddleocr.ppocr.data.imaug.make_shrink_map import MakeShrinkMap
from custom_paddleocr.ppocr.data.imaug.operators import *
from custom_paddleocr.ppocr.data.imaug.pg_process import *
from custom_paddleocr.ppocr.data.imaug.randaugment import RandAugment
from custom_paddleocr.ppocr.data.imaug.random_crop_data import EastRandomCropData, RandomCropImgMask
from custom_paddleocr.ppocr.data.imaug.rec_img_aug import BaseDataAugmentation, RecAug, RecConAug, RecResizeImg, \
    ClsResizeImg, SRNRecResizeImg, GrayRecResizeImg, SARRecResizeImg, PRENResizeImg, ABINetRecResizeImg, \
    SVTRRecResizeImg, ABINetRecAug, VLRecResizeImg, SPINRecResizeImg, RobustScannerRecResizeImg, RFLRecResizeImg
from custom_paddleocr.ppocr.data.imaug.sast_process import *
from custom_paddleocr.ppocr.data.imaug.ssl_img_aug import SSLRotateResize
from custom_paddleocr.ppocr.data.imaug.table_ops import *
from custom_paddleocr.ppocr.data.imaug.vqa import *


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops
