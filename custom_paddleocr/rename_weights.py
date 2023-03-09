import paddle
import numpy as np


# model_dict = paddle.load('pretrained_model/LF_1_ocr.pdparams')
# model_keys = model_dict.keys()
# out_dict = {}
# for key in model_keys:
#     weight = model_dict[key]
#     if 'backbone.conv1_new' in key:
#         key = key.replace('_new', '')
#     out_dict[key] = weight
# paddle.save(out_dict, 'pretrained_model/LF_1_ocr_v1.pdparams')
model_dict = paddle.load('rec_r45_visionlan_train/best_accuracy.pdparams')
for key in model_dict.keys():
    weight = model_dict[key]
    import pdb 
    pdb.set_trace()
    print()