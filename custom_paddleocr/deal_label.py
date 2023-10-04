import json
import os
import random
import string

from zhon.hanzi import punctuation


# new_lines = []
# with open('test_data/label.txt', "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         info = line.strip().split("\t")
#         img_path = info[0]
#         label = info[1]
#         img_path = os.path.basename(img_path)
#         # img_path = 'v4_test_dataset/' + img_path
#         new_line = img_path + '\t' + label + '\n'
#         new_lines.append(new_line)
# with open('test_data/test_label.txt', 'a+') as fd:
#     fd.writelines(new_lines)
# new_lines = []
# with open('./test_data/v4_test_rec_data/v4_lsvt5k_mtwi5k_200_310_800.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         if len(line.strip().split('\t')) == 2:
#             new_lines.append(line)
# with open('./test_data/v4_test_rec_data/v4_lsvt5k_mtwi5k_200_310_800_valid.txt', 'a+') as fd:
#     fd.writelines(new_lines)
# new_lines = []
# with open('/paddle/data/ocr_all/data_all.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         info = line.strip().split('\t')
#         img_path = info[2].replace('/output_imgs/', 'style_fg_imgs/')
#         label = info[0]
#         new_line = img_path + '\t' + label + '\n'
#         # import pdb
#         # pdb.set_trace()
#         new_lines.append(new_line)
# with open('/paddle/data/ocr_all/style_fg_label.txt', 'a+') as fd:
#     fd.writelines(new_lines)


def rm_punc(label):
    for c in string.punctuation:
        label = label.replace(c, '')
    for c in punctuation:
        label = label.replace(c, '')
    label = label.replace(' ', '')
    return label


def _try_parse_filename_list(file_name):
    # multiple images -> one gt label
    if len(file_name) > 0 and file_name[0] == "[":
        try:
            info = json.loads(file_name)
            file_name = random.choice(info)
        except:
            pass
    return file_name


## get styletext image and corpus
save_dir = 'style_data'
# os.makedirs(save_dir, exist_ok=True)

# 26w_result/InvoiceDatasets/rec_train_val_list_new_26w_scor90.txt
# 26w_result/kie_img/kie_new_26w_scor90_exa.txt
# 26w_result/card_number_img/card_number_new_26w_scor90.txt
# 26w_result/hospital_img/hospital_new_26w_scor90.txt
# 26w_result/digital_img/digital_label_new_26w_scor90.txt

# label_file = '26w_result/kie_img/kie_new_26w_scor90_exa.txt'
# root_dir = '/paddle/data/ocr_all/'
# sub_dir = os.path.join(save_dir, 'kie')
# os.makedirs(sub_dir, exist_ok=True)
# img_dir = os.path.join(sub_dir, 'images')
# os.makedirs(img_dir, exist_ok=True)

# corpus = []
# with open(label_file, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         info = line.strip().split('\t')
#         img_name = _try_parse_filename_list(info[0])
#         img_path = os.path.join(root_dir, img_name)
#         label = rm_punc(info[1])

#         if len(label) > 1 and len(label) <= 25 and label != '' and os.path.exists(img_path):
#             dst_path = os.path.join(img_dir, os.path.basename(img_name))
#             shutil.copy(img_path, dst_path)
#             corpus.append(label + '\n')
# with open(os.path.join(sub_dir, 'corpus.txt'), 'a+') as fd:
#     fd.writelines(corpus)

for name in ['InvoiceDatasets', 'kie', 'card_number', 'digital', 'hospital']:
    img_lists = []
    path = os.path.join(save_dir, name, 'images')
    for image in os.listdir(path):
        img_list = os.path.join(path, image).replace(save_dir + '/', '')
        img_lists.append(img_list + '\n')
    with open(os.path.join(save_dir, name, 'image_list.txt'), 'a+') as fd:
        fd.writelines(img_lists)
