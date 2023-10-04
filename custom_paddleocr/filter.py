import os
import random


def rm_single_word_img():
    img_names = []
    with open('../2048_label.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip().split('\t')
            img_name = info[0].split('/')[-1]
            label = info[1]
            if len(label) == 1:
                img_names.append(img_name)
    return img_names


img_names = rm_single_word_img()


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


all_pics = []
listdir("/paddle/github/StyleTextV2/data_corpus/ocr_data/2048", all_pics)

all_pics_valid = []
for pic in all_pics:
    if pic.split('/')[-1] not in img_names:
        all_pics_valid.append(pic)

import string
from zhon.hanzi import punctuation


def rm_punc(label):
    for c in string.punctuation:
        label = label.replace(c, '')
    for c in punctuation:
        label = label.replace(c, '')
    label = label.replace(' ', '')
    return label


all_corpus = []
with open("all_ocrv3_corpus.txt", "r") as f:
    for line in f.readlines():
        img_path, label = line.strip().split("\t")
        label = rm_punc(label)
        all_corpus.append(label)
with open("ocr_v4_pairs_v2.txt", "w") as f:
    for corpus in all_corpus:
        style_imgs = random.sample(all_pics_valid, 20)
        for style in style_imgs:
            f.write(f"{style}\t{corpus}\n")

valid_labels = []
with open('./corpus_select/ocr_v4_pairs_v2.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        info = line.strip().split("\t")
        if len(info) < 2:
            print(line)
            continue
        img_path = os.path.join('/paddle/github/StyleTextV2/data_corpus/ocr_data', info[0])
        label = info[1]
        if len(label) > 1 and len(label) <= 25 and label != '' and os.path.exists(img_path):
            valid_labels.append(line)
with open('./corpus_select/ocr_v4_pairs_v2_valid.txt', 'a+') as fd:
    fd.writelines(valid_labels)
