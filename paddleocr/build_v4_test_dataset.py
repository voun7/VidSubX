import numpy as np
import json
import os
import cv2
import sys
import glob
from PIL import Image, ImageDraw, ImageFont

import math
import glob

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32  
        elif (inside_code >= 65281 and inside_code <= 65374): 
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    # random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


class DetLabelEncode(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool_)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes


def get_image_from_txt(root_dir, datalines):
    # if not os.path.exists(label_file):
    #     raise ValueError(f"not found {label_file}")

    label_encode_func = DetLabelEncode()
    
    # datalines = open(label_file, "r").readlines()
    datas = []
    for line in datalines:
        img_name, label = line.strip().split("\t")
        # print(label)
        img_path = os.path.join(root_dir, img_name)
        if not os.path.exists(img_path):
            print(f"not found {img_path}")
        else:
            data = {'img_file': img_path, 'label': label}
            data = label_encode_func(data)
            if data is None:
                continue
            datas.append(data)
    return datas


def draw_html(images_dir):

    img_list = sorted(os.listdir(images_dir))
    print(len(img_list))

    html_path = "chinese_new_dataset200+100+200+.html"
    with open(html_path, 'w') as html:
        html.write('<html>\n<body>\n')
        html.write('<table border="1">\n')
        html.write("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />")
        
        html.write("<tr>\n")
        # html.write(f'<td> \n GT')

        for img_name in img_list:

            img_path = f"http://10.21.226.176:8097/kaitao/PaddleOCR/test_data/visual/{img_name}"

            html.write(f'<td>{img_name}\n<img src="{img_path}" width=640></td>' )

            # html.write('<td>PPOCRV2\n<img src="%s" width=640></td>' % (base_2))
            # html.write('<td>r50_cml_fpnse_recall887 + PPOCRV2\n<img src="%s" width=640></td>' % (base_3))
            # html.write('<td>r50_cml_fpnse_recall887 + ctc782\n<img src="%s" width=640></td>' % (base_4))

            html.write("</tr>\n")
        
        html.write('<style>\n')
        html.write('span {\n')
        html.write('    color: red;\n')
        html.write('}\n')
        html.write('</style>\n')
        html.write('</table>\n')
        html.write('</html>\n</body>\n')
        print("ok")


def visualization(data_list, save_res_path="./visual/"):
    datas = data_list
    if not os.path.exists(save_res_path):
        os.makedirs(save_res_path)

    for i in range(len(datas)):
        img_path = datas[i]['img_file']
        polys = datas[i]['polys']

        dataset_name = os.path.basename(img_path)

        img_name = os.path.basename(img_path)

        img = cv2.imread(img_path)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = polys.tolist()
        boxes = [tuple(map(tuple, np.array(box))) for box in boxes]
        boxes = polys
        txts = datas[i]['texts']
        scores = None
        draw_img = draw_ocr_box_txt(image, boxes, txts, scores, 0., "doc/fonts/simfang.ttf")
        cv2.imwrite(os.path.join(save_res_path, f"{i}_" + os.path.basename(img_path)),
                            draw_img[:, :, ::-1])
    print(f"Image saved in {save_res_path}")



if __name__ == "__main__":

    import random
    import random
    random.seed(2023)   # 修改 

    test_dir = "test_data/test_set"
    test_310 = "test_data/test_set/Label_refine_310_V2.txt"
    test_200 = "test_data/test_set/Label_refine_v8.txt"

    ch_800 = "test_data/seperate_800/image_2/"
    test_800 = "test_data/seperate_800/label.txt"

    # screen 
    screen_20_dir = "test_data"
    screen_20_label = 'test_data/screen_20/label.txt'

    # xfun
    xfun_20_dir = "test_data"
    xfun_20_label = 'test_data/xfunc_zh_20/label.txt'

    # 待补充范例数据，共50张，包含卡证，车牌，数码管，电表等

    data_200 = open(test_200, "r").readlines()
    collect_200 = random.sample(data_200, 100)

    data_310 = open(test_310, "r").readlines()
    collect_310 = random.sample(data_310, 150)

    data_800 = open(test_800, "r").readlines()
    collect_800 = random.sample(data_800, 250)

    screen_20 = open(screen_20_label, "r").readlines()

    xfun_20 = open(xfun_20_label, "r").readlines()

    
    #############################
    ### build v4 test dataset ###
    #############################
    datas = get_image_from_txt(test_dir, collect_200)
    datas += get_image_from_txt(test_dir, collect_310)
    datas += get_image_from_txt(ch_800, collect_800)
    datas += get_image_from_txt(screen_20_dir, screen_20)
    datas += get_image_from_txt(xfun_20_dir, xfun_20)

    v4_test_save_dir = "test_data/v4_test_dataset/"    # the directory of new dataset saved
    if not os.path.exists(v4_test_save_dir):
        os.makedirs(v4_test_save_dir)
    with open(os.path.join(v4_test_save_dir, "label.txt"), "w") as f:
        f.writelines(collect_200 + collect_310 + collect_800 + screen_20 + xfun_20)
        f.close()

    ## copy image to v4_test_save_dir
    for idx, data in enumerate(datas):
        img_path = data['img_file']
        save_name = os.path.join(v4_test_save_dir, os.path.basename(img_path))
        os.system(f"cp {img_path} {save_name}")

    assert len(os.listdir(v4_test_save_dir))-1 == len(datas), f"Boom! Error, {len(os.listdir(v4_test_save_dir))} vs {len(datas)}"
    print(f"DEBUG, the test image has saved in {v4_test_save_dir}")

    visualization(datas, "test_data/visual/")

    draw_html("test_data/visual/")