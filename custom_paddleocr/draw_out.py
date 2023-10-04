def draw_badcase_v1(html_path="rec_result2.html",
                    label_file="crop_res_file.txt",
                    result_1="res_rec/200_eval.txt",
                    result_2="res_rec/base_v5.txt",
                    img_dir="http://127.0.0.1:8900/PaddleOCR/"):
    err_cnt = 0
    with open(html_path, 'w') as html:
        html.write('<html>\n<body>\n')
        html.write('<table border="1">\n')
        html.write("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />")
        label_file = open(label_file, 'r')
        label_dict = {}
        for i_line in label_file.readlines():
            line = i_line.strip().split('\t')
            # print(line)
            try:
                name_file = line[0]
                label = line[1]
                new_dict = {name_file.split('/')[-1]: label}
                # new_dict = {"./310_crop/"+name_file : label}
                # new_dict = {"./"+name_file : label}
                # new_dict = {"./train_data/real_data/"+name_file : label}
                label_dict.update(new_dict)
            except:
                print("line:", line)
                pass
        f_crnn = open(result_1, 'r')
        f_vit = open(result_2, 'r')

        for i_vit, i_crnn in zip(f_vit.readlines(), f_crnn.readlines()):
            line_vit = i_vit.strip().split('\t')
            # print(line_vit)
            if len(line_vit) < 2:
                continue
            name_file_vit = line_vit[0].split('/')[-1]
            pred_vit = line_vit[1]
            line_crnn = i_crnn.strip().split('\t')
            if len(line_crnn) < 2:
                continue
            print("line_crnn:", line_crnn)
            name_file_crnn = line_crnn[0].split('/')[-1]
            pred_crnn = line_crnn[1]
            # score_crnn = float(line_crnn[2])
            # if len(line_crnn) < 3:
            #     continue
            # img_path = img_dir + name_file_crnn[2:]
            # det_crop
            img_path = img_dir + '/e2e_v4_4/crop/' + name_file_crnn
            try:
                # print(name_file_crnn)
                gt = label_dict[name_file_crnn]

                # print(gt)
            except:
                continue

            # if (pred_vit != gt):
            if pred_crnn == gt and pred_vit != gt:
                if True:
                    html.write("<tr>\n")
                    v2_err = "<td>" + name_file_crnn + "_result1:"
                    for i in range(min(len(pred_crnn), len(gt))):
                        if pred_crnn[i] != gt[i]:
                            v2_err += '<span>%s</span>' % pred_crnn[i]
                        else:
                            v2_err += pred_crnn[i]
                    v2_err += pred_crnn[min(len(pred_crnn), len(gt)):]
                    v2_err += '</td>\n'
                    html.write(v2_err)

                    v3_err = "<td>result2:"
                    for i in range(min(len(pred_vit), len(gt))):
                        if pred_vit[i] != gt[i]:
                            v3_err += '<span>%s</span>' % pred_vit[i]
                        else:
                            v3_err += pred_vit[i]
                    v3_err += pred_vit[min(len(pred_vit), len(gt)):]
                    v3_err += '</td>\n'
                    html.write(v3_err)

                    gt_result = "<td>gt:{}</td>\n".format(gt)
                    html.write(gt_result)

                    html.write('<td><img src="%s"></td>' % (img_path))
                    html.write("</tr>\n")
                    # err_cnt += 1
        html.write('<style>\n')
        html.write('span {\n')
        html.write('    color: red;\n')
        html.write('}\n')
        html.write('</style>\n')
        html.write('</table>\n')
        html.write('</html>\n</body>\n')
    print("ok")
    return


if __name__ == "__main__":
    html_path = "base_wrong_styletext_right.html"
    label_file = "./e2e_v4_4/label.txt"
    result_2 = "./rec_400w_f1_e2e.txt"
    result_1 = "./rec_400w_f1_styletext.txt"
    img_dir = "http://10.21.226.176:8097/kaitao/PaddleOCR/"

    draw_badcase_v1(html_path, label_file, result_1, result_2, img_dir)
