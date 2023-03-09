import os

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

new_lines = []
with open('./test_data/v4_test_rec_data/v4_lsvt5k_mtwi5k_200_310_800.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if len(line.strip().split('\t')) == 2:
            new_lines.append(line)
with open('./test_data/v4_test_rec_data/v4_lsvt5k_mtwi5k_200_310_800_valid.txt', 'a+') as fd:
    fd.writelines(new_lines)