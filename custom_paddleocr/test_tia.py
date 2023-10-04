import cv2

from ppocr.data.imaug.text_image_aug import tia_distort

src = cv2.imread('doc/imgs_words_en/word_10.png')
tia_aug = tia_distort(src)
cv2.imwrite("./tia_distort_1.png", tia_aug)
