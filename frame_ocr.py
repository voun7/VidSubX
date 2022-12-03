import cv2 as cv
import numpy as np


def text_detected(frame: np.array) -> bool:
    model = cv.dnn.TextDetectionModel_EAST("models/frozen_east_text_detection.pb")

    conf_threshold = 0.5
    nms_threshold = 0.4
    model.setConfidenceThreshold(conf_threshold)
    model.setNMSThreshold(nms_threshold)

    det_scale = 1.0
    det_input_size = (320, 320)
    det_mean = (123.68, 116.78, 103.94)
    swap_rb = True
    model.setInputParams(det_scale, det_input_size, det_mean, swap_rb)

    result = model.detect(frame)
    detections = result[0]
    confidences = result[1]

    # # Visualization
    # print(result)
    # new_img = cv.polylines(frame, detections, True, (0, 255, 0), 2)
    # cv.imshow("Text Detection", new_img)
    # cv.waitKey(0)

    if len(detections) and len(confidences):
        print("Text detected")
        return True
    else:
        print("Text not detected")
        return False


def text_recognizer(image: str) -> None:
    print(type(image))
    rgb = cv.IMREAD_COLOR
    image = cv.imread(image, rgb)

    model = cv.dnn.TextRecognitionModel("models/crnn_cs_CN.onnx")
    model.setDecodeType("CTC-greedy")

    voc_file = "models/alphabet_3944.txt"
    vocabulary = np.loadtxt(voc_file, dtype="str", encoding="utf8")

    model.setVocabulary(vocabulary)

    scale = 1.0 / 127.5
    mean = (127.5, 127.5, 127.5)
    input_size = (100, 32)

    model.setInputParams(scale, input_size, mean)

    result = model.recognize(image)
    print(result)


image = "output/I Can Copy Talents/frames/3300.0.jpg"
# image = "tests/evaluation_data_rec/test/2002_2.png"
frame = cv.imread(image)

text_detected(frame)
text_recognizer(image)

# cv.imshow("image", frame)
# cv.waitKey(0)
