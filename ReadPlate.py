import time
import cv2
import numpy as np


# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def find_plate(image, net):
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Thực hiện xác định bằng HOG và SVM
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # print(len(boxes))
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]+5
        y = box[1]+5
        w = box[2]-10
        h = box[3]-10

    #cat nua phan tren bien so
    LpImg = []
    pts1 = np.float32([[x, y], [x + w, y], [x, (y + h + y) / 2 ], [x + w, (y + h + y) / 2]])
    pts2 = np.float32([[0, 0], [280, 0], [0, 100], [280, 100]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (280, 100), borderValue=0)
    LpImg.append(dst)
    # cv2.imshow("dst",dst)
    # cv2.waitKey()
    #cat nua phan duoi bien so
    pts1 = np.float32([[x, (y + h + y) / 2 ], [x + w, (y + h + y) / 2 ], [x, y + h], [x + w, y + h]])
    pts2 = np.float32([[0, 0], [280, 0], [0, 100], [280, 100]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (280, 100), borderValue=0)
    LpImg.append(dst)
    # cv2.imshow("dst", dst)
    # cv2.waitKey()
    return LpImg


def read_plate(image, net):
    LpImg = find_plate(image, net)
    plate_info = ""
    for i in range(len(LpImg)):

        roi = LpImg[i]
        # Chuyen anh bien so ve gray
        gray = cv2.cvtColor(LpImg[i], cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        cv2.waitKey()
        # Ap dung threshold de phan tach so va nen
        binary = cv2.threshold(gray, 127, 255,
                               cv2.THRESH_BINARY_INV)[1]
        cv2.imshow('binary', binary)
        cv2.waitKey()
        # plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
        # Segment kí tự
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

        # kernel = np.ones((2,2),np.uint8)
        # erosion = cv2.erode(binary,kernel,iterations = 1)

        # plt.imshow(cv2.cvtColor(thre_mor, cv2.COLOR_BGR2RGB))

        # plt.imshow(cv2.cvtColor(thre_mor, cv2.COLOR_BGR2RGB))
        cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1.5 <= ratio <= 3.5:  # Chon cac contour dam bao ve ratio w/h
                if 0.95 > h / roi.shape[0] >= 0.45:  # Chon cac contour cao tu 60% bien so tro len

                    # Ve khung chu nhat quanh so
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Tach so va predict
                    curr_num = thre_mor[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num, dtype=np.float32)
                    curr_num = curr_num.reshape(-1, digit_w * digit_h)

                    # Dua vao model SVM
                    result = model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    if result <= 9:  # Neu la so thi hien thi luon
                        result = str(result)
                    else:  # Neu la chu thi chuyen bang ASCII
                        result = chr(result)

                    plate_info += result
        # plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        # cv2.imshow("roi", roi)
        # cv2.waitKey()
        if i == 0:
            plate_info += '-'
    return plate_info


image = cv2.imread('Test02.jpg')

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet('yolo-tinyv4-obj_best.weights', 'yolo-tinyv4-obj.cfg')
digit_w = 30  # Kich thuoc ki tu
digit_h = 60  # Kich thuoc ki tu
model_svm = cv2.ml.SVM_load('svm.xml')
start = time.time()
num_plate = read_plate(image, net)
print(num_plate)
end = time.time()
print("YOLO Execution time: " + str(end - start))

cv2.waitKey()
cv2.destroyAllWindows()
