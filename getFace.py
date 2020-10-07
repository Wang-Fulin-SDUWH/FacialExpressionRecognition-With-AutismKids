import cv2
import sys
import numpy as np
import math
import os
import csv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_csv(filename):
    file = open(filename, 'r')
    dataset = []
    csv_reader = csv.reader(file)
    for line in csv_reader:
        if not line:
            continue
        dataset.append(line)
    return dataset


# filename="./data_full.csv"
# dataset=load_csv(filename)


def getSpecificFile(inputPath, key):
    face = []
    for file in os.listdir(inputPath):
        if os.path.splitext(file)[1] == '.tiff' or os.path.splitext(file)[1] == '.bmp' or os.path.splitext(file)[1] == '.jpg':  # 查找.tif文件
            if file.find(key) != -1:  # 满足条件往下进行
                sourcefile = os.path.join(inputPath, file)  # 拼路径
                face.append(sourcefile)
    return face


def detect(frame):
    face_casade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")  # 使用脸部检测
    # eye_casade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
    # camera = cv2.VideoCapture(0)  # 0代表调用默认摄像头，1代表调用外接摄像头
    while (True):
        # ret, frame = camera.read()
        # print(ret)
        # print(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_casade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:  # 返回的x,y代表roi区域的左上角坐标，w,h代表宽度和高度
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            imagee = frame[y + 1:y + h, x + 1:x + w]
            roi_gray = gray[y + 1:y + h, x + 1:x + w]
            # eyes = eye_casade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))\
            # for (ex, ey, ew, eh) in eyes:
            #    cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        return imagee


# HOG特征提取 cell的梯度
def cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            min_angle = int(gradient_angle / angle_unit) % 8
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    return orientation_centers


def HOG_extract(img):
    img = np.sqrt(img / float(np.max(img)))
    # cv2.imshow('Image', img)
    # cv2.imwrite("Image-test2.jpg", img)
    # cv2.waitKey(0)
    height, width = img.shape
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    # print(gradient_magnitude.shape, gradient_angle.shape)
    # cell_size = 10
    # bin_size = 9
    # angle_unit = 360 / bin_size
    gradient_magnitude = abs(gradient_magnitude)
    cell_gradient_vector = np.zeros((round(height / cell_size), round(width / cell_size), bin_size))
    # print(cell_gradient_vector.shape)
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            # print(cell_angle.max())
            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    # print('kkk:',np.array(hog_vector).shape)
    features = sum(hog_vector, [])
    features.append(1)
    # print(features)
    return features


# 构建Gabor滤波器
def build_filters():
    filters = []
    ksize = [0,1,2,3,4,5,6,7]  # gabor尺度
    lamda = np.pi  # 波长
    for theta in np.arange(0, np.pi, np.pi / 3):  # gabor方向，0°，45°，90°，135°，共四个
        for K in range(len(ksize)):
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 2*np.pi, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


# Gabor特征提取
def getGabor(img, filters):
    res = []  # 滤波结果
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.zeros_like(img)
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            accum = np.maximum(accum, fimg, accum)
        res.append(np.asarray(accum))
    return res


def flipImage(img):
    hor = cv2.flip(img, 1)
    return hor


def getTfeature(img):
    immm = detect(img)
    immm = cv2.resize(immm, (100, 100))
    img1 = immm[20:50, ]
    img2 = immm[50:95, 20:75]
    img1g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    vGH1 = HOG_extract(img1g)
    vGH2 = HOG_extract(img2g)
    vector_A = [vGH1, vGH2]
    vector_A = sum(vector_A, [])
    return vector_A


if __name__ == "__main__":
    # 分类读取文件路径
    inputFilePath = "./jaffe"
    # keys = ['AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU']
    keys = ['FE', 'HA', 'NE', 'SA', 'SU', 'AN', 'DI']
    all_face = []
    for key in keys:
        all_face.append(getSpecificFile(inputFilePath, key))
    # HOG特征提取参数
    cell_size = 10
    bin_size = 9
    angle_unit = 360 / bin_size
    # 统一变为100*100
    dataset = []
    for label in range(len(all_face)):
        for facefile in all_face[label]:
            frame = cv2.imread(facefile)
            img1 = detect(frame)
            img1 = cv2.resize(img1, (100, 100))
            # 先用Gabor提取为24张子图，再对每张子图HOG
            filters = build_filters()
            a = getGabor(img1, filters)
            vector_GH = []
            for im in a:
                grayim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                vector_GH.append(HOG_extract(grayim))
            vGH = sum(vector_GH, [])
            vGH.append(label)
            dataset.append(vGH)
        print('Finish')
    np.random.seed(2020)
    np.random.shuffle(dataset)
    X_train, Y_train = list(), list()
    X_test, Y_test = list(), list()
    # test/train划分比例
    k = 0.7
    for i in range(int(len(dataset) * k)):
        X_train.append(dataset[i][0:len(dataset[0]) - 1])
        Y_train.append(dataset[i][-1])
    for i in range(int(len(dataset) * k), len(dataset)):
        X_test.append(dataset[i][0:len(dataset[0]) - 1])
        Y_test.append(dataset[i][-1])
    '''
    # 随机森林
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(X_train, Y_train)
    print("RF Accuracy on training set is : {}".format(rf.score(X_train, Y_train)))
    print("RF Accuracy on test set is : {}".format(rf.score(X_test, Y_test)))
    Y_pred=rf.predict(X_test)
    print(classification_report(Y_test,Y_pred))
    print(confusion_matrix(Y_test, Y_pred))
    
    # AdaBoost
    ada = AdaBoostClassifier()
    ada.fit(X_train, Y_train)
    print("AdaBoost Accuracy on training set is : {}".format(ada.score(X_train, Y_train)))
    print("AdaBoost Accuracy on test set is : {}".format(ada.score(X_test, Y_test)))
    
    # 多层感知机
    mlp = MLPClassifier(hidden_layer_sizes=(400, 100))
    mlp.fit(X_train, Y_train)
    print("MLP Accuracy on training set is : {}".format(mlp.score(X_train, Y_train)))
    print("MLP Accuracy on test set is : {}".format(mlp.score(X_test, Y_test)))
    Y_pred = mlp.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))
    '''
    # SVM
    svm = SVC(kernel='linear', max_iter=1000)
    svm.fit(X_train, Y_train)
    joblib.dump(svm, 'SVC_L.pkl')
    print("SVM Accuracy on training set is : {}".format(svm.score(X_train, Y_train)))
    print("SVM Accuracy on test set is : {}".format(svm.score(X_test, Y_test)))
    Y_pred = svm.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))
    sns.set()
    heatmap = sns.heatmap(confusion_matrix(Y_test, Y_pred), cmap='Blues')
    plt.show()
    '''
    # 朴素贝叶斯
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    print("Naive Bayes Accuracy on training set is : {}".format(nb.score(X_train, Y_train)))
    print("Naive Bayes Accuracy on test set is : {}".format(nb.score(X_test, Y_test)))
    '''

