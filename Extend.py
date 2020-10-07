from sklearn.externals import joblib
import cv2
import numpy as np
import math


def detect(frame):
    face_casade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")  # 使用脸部检测
    #eye_casade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
    #camera = cv2.VideoCapture(0)  # 0代表调用默认摄像头，1代表调用外接摄像头
    while (True):
        #ret, frame = camera.read()
        #print(ret)
        #print(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_casade.detectMultiScale(gray, 1.3, 5)
        if len(faces)==0:
            print('NULL')
            return -1
        for (x, y, w, h) in faces:  # 返回的x,y代表roi区域的左上角坐标，w,h代表宽度和高度
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            imagee=frame[y+1:y+h,x+1:x+w]
            #eyes = eye_casade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))\
            #for (ex, ey, ew, eh) in eyes:
            #    cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        return imagee


# HOG特征提取 cell的梯度
def cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            min_angle = int(gradient_angle / angle_unit)%8
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
    #cell_size = 10
    #bin_size = 9
    #angle_unit = 360 / bin_size
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
     ksize = [0,1,2,3,4,5,6,7] # gabor尺度，8个
     lamda = np.pi         # 波长
     for theta in np.arange(0, np.pi, np.pi / 3): #gabor方向，0°，45°，90°，135°，共四个
         for K in range(len(ksize)):
             kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
             kern /= 1.5*kern.sum()
             filters.append(kern)
     return filters


# Gabor特征提取
def getGabor(img,filters):
    res = [] #滤波结果
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.zeros_like(img)
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            accum = np.maximum(accum, fimg, accum)
        res.append(np.asarray(accum))
    return res


def featureExtract(frame):
    img=detect(frame)
    if type(img)==int:
        return -1
    img = cv2.resize(img, (100, 100))
    filters = build_filters()
    a = getGabor(img, filters)
    vector_GH = []
    for im in a:
        grayim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        vector_GH.append(HOG_extract(grayim))
    vGH = sum(vector_GH, [])
    return vGH


def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)


def mp4_Img(video):
    success, frame = video.read()
    i = 1000
    timeF = 50
    j = 0
    pics=[]
    while success:
        i = i + 1
        if (i % timeF == 0):
            j = j + 1
            save_image(frame, './mypic/', j)
            print('save image:', i)
            pics.append(frame)
        success, frame = video.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break
    return pics

'''
#图像读取版本：
if __name__ == "__main__":
    svc_l=joblib.load('SVC_L.pkl')
    # HOG特征提取参数
    cell_size = 10
    bin_size = 9
    angle_unit = 360 / bin_size
    me=cv2.imread('./jaffe/17HAPPY.jpg')
    X=featureExtract(me)
    Y=svc_l.predict([X])
    print(Y)
'''


#视频读取版本：
if __name__ == "__main__":
    svc_l=joblib.load('SVC_L.pkl')
    #pca_100=joblib.load('PCA_100.pkl')
    # HOG特征提取参数
    cell_size = 10
    bin_size = 9
    angle_unit = 360 / bin_size
    #videoCapture = cv2.VideoCapture("4.mp4")
    videoCapture = cv2.VideoCapture(0)
    img_list=mp4_Img(videoCapture)
    print(len(img_list))
    X=[]
    ValidList=[]
    for kk in range(len(img_list)):
        x1=featureExtract(img_list[kk])
        if x1 != -1:
            X.append(x1)
            ValidList.append(kk)
        print('FINISH')
    print('有效识别人脸的图片数：', len(X))
    print('有效识别的图片index：', ValidList)
    Y=svc_l.predict(X)
    Final=[]
    # ['FE', 'HA', 'NE', 'SA', 'SU', 'AN', 'DI']
    for item in Y:
        if item==0:
            Final.append('Fear')
        elif item==1:
            Final.append('Happy')
        elif item==2:
            Final.append('Neutral')
        elif item==3:
            Final.append('Sad')
        elif item==4:
            Final.append('Surprise')
        elif item==5 or item==6:
            Final.append('AD')
    print(Final)
