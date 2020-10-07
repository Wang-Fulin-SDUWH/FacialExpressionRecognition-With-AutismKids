from flask import Flask, request
import sys
import Extend
from server2 import wxCloud

app = Flask(__name__)



svc_l=joblib.load('SVC_L.pkl')

@app.route('/picrec', methods=['post']) # 上传图片识别
def picrec():
    img_list = request.files['file']
    # HOG特征提取参数
    cell_size = 10
    bin_size = 9
    angle_unit = 360 / bin_size
    X=[]
    global ValidList = []
    for kk in range(len(img_list)):
        x1 = featureExtract(img_list[kk])
        if x1 != -1:
            X.append(x1)
            ValidList.append(kk)
    global Y=svc_l.predict(X)

    return y

@app.route('/save', methods=['post'])   # 结果保存数据库
def save():
    APP_ID = '***'
    APP_SECRET = '****'
    ENV = 'test-****'
    ID = request.get_json()['id']
    timeStamp=request.get_json()['timestamp']
    timeStamp=[timeStamp[i] for i in ValidList]
    Y=list(Y)

    db = wxCloud(APP_ID, APP_SECRET, ENV)
    db.collection = 'test'
    if db.query_data(id=ID):
        print(1)
        db.update_data(     #存在则更新
            id=ID,
            timeStamp=timeStamp,
            mood=Y
        )
    else:
        print(0)
        db.add_data(        #不存在则添加
            id=ID,
            timeStamp=timeStamp,
            mood=Y
        )

    return 0



if __name__ == '__main__':
    app.run(debug=True,
            host='127.0.0.1',
            port=8080
            )
