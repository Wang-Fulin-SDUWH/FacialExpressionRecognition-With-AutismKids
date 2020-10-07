### 孤独症诊疗辅助系统

Autism Diagnosis and Treatment Assistance System

---

本系统旨在借助机器视觉及机器学习的有关方法, 在孤独症(Autism)的诊断和治疗中提供辅助. 目前已实现通过人脸的情绪识别, 帮助从业人员了解患者或潜在患者的负面情绪出现的时段, 所占比率等统计信息, 并可帮助从业人员快速调取患者产生负面情绪前后的录像视频, 从而帮助从业人员诊断是否有患病风险或参考情绪状态和相关需求，制定更合理的康复方案。

- Extend.py

  情绪识别主文件, 实现摄像头/视频抓图, 人脸提取, 特征提取, 分类及结果上传. 
  
  需输入模型文件, 更改小程序APPID, APP_SECRET参数.

- [人脸提取](https://github.com/Wang-Fulin-SDUWH/Autism/tree/master/cascades)

  使用opencv提供的人脸提取模型, 将图片中的人脸部分选出. 

- server.py

  应用flask的服务器接口, 服务器使用nginx+gunicorn+flask架构. 
- server2.py

  实现本地操作小程序云数据库, 将情绪识别结果写入. 

- [微信小程序](https://github.com/Wang-Fulin-SDUWH/Autism/tree/master/applet)

  实现从云数据库查询识别结果并可视化, 查询需指定ID及查询时段. 
  
  使用需绑定自己的云开发环境, 并创建对应collection. 


