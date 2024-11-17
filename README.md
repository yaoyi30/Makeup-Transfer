<div align="center">   
  
# Makeup Transfer Project
</div>

### 项目介绍
软件的功能是妆容迁移，具体来说，假设有一张上了妆的人脸图像A，还有一张没上妆的素颜人脸图像B，妆容迁移就是给B上A的妆容。所有的模型均已转成onnx格式，不依赖于pytorch。

注意：原图中可以为多个人脸，参考图中最好只有一个人脸，如果有多个人脸，会选择人脸检测模型输出置信度最大的人脸。

演示视频：
https://www.bilibili.com/video/BV1ifmeY7EKq/?spm_id_from=333.999.0.0&vd_source=4cd5ac8dda02d0b3152cd9b05f7e4006

### 环境配置
python version 3.8, opencv-python version 4.4.0.46, onnxruntime version 1.19.2, numpy version 1.24.4, pyqt5 version 5.15.9:
```setup
pip install opencv-python==4.4.0.46 onnxruntime==1.19.2 numpy==1.24.4 pyqt5==5.15.9
```
### 运行效果
![out1.PNG](output%2Fout1.PNG)
![out2.PNG](output%2Fout2.PNG)
![out4.PNG](output%2Fout4.PNG)
![out7.PNG](output%2Fout7.PNG)
![out8.PNG](output%2Fout8.PNG)
![out5.PNG](output%2Fout5.PNG)
![out10.PNG](output%2Fout10.PNG)

### 参考
1. 人脸检测：https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
2. 面部解析：https://github.com/zllrunning/face-parsing.PyTorch
3. 妆容迁移：https://github.com/Snowfallingplum/CSD-MT