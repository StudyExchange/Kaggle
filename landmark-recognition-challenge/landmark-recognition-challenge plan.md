# 地标识别计划

## 核心方案
1. 使用ResNetV2作为特征提取器，后接全链接神经网络作为分类器。一共有1.5k个landmark，全链接网络可以做宽一点，做深一点。因为类别太多，可以考虑其他更加适合的分类器。可以考虑多个特征提取器平行拼接融合，比如加入InceptionV3。
2. 这个landmark可以视为场景识别类似的一个工程。参照，Dog Bread中使用的YOLO提取特定区域的方法，可以考虑识别激活区域，然后，提取激活区域，缩小识别的范围，这样看能不能提升效果。

## 辅助事务
1. 注册bitbicket账户，用于保存代码，以便比赛结束前的保密
2. 下载图片主机环境配置（c4.xlarge）。创建一个专门用于下载图片的主机，配置好anaconda3的python3环境，参考[配置AWS p2.xlarge环境.md](https://github.com/StudyExchange/Udacity/blob/master/MachineLearning(Advanced)/p6_graduation_project/%E9%85%8D%E7%BD%AEAWS%20p2.xlarge%E7%8E%AF%E5%A2%83.md)
    - 安装anaconda3，使用默认的python作为运行环境
    - 安装git
    - 挂载500G数据磁盘并分区
    - 安装pip install kaggle-cli
2. 数据预处理主机环境配置（p2.xlarge）。创建一个专门用于图片分组的主机，配置好anaconda3的python3环境，参考[配置AWS p2.xlarge环境.md](https://github.com/StudyExchange/Udacity/blob/master/MachineLearning(Advanced)/p6_graduation_project/%E9%85%8D%E7%BD%AEAWS%20p2.xlarge%E7%8E%AF%E5%A2%83.md)
    - 安装anaconda3，使用默认的python作为运行环境
    - 安装git
    - 挂载500G数据磁盘

3. 创建核心的CPU主机（t2.2xlarge）。
    - 安装anaconda3，使用默认的python作为运行环境
    - 安装git

## 参考
- https://www.kaggle.com/anokas/python3-dataset-downloader-with-progress-bar
- https://www.kaggle.com/codename007/a-very-extensive-landmark-exploratory-analysis
