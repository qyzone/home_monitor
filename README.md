# home_monitor

#### 介绍
使用OpenCV获取萤石云视频流，并使用PyTorch分析对视频流，将分析结果发送到邮箱。可应用于摄像头实时人形检测，发现人形及时通知，并存储视频。

深度学习模型可以自己定义，也可以使用本仓库的，也可以将yolov5的模型应用在本项目下。

#### 安装教程
1. 安装OpenCV和Pytorch
2. 克隆本项目到本地

#### 训练模型
1. 训练自定义模型
- (1) 在“model/”目录添加自定义网络，或者使用“model/”目录中已写好的网络，然后在train.py中import
- (2) 本项目使用的默认使用ImageFolder作为dataset,具体代码在“tool/dataloader.py”中，有需要可以自行更改。
另外，本项目为了防止resize时造成图片拉伸，所以定义了一个Resize类，resize后的缺少部分用黑边填充，与yolov5类似。
- (3) 将要训练的数据整理好了，只需要运行train.py就可以开始训练了
2. 使用yolov5模型
- (1) 将整个yolov5项目克隆下来, 并放在本项目的根目录下
- (2) 可以去yolov5的github仓库下载weight文件，也可以cd进yolov5目录，用yolov5项目的代码训练模型

#### 检测或预测
1. 修改yaml配置文件
2. 修改根目录部分py文件中__main__函数中的代码
3. 通过设置VideoReceiver(remote:bool)类中的remote参数值为True，来启用萤石云的URL，否则使用局域网的摄像头URL
4. 运行predict.py使用自定模型进行预测，或运行yolo_detect.py使用yolov5模型进行检测

#### 联系作者
1. 源码地址: https://gitee.com/finebit/home_monitor
2. 邮箱: finebit@qq.com
3. 微信公众号: 泛比特
4. 知乎搜索: 青颜君
5. 个人网站: finebit.cn
