基于计算机视觉手势识别控制系统 (利用YOLO实现)



## 1. 目录

- [目录](#目录)
- [性能情况](#性能情况)
- [文件下载](#文件下载)
- [实现的内容](#实现的内容)
- [数据集](#数据集)
- [所需环境](#所需环境)
- [配置文件](#配置文件)
- [快速运行代码](#快速运行代码)
- [使用Tensorboard](#使用tensorboard)
- [训练步骤](#训练步骤)
- [预测步骤](#预测步骤)
- [评估步骤](#评估步骤)
- [Reference](#reference)

在线服务器体验demo [https://share.streamlit.io/dreaming-future/college-students-innovative-entrepreneurial-training-plan-program/gesture/gesture_streamlit.py](https://share.streamlit.io/dreaming-future/college-students-innovative-entrepreneurial-training-plan-program/gesture/gesture_streamlit.py)

## 2. 基于无人机视觉图像手势识别控制系统

![在这里插入图片描述](https://img-blog.csdnimg.cn/839b3c550ac74ac894e2f0395f61f780.png)

### 2.1. 项目已完成的部分

- [x] 数据集的构建
- [x] 代码的基本运行和训练
- [x] 增加数据集 800 -> 1600
- [x] 利用Mosaic数据增强，但是结果不好，之后训练不会采用，除非数据足够多
- [x] 增加yaml文件，利用yaml配置所有参数
- [x] 提高图片的输入shape，从256x256 -> 416x416
- [x] 由于结果不理想，使用部分自制数据集替换，数据集总数不变
- [x] 添加yolov4 tiny 轻量化模型
- [x] 增加注意力机制，可以比轻量化模型得到更不错的结果
- [ ] 使用MobileNet作为backbone，轻量化模型
- [ ] 使用yolov5 或者 yolox 改进方法



### 2.2. 部分尝试结果

- [x] 使用Mosaic 结果较差
- [x] 在运行过程中结果十分差，原因是数据集标注出现错误，会重新修改数据集
- [x] 用SGD的结果没有Adam好
- [x] front的数据集需要重新修改才能得到更好的结果
- [x] 使用tiny模型速度更快，结果虽然差一点，但是只是一个速度与精度的tradeoff





### 2.3. 项目整体框架

1、 了解项目研究的背景以及其意义，学习其中的创新点和科研价值。

2、 使用python语言对项目中的代码进行编写。研究项目源代码，理解项目工程的代码结构、原理及其功能。

3、 学习深度学习算法。理解卷积神经网络的相关概念，包括神经元系统、局部感受野、权值共享和卷积神经网络总体结构；了解目前常见的目标检测方法和YOLOv4算法框架，以及基于YOLOv4的手势识别算法。

4、 设计并制作针对本项目手势控制数据集，并使用数据增广的方式对数据集进行扩充，同时使用图像处理的方法包括中值滤波、阈值分割等对数据进行预处理。

5、 训练模型，对目标检测性能进行测试。了解实验环境以及评价标准，测试本项目研究的手势识别算法的实验结果，然后通过采用控制变量方法对手势识别算法进行多组实验，以评估其在不同环境下的识别效果，使用验证集对手势识别算法的精度和速度进行性能测试。

6、 总结本项目的研究工作，对基于无人机的手势识别演剧提供创新点与发展建议。

### 2.4. 数据集构建

1. 设计并制作针对本项目手势控制数据集，对数据集进行分类。

![在这里插入图片描述](https://img-blog.csdnimg.cn/5b3c7cc2c58c404987d54d9a2f5bb68d.png)



2. 使用Labelimg标注工具设计针对本项目的手势数据集，对数据集进行标注。

![在这里插入图片描述](https://img-blog.csdnimg.cn/3312206223674362b64026836184e3e4.png)

### 2.5. 模型选择

 **YOLOv4 = CSPDarknet53（主干） + SPP** **附加模块（颈** **） +** **PANet** **路径聚合（颈** **） + YOLOv3（头部）**

![img](https://pdf.cdn.readpaper.com/parsed/fetch_target/699143cdb334ecfc63caf8192472490c_0_Figure_1.png)





## 3. 性能情况

| 训练数据集 |                         权值文件名称                         | 迭代次数 | Batch-size | 图片shape | 平均准确率 | mAP 0.5 | fps   |
| :--------: | :----------------------------------------------------------: | :------: | :--------: | :-------: | :--------: | :-----: | ----- |
| Gesture v1 | [yolo4_gesture_weights.pth](https://github.com/Dreaming-future/College-Students-Innovative-Entrepreneurial-Training-Plan-Program/releases/download/v1.0/yolo4_gesture_weights.pth) |   150    |    4->8    |  256x256  |   61.65    |  51.66  |       |
| Gesture v2 | [yolo4tiny_gesture_SE.pth](https://github.com/Dreaming-future/College-Students-Innovative-Entrepreneurial-Training-Plan-Program/blob/main/yolov4-gesture/model_data/yolotiny_SE_ep100.pth) |   100    |   64->32   |  416x416  |    83.6    |  95.18  | 76.08 |
| Gesture v2 | [yolo4tiny_gesture_CBAM.pth](https://github.com/Dreaming-future/College-Students-Innovative-Entrepreneurial-Training-Plan-Program/blob/main/yolov4-gesture/model_data/yolotiny_CBAM_ep100.pth) |   100    |   64->32   |  416x416  |   89.35    |  98.85  | 70.01 |
| Gesture v2 | [yolo4tiny_gesture_ECA.pth](https://github.com/Dreaming-future/College-Students-Innovative-Entrepreneurial-Training-Plan-Program/blob/main/yolov4-gesture/model_data/yolotiny_ECA_ep100.pth) |   100    |   64->32   |  416x416  |   88.37    |  96.26  | 77.19 |
| Gesture v2 | [yolo4tiny_gesture.pth](https://github.com/Dreaming-future/College-Students-Innovative-Entrepreneurial-Training-Plan-Program/blob/main/yolov4-gesture/model_data/yolotiny_ep100.pth) |   100    |   64->32   |  416x416  |   87.01    |  95.86  | 81.81 |
| Gesture v2 | [yolo4_gesture_weightsv2.pth](https://github.com/Dreaming-future/College-Students-Innovative-Entrepreneurial-Training-Plan-Program/releases/download/v2.0/yolov4_ep100.pth) |   100    |    4->8    |  256x256  |   84.51    |  90.77  | 24.21 |
| Gesture v3 | [yolov4_tiny.pth](https://github.com/Dreaming-future/my_weights/releases/download/v1.3/yolov4_tiny.pth) |   150    |   64->32   |  416x416  |   75.05    |  91.30  |       |
| Gesture v3 | [yolov4_SE.pth](https://github.com/Dreaming-future/my_weights/releases/download/v1.3/yolov4_SE.pth) |   150    |   64->32   |  416x416  |   78.06    |  90.13  |       |
| Gesture v3 | [yolov4_CBAM.pth](https://github.com/Dreaming-future/my_weights/releases/download/v1.3/yolov4_CBAM.pth) |   150    |   64->32   |  416x416  |   91.09    |  94.97  |       |
| Gesture v3 | [yolov4_ECA.pth](https://github.com/Dreaming-future/my_weights/releases/download/v1.3/yolov4_ECA.pth) |   150    |   64->32   |  416x416  |   94.58    |  83.24  |       |
| Gesture v3 | [yolov4_weights_ep150_416.pth](https://github.com/Dreaming-future/my_weights/releases/download/v1.3/yolov4_weights_ep150_416.pth) |   150    |   64->32   |  416x416  |   95.145   |  98.35  |       |
| Gesture v3 | [yolov4_weights_ep150_608.pth](https://github.com/Dreaming-future/my_weights/releases/download/v1.3/yolov4_weights_ep150_608.pth) |   150    |   64->32   |  608x608  |   93.64    |  97.23  |       |

> Gesture v1中存在数据集问题，所以模型结构不好
>
> Gesture v2中重新修改数据集
>
> Gesture v3中修改front数据集

Batch-Size 64->32是指在进行训练的时候，前半段冻结的时候使用的bs为64，在后续不冻结训练使用bs=32



## 4. 文件下载

训练所需的yolo4_weights.pth有两种方式下载。（release包含所有过程的权重，百度网盘和奶牛只记录最新的权重）

- 可以从我的release下载权重

  https://github.com/Dreaming-future/College-Students-Innovative-Entrepreneurial-Training-Plan-Program/releases

- 也可以百度网盘下载

  链接：https://pan.baidu.com/s/1Pt11VHMaHqSsPjb50W5IeQ
  提取码：6666
  
- 由于百度网盘下载速度较慢，这里也给一个不限速的链接 （永久有效）

  传输链接：https://cowtransfer.com/s/dc5e0f7f43a940 或 打开【奶牛快传】cowtransfer.com 使用传输口令：ftyvu0 提取；
  
  

## 5. 实现的内容

- [x] 主干特征提取网络：DarkNet53 => CSPDarkNet53
- [x] 特征金字塔：SPP，PAN
- [x] 训练用到的小技巧：Mosaic数据增强、Label Smoothing平滑、CIOU、学习率余弦退火衰减
- [x] 激活函数：使用Mish激活函数
- [x] 增加yaml配置文件，只需要修改配置文件即可
- [x] 添加detect.py，利用此进行半自动标注，可以方便标注其他类似于对应👋的数据集
- [x] 修改成命令行运行的快速模式，很方便，快速运行和理解
- [x] 利用streamlit部署到服务器上，可以随时使用，在线demo [https://share.streamlit.io/dreaming-future/college-students-innovative-entrepreneurial-training-plan-program/gesture/gesture_streamlit.py](https://share.streamlit.io/dreaming-future/college-students-innovative-entrepreneurial-training-plan-program/gesture/gesture_streamlit.py)
- [ ] ......

## 6. 数据集

![在这里插入图片描述](https://img-blog.csdnimg.cn/5b3c7cc2c58c404987d54d9a2f5bb68d.png)

- **Gesture v1** 只有800张图片，数量较少
- **Gesture** **v2** 增加了800张图片，数量增多，一共1600张图片

在运行过程中结果十分差，原因是数据集标注出现错误，会重新修改数据集

- **Gesture v3** 中修改了front的手势，使得front结果大大提升，平均准确率增大

> 上述展示图是关于Gesture v1的手势，后续数据进行了修改
>

## 7. 所需环境

torch==1.8.1 torchvision==0.9.1

> 代码在更高的版本也是适配的

相对应的库可以直接利用以下代码在当前路径进行运行，利用清华源进行换源

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```



## 8. 配置文件

这个重中之重，在model_data文件夹下，有一个yaml文件，里面包括部分需要运行的参数

只需要调节里面的参数，然后运行就可以得到我们的结果，完全是ok的，只需要改配置文件，其他参数可以在命令行修改，直接运行也是可以使用的，下面会详细介绍

```yaml
#------------------------------detect.py--------------------------------#
# 这一部分是为了半自动标注数据，可以减轻负担，需要提前训练一个权重，以Labelme格式保存
# dir_origin_path 图片存放位置
# dir_save_path Annotation保存位置
# ----------------------------------------------------------------------#
dir_detect_path: ./JPEGImages
detect_save_path: ./Annotation

# ----------------------------- train.py -------------------------------#
nc: 8 # 类别的数量
classes: ["up","down","left","right","front","back","clockwise","anticlockwise"] # 类别
confidence: 0.5 # 置信度
nms_iou: 0.3
letterbox_image: False

lr_decay_type: cos # 使用到的学习率下降方式，可选的有step、cos
# 用于设置是否使用多线程读取数据
# 开启后会加快数据读取速度，但是会占用更多内存
# 内存较小的电脑可以设置为2或者0，win建议设为0
num_workers: 4
```



## 1. 快速运行代码

以下可以在命令行中运行

<details open>
<summary>Install</summary>


```bash
git clone 
cd 
pip install -r requirements.txt  # -i https://pypi.tuna.tsinghua.edu.cn/simple/ # install 可以加上清华源
```

</details>

<details open>
<summary>Data</summary> 
这一部分会生成两个文件，分别是2007_train.txt和2007_val.txt，在每一行包括了图片路径和对应的标签

```bash
python voc_annotation.py
```

</details>

<details open>
<summary>Optional</summary> 
可选，yolov4有对anchors进行Kmeans计算，但是用yolov4自带的也可以

```python
python kmeans_for_anchors.py
```

</details>

<details open>
<summary>Training</summary> 
我们可以在里面设置所需要的参数，phi代表着不同的注意力机制，weights代表着权重，其他的是我们的一些参数的设置，都是可调的，参数的部分解释都可以从python train.py -help看到
```bash
usage: train.py [-h] [--init INIT] [--epochs EPOCHS] [--weights WEIGHTS]
                [--freeze] [--freeze-epochs FREEZE_EPOCHS]
                [--freeze-size FREEZE_SIZE] [--batch-size BATCH_SIZE]
                [--optimizer {sgd,adam,adamw}] [--num_workers NUM_WORKERS]
                [--lr LR] [--tiny] [--phi PHI] [--weight-decay WEIGHT_DECAY]
                [--momentum MOMENTUM] [--save-period SAVE_PERIOD] [--cuda]
                [--shape SHAPE] [--fp16] [--mosaic]
                [--lr_decay_type {cos,step}] [--distributed]

optional arguments:
  -h, --help            show this help message and exit
  --init INIT           从init epoch开始训练
  --epochs EPOCHS       epochs for training
  --weights WEIGHTS     initial weights path 初始权重的路径
  --freeze              表示是否冻结训练
  --freeze-epochs FREEZE_EPOCHS
                        epochs for feeze 冻结训练的迭代次数
  --freeze-size FREEZE_SIZE
                        total batch size for Freezeing
  --batch-size BATCH_SIZE
                        total batch size for all GPUs
  --optimizer {sgd,adam,adamw}
                        训练使用的optimizer
  --num_workers NUM_WORKERS
                        用于设置是否使用多线程读取数据
  --lr LR               Learning Rate 学习率的初始值
  --tiny                使用yolov4-tiny模型
  --phi PHI             yolov4-tiny所使用的注意力机制的类型
  --weight-decay WEIGHT_DECAY
                        权值衰减，可防止过拟合
  --momentum MOMENTUM   优化器中的参数
  --save-period SAVE_PERIOD
                        多少个epochs保存一次权重
  --cuda                表示是否使用GPU
  --shape SHAPE         输入图像的shape，一定要是32的倍数
  --fp16                是否使用混合精度训练
  --mosaic              Yolov4的tricks应用 马赛克数据增强
  --lr_decay_type {cos,step}
                        cos
  --distributed         是否使用多卡运行
```

这里对一些常用参数进行解释

- fp16

  由于训练神经网络，有时候得到的权重的精度都是64位或者32位的，保存和训练的时候都占了很多显存，但是有时候这些是不必要的，所以可以利用fp16将精度设为16位，这样大概可以减少一半的显存

- phi

  这里解释一下，phi = 0代表的是yolov4_tiny，也就是改进的轻量化yolov4，而phi = 1,2,3分别是加了SE，CBAM，ECA三种注意力机制得到的结果。

- freeze

  除此之外，可以从下面的代码看出，我们可以冻结进行迁移学习，也可以选择不冻结，通过参数freeze来控制，还可以控制冻结次数的冻结时的batch-size，冻结的时候，可以把batch-size调高一点

```python
# 冻结进行迁移学习
python train.py --tiny --phi 1 --epochs 100 \
        --weights model_data/yolov4_SE.pth \
        --freeze --freeze-epochs 50 --freeze-size 64 \
        --batch-size 32 --shape 416 \
        --fp16 --cuda

# 快速运行尝试，重新学习
python train.py --tiny --phi 1 --epochs 10 \
        --batch-size 4 --shape 416 \
        --fp16 --cuda
```

在后续为了简化操作，不用打那么多的字母，还进行了缩写的修改，把--freeze简化成-f，--weights 简化成 -w, --freeze-epochsj简化成-fe，--freeze-size 简化成fb， --batch-size简化成-bs，这是为了方便运行的时候设置参数

这段代码和上面是等价的

```
# 冻结进行迁移学习
python train.py --tiny --phi 1 --epochs 100 \
        -w model_data/yolov4_SE.pth \
        -f -fe 50 -fs 64 \
        --bs 32 --shape 416 \
        --fp16 --cuda

# 快速运行尝试，重新学习
python train.py --tiny --phi 1 --epochs 10 \
        --batch-size 4 --shape 416 \
        --fp16 --cuda
```

</details>

<details open>
<summary>Predict</summary> 
```python
# python predict.py --mode dir_predict \
# --tiny --phi 1 \
# --weights model_data/yolov4_SE.pth \
# --cuda --shape 416
python predict.py --tiny --cuda
```

</details>

<details open>
<summary>Get Map</summary> 
```python
# 对验证集进行计算
# python get_map.py --mode 0 \
# --tiny --phi 1 \
# --weights model_data/yolov4_SE.pth \
# --cuda --shape 416
# python .\get_map.py --cuda --mode 0 --tiny --phi 3 --weights model_data/yolotv4_ECA.pth
!python get_map.py --tiny --cuda
```

</details>

所有的参数都可以通过，通过help看到解释

```python
python train.py -h
python get_map.py -h
python predict.py -h
```

除此之外，如果有多个GPU，需要设定指定的GPU，在python前加上配置CUDA_VISIBLE_DEVICES=3，表示使用第四块GPU

```python
# 比如使用第四块GPU
CUDA_VISIBLE_DEVICES=3 python train.py ...
```



## 1. 使用Tensorboard

在我们训练的过程中，我们可以用TensorBoard实时查看训练情况，也可以看到训练的网络模型结构，非常方便

只需要在我们的文件夹的命令行下，运行

```bash
tensorboard --logdir='logs/'
```

之后大概我们的6006端口就可以实时看到我们的结果，即是https://localhost:6006

> 如果是使用Ubuntu，有可能会出现一些bug，所以需要进行一些操作，因为会显示无法找到命令
>
> 这时候首先需要找到TensorBoard在库的哪里
>
> ```bash
> pip show tensorboard
> ```
>
> 这样子就能看到自己tensorboard下载的路径
>
> 然后找到TensorBoard的文件夹下，找到main.py文件，就可以进行了，利用绝对路径就可以了
>
> ```
> python .../python3.8/site-packages/tensorboard/main.py --logdir='logs/' 
> ```



## 2. 训练步骤

1. 数据集的准备 （这一部分已经做了下载，所以说不用做操作，如果需要换自己的数据集，那这里还是重中之重的）
   训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
   训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

   

2. 数据集的处理 
   在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
   修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   

   然后再前面所说的data.yaml中写清楚自己的类别以及类别的数量

   ```bash
   nc: 8 # 类别的数量
   classes: ["up","down","left","right","front","back","clockwise","anticlockwise"] # 类别
   ```

   

3. 开始网络训练  

   之后根据快速运行train.py，运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中，可以自己设迭代次数保存权重，如上述快速运行即可。

   

4. 训练结果预测 
   训练结果预测需要用到两个文件，分别是yolo.py和predict.py。  
   完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  （可以自己设置模式得到结果）



## 3. 预测步骤

1. 下载完库后解压，在百度网盘后者其他地方下载yolo_gesture_weights.pth，放入model_data，运行predict.py，调整权重路径

   在predict.py中事先设置了`dir_predict`表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，这样就可以在img_out中得到文件

   有很多种模式，可以通过mode来调节，这一部分还可以设置参数，我们都可以从help里看到

   ```bash
   predict.py -h
   usage: predict.py [-h] [--weights WEIGHTS] [--tiny] [--phi PHI]
                     [--mode {dir_predict,video,fps,predict,heatmap,export_onnx}]
                     [--cuda] [--shape SHAPE] [--video VIDEO]
                     [--save-video SAVE_VIDEO] [--confidence CONFIDENCE]
                     [--nms_iou NMS_IOU]
   
   optional arguments:
     -h, --help            show this help message and exit
     --weights WEIGHTS     initial weights path
     --tiny                使用yolotiny模型
     --phi PHI             yolov4tiny注意力机制类型
     --mode {dir_predict,video,fps,predict,heatmap,export_onnx}
                           预测的模式
     --cuda                表示是否使用GPU
     --shape SHAPE         输入图像的shape
     --video VIDEO         需要检测的视频文件
     --save-video SAVE_VIDEO
                           保存视频的位置
     --confidence CONFIDENCE
                           只有得分大于置信度的预测框会被保留下来
     --nms_iou NMS_IOU     非极大抑制所用到的nms_iou大小
   ```

   如果下载了权重于路径model_data/yolov4_tiny.pth，默认是文件夹中的图片模式运行，我们就可以直接运行得到结果

   ```python
   python predict.py --tiny --phi 0 --weights model_data/yolov4_tiny.pth
   ```

   

2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  （这一部分可以自己尝试）

这一部分只要设置一下路径和视频即可



## 4. 评估步骤

1. 本文使用VOC格式进行评估。  

2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。

3. 利用voc_annotation.py划分测试集

4. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

   

## 5. Reference

- [https://github.com/bubbliiiing/yolov4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch)
- https://github.com/qqwweee/keras-yolo3/  
- https://github.com/Ma-Dan/keras-yolo4  
