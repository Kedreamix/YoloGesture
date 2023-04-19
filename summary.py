#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchinfo import summary

from nets.yolo import YoloBody
from nets.yolo_tiny import YoloBodytiny
if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 80).to(device)
    summary(m, input_size=(1,3, 416, 416))

    m       = YoloBodytiny([[3, 4, 5], [1, 2, 3]], 80, phi = 1).to(device)
    summary(m, input_size=(1,3, 416, 416))
