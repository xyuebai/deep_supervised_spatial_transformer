# deep_supervised_spatial_transformer
Deep learning method for face augumentation with unsovable bug

## Getting Started
```
cd deep_supervised_spatial_transformer
python3 stn_mobileV2_multi.py --dataroot "/pathtoyourdatabase" --resume "results/mobilenet_v2_1.0_224/model_best.pth.tar" 

```
### Prerequisites

1. Ubuntu 18.04.1 LTS
2. Anaconda 3.6
3. Cuda 9.2
4. Pytorch 0.4.0

### Installing

### Result
###### Good looking loss graph
![Good looking loss function](result/loss.png)
###### However, disaster performing
![Poor performance](result/result.gif)


### Reference
MobileNetv2 in PyTorch: https://github.com/Randl/MobileNetV2-pytorch

