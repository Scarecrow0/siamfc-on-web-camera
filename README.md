# SiamFC on PC Web Camera - TensorFlow
TensorFlow port of the tracking method described in the paper [*Fully-Convolutional Siamese nets for object tracking*](https://www.robots.ox.ac.uk/~luca/siamese-fc.html).

## how work
从笔记本的web camera获取图像，并对选中的目标进行跟踪。

## usage
需要安装 tensorflow, opencv-python.
直接运行main.py, 首先会提示用户进行多长帧数的跟踪。输入完成后弹出窗口，将待跟踪的物体放入窗口的中的方框中。

方框消失后，开始进行跟踪的视频采集，会有一个窗口显示采集视频的结果。

采集完成后，开始进行跟踪的过程，算法会将每一帧跟踪的结果实时显示出来。
###performance
在Huawei Matebook X Pro(I5 8250U + 8GB ram), 平均每一帧inference的时间为0.4~0.7s, 对于24fps的视频不能做到实时。 

## License
This code can be freely used for personal, academic, or educational purposes.
Please contact us for commercial use.

