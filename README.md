# HR101-MXNet
**This is an inference-only implementation for MXNet of [tiny face](https://github.com/peiyunh/tiny).**
See the following references for more information:
```
"Finding Tiny Faces."
Peiyun Hu, Deva Ramanan
arXiv preprint arXiv:1612.04402, 2016.
```
[https://arxiv.org/abs/1612.04402](https://arxiv.org/abs/1612.04402)

## Getting Started
  * Download the origin author's model [from here.](https://www.cs.cmu.edu/~peiyunh/tiny/hr_res101.mat)
  * Transform the origin Matconvnet model to MXNet model using matconvnet_hr101_to_mxnet.py.
  * To run:
    * simply run tiny_detection_mxnet.py
    * use from your code

```python
from tiny_fd import TinyFacesDetector
import cv2 as cv

detector = TinyFacesDetector(model_root='./', prob_thresh=0.5, gpu_idx=0)
img = cv.imread('./selfie.jpg')
boxes = detector.detect(img)
```

## Differences
  You should install opencv-python 2.4 to get the nearly completely identical result to the origin implemetation.

## Other implementations
[cydonia999's Tensorflow implemetation](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow)
