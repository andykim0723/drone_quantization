# drone_quantization

this repository is for model quantization for object tracking model for visdrone. 


## train and evaluation
You have to first train the object detection model, then infer with object tracker to get object tracking results.
### 0. download datasets
* download the VisDrone detection/tracking data in [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset). 
* **VisDrone2019-DET-train** and **VisDrone2019-DET-test-dev** -> train MOD, **VisDrone2019-DET-val** and **VisDrone2019-MOT-val** -> test MOD, MOT.   

### 1.  train object detector
```
python train.py
```
### 2. generate tracking results
```
python track_yolov8.py --weights ./runs/detect/train/weights/best.pt --data ./cfgs/VisDrone_custom.yaml
```
### 3. evaluate mAP score in using result .txt files
* the evaluation code is implemented with MATLAB, and is in ```track/matlab_eval```.  We modified [VisDrone2018-DET-toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit) to evaluate MOT.
* set correct datasetPath and resPath in  ```evalVID.m```, then run ```evalVID```.

## model quantization
we use ONNX and TensortRT for model quantization:
## ONNX


## TensorRT
