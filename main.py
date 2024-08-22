from ultralytics import YOLO

model = YOLO('./weights/yolov8l.pt', task='detect')

model.train(data='./cfgs/VisDrone_custom.yaml', epochs=100, imgsz=640, device='0', batch=24, optimizer='Adam', 
         weight_decay=1e-5)
# model.train(data='./cfgs/VisDrone_custom.yaml', epochs=100, imgsz=640, device='0, 1', batch=64, optimizer='Adam', 
#          weight_decay=5e-6)