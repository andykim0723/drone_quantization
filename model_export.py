import os
from ultralytics import YOLO

root_path = "runs/detect/train8/weights/epoch{}.pt"

model = YOLO("runs/detect/train8/weights/epoch0.onnx")
model.export(format="engine", dynamic=False, batch=1, imgsz=1024, 
            simplify=True, device='cuda', optimize=False, nms=True, half=True,
            workspace=8)
    # model.export(format="engine", dynamic=False, batch=1, imgsz=1024, 
    #             simplify=True, device='cuda', optimize=False, nms=True, half=True,
    #             workspace=8)
# for i in range(0, 301, 5):
#     print(root_path.format(i))
#     # if root_path exists
#     if not os.path.exists(root_path.format(i)):
#         continue
#     model = YOLO(root_path.format(i))  
#     model.export(format="engine", dynamic=False, batch=1, imgsz=1024, 
#                 simplify=True, device='cuda', optimize=False, nms=True, half=True,
#                 workspace=8)
# exit()


# model.export(format="engine", int8=True, data="./cfgs/VisDrone_train.yaml")
# model.export(
#     format="engine",
#     dynamic=True,  
#     batch=32,  1
#     int8=True,
#     data="./cfgs/VisDrone_train.yaml",  
# )

# model = YOLO("yolov8n.engine", task="detect")
