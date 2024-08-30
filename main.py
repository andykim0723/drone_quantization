import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def main(opt):
    model = YOLO(opt.weight, task='detect')
    model.train(data=opt.data_cfg, epochs=opt.epoch, imgsz=opt.imgsz, device=opt.device, batch=opt.batch_size, optimizer='Adam', 
            weight_decay=1e-5)
    # model.train(data='./cfgs/VisDrone_custom.yaml', epochs=100, imgsz=640, device='0, 1', batch=64, optimizer='Adam', 
    #          weight_decay=5e-6)


def parse_opt():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--weight', type=str, default=ROOT / 'weights/yolov8l.pt' ,help='model.pt path')
    parser.add_argument('--data-cfg', type=str, default=ROOT / 'cfgs/VisDrone_custom.yaml' ,help='data config path')
    parser.add_argument('--epoch', type=int, default=100, help='train epochs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=24, help='batch size')
    parser.add_argument('--device', type=str, default='0, 1', help='gpu device')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)