import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from track.utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml, increment_path, print_args)
from track.utils.pathloader import PathLoader
from track.utils.torch_utils import select_device
from ultralytics import YOLO
import cv2
from PIL import Image 
from collections import defaultdict 
from datetime import datetime

def save_tracking_txt(predn, frame_num, video_name, result_path):
    if os.path.exists(os.path.join(result_path, video_name+'.txt')):
        print()
        f = open(os.path.join(result_path, video_name + '.txt'), 'a')
    else:
        f = open(os.path.join(result_path, video_name + '.txt'), 'a')
    for pred in predn:
        f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(frame_num, pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], int(pred[6]) + 1, -1, -1))
    f.close()


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        save_txt=False,  # save results to *.txt
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        config_deepsort=None,
        augment=False,
        verbose=False,
        save_hybrid=False,
        save_conf=False,
        save_json=False,
        end2end=False,
        ):
    # Initialize/load model and set device
    training = model is not None
    suffix = None 
    result_path = ROOT / "track" / "results"
    os.system(f"rm {result_path}/uav*.txt")
    track_path = "../track"

    device = select_device(device, batch_size=batch_size)
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = YOLO(weights, )
    gs = 32
    imgsz = check_img_size(imgsz, s=gs)  # check image size

    # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
    # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # Data
    data = check_dataset(data)  # check
    dataset_path = data['path'] +'/VisDrone_mot_val' + '/sequences'

    # Half
    if suffix == '.pt':
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
        # Configure
        model.eval()
    
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    

    # initialize deepsort
    video_name = None
    pad = 0.0 if task == 'speed' else 0.5
    task='val'

    dataloader = PathLoader(dataset_path) 
    print("prefix")
    print(f'{task}: ')
    print("pad", pad)

    tbar = tqdm(dataloader)
    time_statistics = defaultdict(lambda:[]) 

    # get current date and time for result directory name
    dirname = datetime.now().strftime("%Y%m%d-%H%M%S")
    # if dirname doesnt exist create it
    if not os.path.exists(f'{result_path}/{dirname}'):
        os.makedirs(f'{result_path}/{dirname}')

    for x, y in enumerate(tbar):
        paths, has_next = y 
        if video_name is None:
            video_name = paths.split('/')[-2]
            frame_num = 0
        else:
            if video_name != paths.split('/')[-2]:
                video_name = paths.split('/')[-2]
                frame_num = 0
        img_name = paths.split('/')[-2:]
        tbar.set_description("video: %s frame: %s"%(img_name[0], img_name[1]))
        frame_num += 1
        im = Image.open(paths)
        if not os.path.exists(f'{track_path}/example/inputs'):
            os.makedirs(f'{track_path}/example/inputs')
        im.save(f'{track_path}/example/inputs/frame_{x}.png',)
        out = model.track(im, persist=True, 
                tracker='bytetrack.yaml', 
                iou=iou_thres, conf=conf_thres, imgsz=(imgsz, imgsz))  # inference and training outputs
        
        out = out[0]
        annotated_frame = out.plot() 
        cv2.imwrite(f'example/outputs/frame_{x}.png', annotated_frame)

        has_track = (out.boxes.id is not None)
        im_ndarr = np.asarray(im)

        if has_track:
            boxes = out.boxes.xyxy.cpu().numpy()
            w = boxes[..., 2] - boxes[..., 0] 
            h = boxes[..., 3] - boxes[..., 1] 
            boxes[:, 2] = w 
            boxes[:, 3] = h 

            conf = out.boxes.conf.cpu().numpy()[..., None]
            class_id = out.boxes.cls.int().cpu().numpy()[..., None]
            track_ids = out.boxes.id.int().cpu().numpy()[..., None] 
        
            predn = np.concatenate([track_ids, boxes, conf, class_id], axis=-1)
            predn = list(predn)
            save_tracking_txt(predn, frame_num, video_name, f'{result_path}/{dirname}')

        speed = out.speed  
        for k, v in speed.items():
            time_statistics[k].append(v) 
            
        if not has_next:
            # need to revise here 
            print("RELOAD")
            del model 

            model = YOLO(weights)

    for k, v in time_statistics.items():
        print(f'{k}:{np.mean(v)}') 

    # save inference time in txt file
    inference_time = time_statistics['inference'] +  time_statistics['postprocess']
    with open(f'{result_path}/inference_time.txt', 'w') as f:
        f.write(f'{inference_time}\n')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT/ 'data/VisDrone.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / '/workspace/yolov10n_ff.pt' ,help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3 , help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--end2end', type=bool, default=False, help='is model compiled to end2end?') 

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=opt.conf_thres, iou_thres=opt.iou_thers,
                device=opt.device, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
