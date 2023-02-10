<구조>
<<임포트>>
from pathlib import Path
import argparse
import torch 
import cv2
import sys
import os
import warnings
warnings.filterwarnings('ignore')


<<경로>>??
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory

<<경로관련어쩌구>>????
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


<<임포트>>
from strong_sort.utils.parser import get_config ####
from strong_sort.strong_sort import StrongSORT ####
from yolov5.utils.general import non_max_suppression_face, scale_coords, xyxy2xywh
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import scale_coords, xyxy2xywh, non_max_suppression_face, xyxy2xywh_wider
from yolov5.models.experimental import attempt_load ####


<<Yolov5Sort클래스 정의>>
class Yolov5Sort:

	<< 그 안에  메서드들 정의 >>
	def __init__(
	        self,
	        model_path: str = "yolov5m.pt",
	        config_path: str = 'osnet_x0_25_market1501.pt',
	        sort_cfg: str = './',
	        device: str = "cpu",
	        confidence_threshold: float = 0.5,
	        image_size: int = 640,
	        view_img: bool = True,
	        augment: bool = False,
	        save_vid: bool = False,
	        save_crop: bool = False,
	        save_dir: str='./output',
	        except_ids: list=[]
	    ):
	        self.model_path = model_path
	        self.device = select_device(device)
	        self.confidence_threshold = confidence_threshold
	        self.load_model()
	        self.prediction_list = None
	        self.image_size = image_size
	        self.config_path = config_path
	        self.view_img = view_img
	        self.augment = augment
	        self.save_vid = save_vid
	        self.save_dir = save_dir
	        self.save_crop = save_crop
	        self.except_id = dict()
	        # for id in list([int(ids) for ids in except_id.split(',')]):
	        #     self.except_id[id] = True
	        for id in except_ids:
	            self.except_id[int(id)]=True
	        # initialize strongsort
	        self.cfg = get_config()
	        self.cfg.merge_from_file(sort_cfg)
	
	def load_model(self):
	        import yolov5
	        model = attempt_load(self.model_path, device=self.device)
	        model.conf = self.confidence_threshold
	        self.model = model
	
	
	def yolo_tracker(self, video_path):
	        dataset = LoadImages(video_path, self.image_size)
	        vid_path, vid_writer = [None], [None]
	        strongsort_list = []        
	        strongsort_list.append(
	            StrongSORT(
	                model_weights= self.config_path,
	                device=self.device,
	                max_dist=self.cfg.STRONGSORT.MAX_DIST,
	                max_iou_distance = self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
	                max_age = self.cfg.STRONGSORT.MAX_AGE,
	                n_init = self.cfg.STRONGSORT.N_INIT,
	                nn_budget = self.cfg.STRONGSORT.NN_BUDGET,
	                mc_lambda = self.cfg.STRONGSORT.MC_LAMBDA,
	                ema_alpha = self.cfg.STRONGSORT.EMA_ALPHA,
	            )
	        )
	
	
	outputs = [None] 
	        seen = 0
	        curr_frames, prev_frames = [None], [None]
	        for path, im, im0s, vid_cap, s in dataset:
	            im = torch.from_numpy(im).to(self.device)
	            im = im.float()  # uint8 to fp16/32
	            im /= 255  # 0 - 255 to 0.0 - 1.0
	            if len(im.shape) == 3:
	                im = im[None]  # expand for batch dim
	            inf_out, _ = self.model(im,augment=self.augment) #  size=self.image_size,
	            pred = non_max_suppression_face(inf_out, conf_thres=self.model.conf) #, conf_thres=self.model.conf, iou_thres=self.model.iou, classes=self.model.classes, agnostic=self.model.agnostic)
	            for i, det in enumerate(pred):
	                annotator = Annotator(im0s, line_width=2, example=str(self.model.names))
	                det = torch.cat((det[:, :5], det[:, 15:]), 1)
	                det.to(self.device)
	                path = Path(path)
	                save_path=os.path.join(self.save_dir, path.name)
	                curr_frames[i] = im0s.copy()
	                if self.cfg.STRONGSORT.ECC:  # camera motion compensation
	                    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
	                if len(det):
	                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
	                    xywhs = xyxy2xywh_wider(det[:, 0:4]).cpu().detach() #.numpy()
	                    confs = det[:, 4].cpu().detach() #.numpy()
	                    clss = det[:, 5].cpu().detach() #.numpy()
	                    outputs[i] = strongsort_list[i].update(xywhs, confs, clss, im0s)
	                    if len(outputs[i]) > 0:  ## DETECT 안될때 우쨔
	                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):
	                            bboxes = det[j, 0:4].cpu().numpy()  ### bounding box change
	                            id = output[4]
	                            cls = output[5]
	                            # bboxes = output[0:4]
	                            # id = output[4]
	                            # cls = output[5]
	                            
	                            if self.view_img or self.save_crop or self.save_vid:  # Add bbox to image
	                                c = int(cls)  # integer class
	                                id = int(id)  # integer id
	                                label = label = "%s %.2f %d" % (self.model.names[int(cls)], conf, id)
	                                annotator.box_label(bboxes, label, color=colors(c, True))
	                                os.makedirs(os.path.join(self.save_dir, str(id)), exist_ok=True)
	                                img_crop = im0s[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]
	                                cv2.imwrite(os.path.join(self.save_dir, str(id),f'{i}-{j}-{seen}.jpg'),img_crop)
	
	                            if self.save_crop:
	                                os.makedirs(os.path.join(self.save_dir, 'crop', str(id)), exist_ok=True)
	                                img_crop = im0s[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]),:]
	                                cv2.imwrite(os.path.join(self.save_dir, 'crop',str(id), f'{i}-{j}-{seen}.jpg'), img_crop)
	                                seen += 1
	                            if self.except_id.get(id):
	                                pass
	                            else:
	                                im0s[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]),:] = \
	                                    cv2.GaussianBlur(im0s[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]),:], (0,0), 10)
	                prev_frames[i] = curr_frames[i]
	
	
	
	# Stream results
	            #     prev_frames[i] = curr_frames[i]
	                im0 = annotator.result()
	                if self.view_img:
	                    cv2.namedWindow('show-vid', cv2.WINDOW_NORMAL)
	                    cv2.resizeWindow('show-vid', 640, 360)
	                    cv2.imshow('show-vid', im0)
	                    if cv2.waitKey(1) == ord('q'):
	                        break
	                if self.save_vid:
	                    if vid_path[i] != save_path:  # new video
	                        vid_path[i] = save_path
	                        if isinstance(vid_writer[i], cv2.VideoWriter):
	                            vid_writer[i].release()  # release previous video writer
	                        if vid_cap:  # video
	                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
	                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	                        else:  # stream
	                            fps, w, h = 30, im0.shape[1], im0.shape[0]
	                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
	                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
	                    vid_writer[i].write(im0)
	
	
	
<< 그러고 명령어.. 파서함수 생성 >>
def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO v5 video stream detector')
    parser.add_argument('--model_path', type=str, default='yolov5s-face.pt', help='path to weights file')
    parser.add_argument('--config_path', type=str, default='osnet_x1_0_market1501.pt', help='path to configuration file')
    parser.add_argument('--sort_data', type=str, default='.\strong_sort\configs\strong_sort.yaml', help='strong_sort.yaml')
    parser.add_argument('--image_size', type=int, default=640, help='size of each image dimension')
    parser.add_argument('--video_path', type=str, default='test_hyeri.mp4', help='path to input video file')
    parser.add_argument('--confidence', type=float, default=0.5, help='minimum probability to filter weak detections')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view_img', action='store_true', help='display results')
    parser.add_argument('--augment', action='store_true', help='augmented video')
    parser.add_argument('--save-dir', default='./output', help='save dir path')
    parser.add_argument('--save_vid', action='store_true', help='save modified video')
    parser.add_argument('--except_id', type=int, nargs='*', default=[],help='id list excepted in blur process')

    return parser.parse_args()


<< 이제 Yolov5Sort클래스 이용해서 파서값 집어넣는 함수 정의 !!!!!! >>
def run(args):
    Yolov5Sort(args.model_path, args.config_path, args.sort_data, args.device, args.confidence, args.image_size,
               args.view_img, args.augment, args.save_vid, save_dir=args.save_dir, except_ids=args.except_id).yolo_tracker(args.video_path)


<< 마지막으로 이 코드를 실행하겠다~~ >>
if __name__ == '__main__':
    args = parse_arguments()
    run(args)
