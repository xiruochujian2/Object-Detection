import warnings
warnings.filterwarnings('ignore')
import os
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune
# from ultralytics.models.yolo.segment.compress import SegmentationCompressor, SegmentationFinetune
# from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune
# from ultralytics.models.yolo.obb.compress import OBBCompressor, OBBFinetune



def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    # compressor = SegmentationCompressor(overrides=param_dict)
    # compressor = PoseCompressor(overrides=param_dict)
    # compressor = OBBCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    # trainer = SegmentationFinetune(overrides=param_dict)
    # trainer = PoseFinetune(overrides=param_dict)
    # trainer = OBBFinetune(overrides=param_dict)
    trainer.train()
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support() # Windows多进程保护 # 训练代码
    param_dict = {
        # origin
        'model': 'runs/train/yolo11-efficientViT-lstmSimpleAlign-pkm-224-ewtblock1-C2TSSA(ffnattn)-u/weights/best.pt',
        'data':'dataset/data.yaml',
        'imgsz': 640,
        'epochs': 150,
        'batch': 4,
        'workers': 0,
        'cache': False,
        'optimizer': 'SGD',
        'device': 0,
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'yolo11-efficientViT-lstmdcn-pkm-ewtblock1-C2TSSA(ffnattn)-u-61.6',
        'amp': False,
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 2 ,  #计算量剪枝到50%
        'reg': 0.0005,
        'sl_epochs': 500,  #稀疏训练的epoch
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)