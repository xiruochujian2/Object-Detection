# Compress Experiment (For BiliBili魔鬼面具)
### Model:yolov8n.yaml Dataset:Visdrone only using 30% Training Data

```
------------------ train base model ------------------
model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_visdrone/data_exp.yaml',
            cache=True,
            imgsz=640,
            epochs=300,
            batch=32,
            close_mosaic=30,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8n-visdrone',
            )

nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log
```

```
------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-fps.log
------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-fps.log
------------------ lamp exp3 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-lamp-exp3',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-lamp-exp3.log 2>&1 & tail -f logs/yolov8n-lamp-exp3.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp3-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp3-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp3-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-lamp-exp3-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp3-fps.log
```

```
------------------ group-taylor exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '1',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouptaylor-exp1',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-grouptaylor-exp1.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-grouptaylor-exp1-test.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouptaylor-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouptaylor-exp1-fps.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1-fps.log
------------------ group-taylor exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '1',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouptaylor-exp2',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-grouptaylor-exp2.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-grouptaylor-exp2-test.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouptaylor-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouptaylor-exp2-fps.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2-fps.log
```

```
------------------ group-hessian exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 24,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouphessian-exp1',
    
    # prune
    'prune_method':'group_hessian',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-grouphessian-exp1.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-grouphessian-exp1-test.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouphessian-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouphessian-exp1-fps.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1-fps.log
```

```
------------------ slim exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-slim-exp1',
    
    # prune
    'prune_method':'slim',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.04,
    'reg_decay': 0.05,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-slim-exp1.log 2>&1 & tail -f logs/yolov8n-slim-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-slim-exp1-test.log 2>&1 & tail -f logs/yolov8n-slim-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-slim-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-slim-exp1-fps.log 2>&1 & tail -f logs/yolov8n-slim-exp1-fps.log
```

```
------------------ group_sl exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-groupsl-exp1',
    
    # prune
    'prune_method':'group_sl',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.015,
    'reg_decay': 0.05,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-groupsl-exp1.log 2>&1 & tail -f logs/yolov8n-groupsl-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-groupsl-exp1-test.log 2>&1 & tail -f logs/yolov8n-groupsl-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-groupsl-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-groupsl-exp1-fps.log 2>&1 & tail -f logs/yolov8n-groupsl-exp1-fps.log
```

```
------------------ group_slim exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-groupslim-exp1',
    
    # prune
    'prune_method':'group_slim',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.02,
    'reg_decay': 0.05,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-groupslim-exp1.log 2>&1 & tail -f logs/yolov8n-groupslim-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-groupslim-exp1-test.log 2>&1 & tail -f logs/yolov8n-groupslim-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-groupslim-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-groupslim-exp1-fps.log 2>&1 & tail -f logs/yolov8n-groupslim-exp1-fps.log
```

```
python plot_channel_image.py --base-weights runs/prune/yolov8n-visdrone-lamp-exp3-prune/weights/model_c2f_v2.pt --prune-weights runs/prune/yolov8n-visdrone-lamp-exp3-prune/weights/prune.pt
```

### Model:yolov8n-Faster-GFPN-P2-EfficientHead.yaml Dataset:Visdrone

```
------------------ train base model ------------------
model = YOLO('yolov8n-Faster-GFPN-P2-EfficientHead.yaml')
model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_visdrone/data_exp.yaml',
            cache=True,
            imgsz=640,
            epochs=300,
            batch=12,
            close_mosaic=30,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8n-visdrone',
            )

nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log
```

```
------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}
CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-fps.log
```

```
------------------ group-taylor exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 12,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '1',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouptaylor-exp1',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-grouptaylor-exp1.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-grouptaylor-exp1-test.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouptaylor-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouptaylor-exp1-fps.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1-fps.log

------------------ group-taylor exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 12,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '1',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouptaylor-exp2',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-grouptaylor-exp2.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-grouptaylor-exp2-test.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouptaylor-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouptaylor-exp2-fps.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2-fps.log
```

```
------------------ group-hessian exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 12,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouphessian-exp1',
    
    # prune
    'prune_method':'group_hessian',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-grouphessian-exp1.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-grouphessian-exp1-test.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouphessian-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouphessian-exp1-fps.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1-fps.log
```

### Model:yolov8-BIFPN-EfficientRepHead.yaml Dataset:Seaship
```
------------------ train base model ------------------
model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_seaship/data.yaml',
            cache=True,
            imgsz=640,
            epochs=200,
            batch=32,
            close_mosaic=20,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8n',
            )

CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
nohup python get_FPS.py --weights runs/train/yolov8n/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log

------------------ train base-light model ------------------
model = YOLO('yolov8n-BIFPN-EfficientRepHead.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_seaship/data.yaml',
            cache=True,
            imgsz=640,
            epochs=200,
            batch=32,
            close_mosaic=20,
            workers=8,
            device='1',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8n-light',
            )

nohup python train.py > logs/yolov8n-light.log 2>&1 & tail -f logs/yolov8n-light.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-light-test.log 2>&1 & tail -f logs/yolov8n-light-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-light-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-light-fps.log 2>&1 & tail -f logs/yolov8n-light-fps.log
```

```
------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-light/weights/best.pt',
    'data':'/root/data_ssd/dataset_seaship/data.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-light-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-light-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-light-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-light-lamp-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-light-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp1-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-light/weights/best.pt',
    'data':'/root/data_ssd/dataset_seaship/data.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-light-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': False,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-light-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-light-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-light-lamp-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-light-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp2-fps.log
```

### Model:yolov8-ASF-P2.yaml Dataset:Visdrone
```
nohup python get_FPS.py --weights runs/prune/yolov8n-asf-p2-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-asf-p2-fps.log 2>&1 & tail -f logs/yolov8n-asf-p2-fps.log
------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-asf-p2/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 8,
    'workers': 4,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-asf-p2-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-asf-p2-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-asf-p2-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-asf-p2-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-asf-p2-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-asf-p2-lamp-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-asf-p2-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8n-asf-p2-lamp-exp1-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-asf-p2/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 8,
    'workers': 4,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-asf-p2-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 1.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-asf-p2-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-asf-p2-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-asf-p2-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-asf-p2-lamp-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-asf-p2-lamp-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-asf-p2-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8n-asf-p2-lamp-exp2-fps.log

------------------ lamp exp3 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-asf-p2/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 8,
    'workers': 4,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-asf-p2-lamp-exp3',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 1.7,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-asf-p2-lamp-exp3.log 2>&1 & tail -f logs/yolov8n-asf-p2-lamp-exp3.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-asf-p2-lamp-exp3-test.log 2>&1 & tail -f logs/yolov8n-asf-p2-lamp-exp3-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-asf-p2-lamp-exp3-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-asf-p2-lamp-exp3-fps.log 2>&1 & tail -f logs/yolov8n-asf-p2-lamp-exp3-fps.log
```

### Model:yolov8-ASF-P2.yaml Dataset:Visdrone only using 30% Training Data
```
------------------ train base model ------------------
model = YOLO('yolov8-GhostHGNetV2-SlimNeck-ASF.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_visdrone/data_exp.yaml',
            cache=True,
            imgsz=640,
            epochs=250,
            batch=8,
            close_mosaic=20,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8n',
            )

CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log

model = YOLO('yolov8s-GhostHGNetV2-SlimNeck-ASF.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_visdrone/data_exp.yaml',
            cache=True,
            imgsz=640,
            epochs=250,
            batch=8,
            close_mosaic=20,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8s',
            )

CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8s.log 2>&1 & tail -f logs/yolov8s.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8s-test.log 2>&1 & tail -f logs/yolov8s-test.log
nohup python get_FPS.py --weights runs/train/yolov8s/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8s-fps.log 2>&1 & tail -f logs/yolov8s-fps.log
------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-lamp-exp1-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-fps.log
```

### Model:yolov8-convnextv2-goldyolo-asf.yaml Dataset:CrowdHuman only using 20% Training Data
```
------------------ train base model ------------------
model = YOLO('yolov8-convnextv2-goldyolo-asf.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
            cache=True,
            imgsz=640,
            epochs=300,
            batch=16,
            close_mosaic=0,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8-convnextv2-goldyolo-asf',
            )
CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov8-convnextv2-goldyolo-asf-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log

------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-convnextv2-goldyolo-asf/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-convnextv2-goldyolo-asf-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-convnextv2-goldyolo-asf-lamp-exp1-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8-convnextv2-goldyolo-asf-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8-convnextv2-goldyolo-asf-lamp-exp1-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-convnextv2-goldyolo-asf/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-convnextv2-goldyolo-asf-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-lamp-exp2.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-convnextv2-goldyolo-asf-lamp-exp2-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8-convnextv2-goldyolo-asf-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8-convnextv2-goldyolo-asf-lamp-exp2-fps.log
python plot_channel_image.py --base-weights runs/prune/yolov8-convnextv2-goldyolo-asf-lamp-exp2-prune/weights/model_c2f_v2.pt --prune-weights runs/prune/yolov8-convnextv2-goldyolo-asf-lamp-exp2-finetune/weights/best.pt
```

### Model:yolov8-dyhead-prune.yaml Dataset:CrowdHuman only using 20% Training Data
```
------------------ train base model ------------------
model = YOLO('yolov8-dyhead.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
            cache=True,
            imgsz=640,
            epochs=250,
            batch=16,
            close_mosaic=0,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8-DyHead',
            )
CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov8-DyHead-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log

------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-DyHead/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-DyHead-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-DyHead-lamp-exp1-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-DyHead/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-DyHead-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': False,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-DyHead-lamp-exp2-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-DyHead/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-DyHead-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-DyHead-lamp-exp2-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-fps.log

------------------ lamp exp3 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-DyHead/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-DyHead-lamp-exp3',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 3.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-lamp-exp3.log 2>&1 & tail -f logs/yolov8n-lamp-exp3.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp3-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp3-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-DyHead-lamp-exp3-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-lamp-exp3-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp3-fps.log
```

### Model:yolov8-dyhead-prune.yaml Dataset:CrowdHuman only using 20% Training Data
```
------------------ train base model ------------------
model = YOLO('yolov8-repvit-RepNCSPELAN.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
            cache=True,
            imgsz=640,
            epochs=250,
            batch=16,
            close_mosaic=0,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8-repvit-RepNCSPELAN',
            )
CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov8-repvit-RepNCSPELAN-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log

------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-repvit-RepNCSPELAN/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-repvit-RepNCSPELAN-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov8-repvit-RepNCSPELAN-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-repvit-RepNCSPELAN/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-repvit-RepNCSPELAN-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 3.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-repvit-RepNCSPELAN-lamp-exp2-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-fps.log

------------------ lamp exp3 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-repvit-RepNCSPELAN/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-repvit-RepNCSPELAN-lamp-exp3',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 3.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-lamp-exp3.log 2>&1 & tail -f logs/yolov8n-lamp-exp3.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp3-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp3-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-repvit-RepNCSPELAN-lamp-exp3-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-lamp-exp3-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp3-fps.log

------------------ lamp exp4 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-repvit-RepNCSPELAN/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 16,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-repvit-RepNCSPELAN-lamp-exp4',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 4.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp4.log 2>&1 & tail -f logs/yolov8n-lamp-exp4.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp4-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp4-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-repvit-RepNCSPELAN-lamp-exp4-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-lamp-exp4-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp4-fps.log

------------------ slim exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-repvit-RepNCSPELAN/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-repvit-RepNCSPELAN-slim-exp1',
    
    # prune
    'prune_method':'slim',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.008,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-slim-exp1.log 2>&1 & tail -f logs/yolov8n-slim-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-slim-exp1-test.log 2>&1 & tail -f logs/yolov8n-slim-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-repvit-RepNCSPELAN-slim-exp1-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-slim-exp1-fps.log 2>&1 & tail -f logs/yolov8n-slim-exp1-fps.log

------------------ group_taylor exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-repvit-RepNCSPELAN/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-repvit-RepNCSPELAN-group_taylor-exp1',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-group_taylor-exp1.log 2>&1 & tail -f logs/yolov8n-group_taylor-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-group_taylor-exp1-test.log 2>&1 & tail -f logs/yolov8n-group_taylor-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-repvit-RepNCSPELAN-group_taylor-exp1-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-group_taylor-exp1-fps.log 2>&1 & tail -f logs/yolov8n-group_taylor-exp1-fps.log

------------------ group_taylor exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-repvit-RepNCSPELAN/weights/best.pt',
    'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-repvit-RepNCSPELAN-group_taylor-exp2',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-group_taylor-exp2.log 2>&1 & tail -f logs/yolov8n-group_taylor-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-group_taylor-exp2-test.log 2>&1 & tail -f logs/yolov8n-group_taylor-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8-repvit-RepNCSPELAN-group_taylor-exp2-finetune/weights/best.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-group_taylor-exp2-fps.log 2>&1 & tail -f logs/yolov8n-group_taylor-exp2-fps.log

python plot_channel_image.py --base-weights runs/prune/yolov8-repvit-RepNCSPELAN-lamp-exp4-prune/weights/model_c2f_v2.pt --prune-weights runs/prune/yolov8-repvit-RepNCSPELAN-lamp-exp4-finetune/weights/best.pt

```

### Model:yolov8-starnet-C2f-Star-LSCD Dataset:SeaShip
```
------------------ train base model ------------------
model = YOLO('ultralytics/cfg/models/v8/yolov8-starnet-C2f-Star-LSCD.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/dataset/dataset_seaship/data.yaml',
            cache=True,
            imgsz=640,
            epochs=300,
            batch=32,
            close_mosaic=0,
            workers=16,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False, # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8-starnet-C2f-Star-LSCD',
            )

CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov8-starnet-C2f-Star-LSCD-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log

------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-starnet-C2f-Star-LSCD/weights/best.pt',
    'data':'/root/dataset/dataset_seaship/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-starnet-C2f-Star-LSCD-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov8-starnet-C2f-Star-LSCD-lamp-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8-starnet-C2f-Star-LSCD-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8-starnet-C2f-Star-LSCD-lamp-exp1-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-starnet-C2f-Star-LSCD/weights/best.pt',
    'data':'/root/dataset/dataset_seaship/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-starnet-C2f-Star-LSCD-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 3.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov8-starnet-C2f-Star-LSCD-lamp-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8-starnet-C2f-Star-LSCD-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8-starnet-C2f-Star-LSCD-lamp-exp2-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8-starnet-C2f-Star-LSCD/weights/best.pt',
    'data':'/root/dataset/dataset_seaship/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov8-starnet-C2f-Star-LSCD-lamp-exp3',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp3.log 2>&1 & tail -f logs/yolov8n-lamp-exp3.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp3-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp3-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov8-starnet-C2f-Star-LSCD-lamp-exp3-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8-starnet-C2f-Star-LSCD-lamp-exp3-fps.log 2>&1 & tail -f logs/yolov8-starnet-C2f-Star-LSCD-lamp-exp3-fps.log
```

### Model:yolov10n.yaml Dataset:Visdrone
```
------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov10n-visdrone/weights/best.pt',
    'data':'/home/hjj/Desktop/dataset/dataset_visdrone/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov10n-visdrone-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov10n-lamp-exp1.log 2>&1 & tail -f logs/yolov10n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov10n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov10n-lamp-exp1-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov10n-visdrone-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov10n-fps.log 2>&1 & tail -f logs/yolov10n-fps.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov10n-visdrone-lamp-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov10n-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov10n-lamp-exp1-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov10n-visdrone/weights/best.pt',
    'data':'/home/hjj/Desktop/dataset/dataset_visdrone/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov10n-visdrone-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov10n-lamp-exp2.log 2>&1 & tail -f logs/yolov10n-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov10n-lamp-exp2-test.log 2>&1 & tail -f logs/yolov10n-lamp-exp2-test.log
CUDA_VISIBLE_DEVICES=0 nohup python get_FPS.py --weights runs/prune/yolov10n-visdrone-lamp-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov10n-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov10n-lamp-exp2-fps.log
```

# 使用教程
    剪枝操作问题，报错问题统一群里问，我群里回复谢谢~

# 环境

    pip install torch-pruning==1.5.1 tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

## 视频

里面的使用教程.zip(第一个必须要看的视频):
链接: https://pan.baidu.com/s/1k6Mj2LP_tXa-oqLkB_pPGQ?pwd=6fgd 
提取码: 6fgd # BiliBili 魔傀面具
1. 20240114 增加一个c2f_v2转回去c2f的教程视频(transform_weight.py)(需要对剪枝后的模型做蒸馏学习的必看!)
2. 20240316 增加一个稀疏训练tensorboard结果可视化
3. 20240415 增加segment,pose,obb剪枝说明
4. 20240418 增加一个obb剪枝额外注意点的视频(obb剪枝必看！)

--------------------------------------------------------------

通过网盘分享的文件：yolov8-Faster-GFPN-P2-EfficientHead视频说明.zip等15个文件
链接: https://pan.baidu.com/s/1uUZSOiFrh9mdQzZqgKa2hA?pwd=rdzb 
提取码: rdzb # BiliBili 魔傀面具
1. yolov8-Faster-GFPN-P2-EfficientHead教程(更正:里面的不是GFPN,是YOLOV6的EfficientRepBiPAN)
2. yolov8-BIFPN-EfficientRepHead教程
3. EfficientHead中的PConv跳层教程
4. yolov8-GhostHGNetV2-SlimNeck-ASF教程
5. yolov8-convnextv2-goldyolo-asf.yaml教程
6. yolov8-dyhead.yaml教程
7. yolov8-repvit-RepNCSPELAN.yaml教程
8. yolov8-starnet-C2f-Star-LSCD.yaml教程
9. yolov10n.yaml教程
10. yolov11n.yaml教程
11. yolo11-C3k2-Faster-CGLU-RSCD教程
12. yolo11-FDPN-TADDH教程
13. yolo11-C3k2-MutilScaleEdgeInformationEnhance教程
14. yolo11-SOEP教程
15. yolo12教程

--------------------------------------------------------------

yolov5v7的示例讲解(主要是增加剪枝跳层的理解):
链接：https://pan.baidu.com/s/11vI2YIgd8JLCmaIwanzVew?pwd=img8 
提取码：img8  BiliBili 魔鬼面具
1. yolov5n+C3-Faster+RepConv
2. yolov5n+RepViT+C2f
3. 原yolov7-tiny, yolov7-tiny+mobilenetv3+LSKBlock+TSCODE+RepConv, yolov7-tiny+Yolov7_Tiny_E_ELAN_DCN+AFPN
4. yolov7-tiny+FasterNet+DBB
5. yolov7-tiny+ReXNet+VoVGSCSP+DyHead+DecoupledHead

## 常见问题与报错
1. speedup是什么参数，修改其会对剪枝有什么影响？

    speedup决定剪枝率,speedup=剪枝前GFLOPS/剪枝后GFLOPS.比如speedup=2.0,相当于压缩百分之50的计算量.

2. TypeError: object of type 'NoneType' has no len()

    尝试关闭全局剪枝(global_pruning).

3. 程序在剪枝开始前卡主一直不动,或者等了很久后报内存溢出的问题.

    有模块剪不了,如果你添加了多个模块,需要一个一个模块去排查,排查到哪个模块导致的,需要把其换掉.