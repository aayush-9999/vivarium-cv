# exps/vivarium_yolox_tiny.py
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name    = "vivarium_yolox_tiny"
        self.num_classes = 9          # your 9-class scheme
        self.depth       = 0.33       # tiny model depth multiplier
        self.width       = 0.375      # tiny model width multiplier
        self.input_size  = (640, 640)
        self.test_size   = (640, 640)
        self.data_num_workers = 0  
        self.random_size = (10, 20)
        self.max_epoch   = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 15
        self.basic_lr_per_img = 0.01 / 64
        self.data_dir    = "E:\\AI\\vivarium-project\\vivarium-cv\\dataset\\coco"
        self.train_ann   = "train.json"   # YOLOx expects COCO JSON format (see below)
        self.val_ann     = "val.json"
        self.eval_interval = 9999 