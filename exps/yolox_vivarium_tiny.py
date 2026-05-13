# exps/yolox_vivarium_tiny.py
"""
YOLOX experiment config — 13-class vivarium detector.

Class scheme:
    0   mouse
    1   water_critical   (0–15%)
    2   water_low        (15–35%)
    3   water_ok         (35–80%)
    4   water_full       (80–100%)
    5   food_critical    (0–15%)
    6   food_low         (15–35%)
    7   food_ok          (35–80%)
    8   food_full        (80–100%)
    9   bedding_worst
    10  bedding_bad
    11  bedding_ok
    12  bedding_perfect
"""
import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name         = "yolox_vivarium_tiny"
        self.num_classes      = 13
        self.depth            = 0.33
        self.width            = 0.375
        self.input_size       = (416, 416)   # ← reduced from 640 (big VRAM saving)
        self.test_size        = (416, 416)   # ← match input_size
        self.data_num_workers = 0
        self.random_size      = (10, 20)
        self.max_epoch        = 100        
        self.warmup_epochs    = 3
        self.no_aug_epochs    = 10
        self.basic_lr_per_img = 0.01 / 64
        self.data_dir         = r"E:\AI\vivarium-project\vivarium-cv\dataset\coco"
        self.train_ann        = "train.json"
        self.val_ann          = "val.json"
        self.eval_interval    = 5
        self.test_conf        = 0.25
        self.nmsthre          = 0.45
        self.output_dir       = "YOLOX_outputs"
        self.batch_size       = 4  

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir   = self.data_dir,
            json_file  = self.train_ann,
            name       = "train2017",
            img_size   = self.input_size,
            preproc    = TrainTransform(
                max_labels = 50,
                flip_prob  = 0.5,
                hsv_prob   = 1.0,
            ),
            cache      = cache,
            cache_type = cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(
            data_dir  = self.data_dir,
            json_file = self.val_ann,
            name      = "val2017",
            img_size  = self.test_size,
            preproc   = ValTransform(legacy=False),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator
        from pycocotools.cocoeval import COCOeval as _COCOeval
        import yolox.layers.fast_coco_eval_api as _fast_api
        _fast_api.COCOeval_opt = _COCOeval

        return COCOEvaluator(
            dataloader  = self.get_eval_loader(
                batch_size, is_distributed, testdev=testdev, legacy=legacy,
            ),
            img_size    = self.test_size,
            confthre    = self.test_conf,
            nmsthre     = self.nmsthre,
            num_classes = self.num_classes,
            testdev     = testdev,
        )