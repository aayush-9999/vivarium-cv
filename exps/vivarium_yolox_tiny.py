# exps/vivarium_yolox_tiny.py
import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name         = "vivarium_yolox_tiny"
        self.num_classes      = 9
        self.depth            = 0.33
        self.width            = 0.375
        self.input_size       = (640, 640)
        self.test_size        = (640, 640)
        self.data_num_workers = 0
        self.random_size      = (10, 20)
        self.max_epoch        = 100
        self.warmup_epochs    = 5
        self.no_aug_epochs    = 15
        self.basic_lr_per_img = 0.01 / 64
        self.data_dir         = "E:\\AI\\vivarium-project\\vivarium-cv\\dataset\\coco"
        self.train_ann        = "train.json"
        self.val_ann          = "val.json"
        self.eval_interval    = 55
        self.test_conf        = 0.35
        self.nmsthre          = 0.45

        self.output_dir       = "YOLOX_outputs"

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir   = self.data_dir,
            json_file  = self.train_ann,      # ← just "train.json", no annotations/ prefix
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
            json_file = self.val_ann,         # ← just "val.json", no annotations/ prefix
            name      = "val2017",
            img_size  = self.test_size,
            preproc   = ValTransform(legacy=False),
        )
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        # Fix: replace fast C++ COCO eval with standard pycocotools
        # (fast eval requires Visual Studio cl.exe which is not available)
        import yolox.layers.fast_coco_eval_api as fast_api
        from pycocotools.cocoeval import COCOeval
        fast_api.COCOeval_opt = COCOeval

        return COCOEvaluator(
            dataloader  = self.get_eval_loader(
                batch_size,
                is_distributed,
                testdev=testdev,
                legacy=legacy,
            ),
            img_size    = self.test_size,
            confthre    = self.test_conf,
            nmsthre     = self.nmsthre,
            num_classes = self.num_classes,
            testdev     = testdev,
        )