# scripts/train.py
import os
import sys
import argparse
import multiprocessing
from yolox.core import Trainer
from yolox.exp import get_exp

def make_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--exp_file",        default="exps/vivarium_yolox_tiny.py")
    ap.add_argument("-d", "--devices",         type=int,   default=0)
    ap.add_argument("-b", "--batch_size",      type=int,   default=16)
    ap.add_argument("-n", "--experiment_name", type=str,   default=None)
    ap.add_argument("-c", "--ckpt",            type=str,   default=None)
    ap.add_argument("--fp16",                  action="store_true", default=False)
    ap.add_argument("--resume",                action="store_true", default=False)
    ap.add_argument("--cache",                 type=str,   nargs="?", const="ram", default=None)
    ap.add_argument("--occupy",                action="store_true", default=False)
    ap.add_argument("--logger",                type=str,   default="tensorboard")
    ap.add_argument("--save_history_ckpt",     action="store_true", default=True)
    ap.add_argument("--num_machines",          type=int,   default=1)
    ap.add_argument("--machine_rank",          type=int,   default=0)
    ap.add_argument("--dist_url",              type=str,   default="auto")
    ap.add_argument("opts",                    nargs=argparse.REMAINDER, default=None)
    ap.add_argument("--start_epoch", type=int, default=None)
    return ap.parse_args()

def main():
    args = make_args()
    exp  = get_exp(args.exp_file, exp_name=None)
    exp.merge(args.opts or [])
    if args.experiment_name is None:
        args.experiment_name = exp.exp_name

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()