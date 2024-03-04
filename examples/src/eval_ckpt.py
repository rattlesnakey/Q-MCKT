import argparse
import numpy as np
import json
from collections import defaultdict
import pandas as pd
import os
import wandb
import sys



def get_prediction_result(cur_ckpt_dir):
    all_fold_result = defaultdict(list)
    cur_py_dir = os.path.dirname(sys.argv[0])
    cur_fold_test_result_file = os.path.join(cur_ckpt_dir, 'test_result.json')
        
    if os.path.exists(cur_fold_test_result_file):
        pass 
    else:
        cmd = f"python -u {cur_py_dir}/wandb_predict.py --use_wandb 0 --save_dir {cur_ckpt_dir} --bz {args.bz}"
        if os.system(cmd):
            raise ValueError('running wandb_predict unsucessfully !')
    
    print('done')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    

    args = parser.parse_args()    
    get_prediction_result(args.ckpt_dir)
