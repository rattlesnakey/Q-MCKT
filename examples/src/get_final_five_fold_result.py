import argparse
import numpy as np
import json
from collections import defaultdict
import pandas as pd
import os
import wandb
import sys



def extract_wandb_best_ckpt_path(args):
    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    api = wandb.Api(timeout=180)
    project = api.project(name=args.wandb_project_name)

    output_file = os.path.join(args.output_dir, 'best_ckpt_path.json')
    if os.path.exists(output_file):
        each_fold_best_ckpt_path = json.load(open(output_file))
    else:
        sweeps = project.sweeps()
        each_fold_best_ckpt_path = {}
        
        for sweep in sweeps:
            print(sweep.name, sweep.id)
            

            new_cur_sweep = api.sweep(path=f"{args.wandb_entity_name}/{args.wandb_project_name}/{sweep.id}")
            cur_sweep_fold = new_cur_sweep.name[-1]

            cur_sweep_best_run = new_cur_sweep.best_run()
            cur_sweep_best_run_dict = cur_sweep_best_run.summary._json_dict
            each_fold_best_ckpt_path[cur_sweep_fold] = cur_sweep_best_run_dict

        
        json.dump(each_fold_best_ckpt_path, open(output_file, 'w+'), indent=4)
    return each_fold_best_ckpt_path

def get_prediction_result(each_fold_best_ckpt_path=None, saved_model_dir=None):
    all_fold_result = defaultdict(list)
    cur_py_dir = os.path.dirname(sys.argv[0])
    # import pdb; pdb.set_trace()

    if each_fold_best_ckpt_path:

        for fold, fold_dict in each_fold_best_ckpt_path.items():

            ckpt_dir = os.path.dirname(fold_dict['model_save_path'])
            cur_fold_test_result_file = os.path.join(ckpt_dir, 'test_result.json')
            
            if os.path.exists(cur_fold_test_result_file):
                pass 
            else:
                cmd = f"python -u {cur_py_dir}/wandb_predict.py --use_wandb 0 --save_dir {ckpt_dir} --bz {args.bz}"
                if os.system(cmd):
                    raise ValueError('running wandb_predict unsucessfully !')
            
            cur_fold_test_result = json.load(open(cur_fold_test_result_file))
            all_fold_result['fold'].append(fold)
            
            for metric_name, metric_value in cur_fold_test_result.items():
                all_fold_result[metric_name].append(metric_value)
    else:

        ckpt_dirs = os.listdir(saved_model_dir)
        for ckpt_dir in ckpt_dirs:
            cur_ckpt_dir = os.path.join(saved_model_dir, ckpt_dir)

            cur_fold_test_result_file = os.path.join(cur_ckpt_dir, 'test_result.json')
            
            if os.path.exists(cur_fold_test_result_file):
                pass 
            else:

                cmd = f"python -u {cur_py_dir}/wandb_predict.py --use_wandb 0 --save_dir {cur_ckpt_dir} --bz {args.bz}"
                if os.system(cmd):
                    raise ValueError('running wandb_predict unsucessfully !')
            
            cur_fold_test_result = json.load(open(cur_fold_test_result_file))
            
            for metric_name, metric_value in cur_fold_test_result.items():
                all_fold_result[metric_name].append(metric_value)
            
        
    
    return pd.DataFrame(all_fold_result)
        


def get_final_result(all_res, print_std=True):

    final_result = {}
    repeated_aucs = np.unique(all_res["testauc"].values)
    repeated_accs = np.unique(all_res["testacc"].values)
    repeated_window_aucs = np.unique(all_res["window_testauc"].values)
    repeated_window_accs = np.unique(all_res["window_testacc"].values)
    repeated_auc_mean, repeated_auc_std = np.mean(repeated_aucs), np.std(repeated_aucs, ddof=0)
    repeated_acc_mean, repeated_acc_std = np.mean(repeated_accs), np.std(repeated_accs, ddof=0)
    repeated_winauc_mean, repeated_winauc_std = np.mean(repeated_window_aucs), np.std(repeated_window_aucs, ddof=0)
    repeated_winacc_mean, repeated_winacc_std = np.mean(repeated_window_accs), np.std(repeated_window_accs, ddof=0)
    # key = dataset_name + "_" + model_name
    if print_std:
        print("_repeated:", "%.4f"%repeated_auc_mean + "±" + "%.4f"%repeated_auc_std + "," + "%.4f"%repeated_acc_mean + "±" + "%.4f"%repeated_acc_std + "," + "%.4f"%repeated_winauc_mean + "±" + "%.4f"%repeated_winauc_std + "," + "%.4f"%repeated_winacc_mean + "±" + "%.4f"%repeated_winacc_std) 
    else:
        print("_repeated:", "%.4f"%repeated_auc_mean + "," + "%.4f"%repeated_acc_mean + "," + "%.4f"%repeated_winauc_mean + "," + "%.4f"%repeated_winacc_mean)
    # import pdb;
    final_result['repeated_auc_mean'] = repeated_auc_mean
    final_result['repeated_auc_std'] = repeated_auc_std
    final_result['repeated_acc_mean'] = repeated_acc_mean
    final_result['repeated_acc_std'] = repeated_acc_std
    final_result['repeated_auc'] = "%.4f"%repeated_auc_mean + "±" + "%.4f"%repeated_auc_std
    final_result['repeated_acc'] = "%.4f"%repeated_acc_mean + "±" + "%.4f"%repeated_acc_std
    final_result['repeated_winauc'] = "%.4f"%repeated_winauc_mean + "±" + "%.4f"%repeated_winauc_std
    final_result['repeated_winacc'] = "%.4f"%repeated_winacc_mean + "±" + "%.4f"%repeated_winacc_std
    
    try: 

        question_aucs = np.unique(all_res["oriaucconcepts"].values)
        question_accs = np.unique(all_res["oriaccconcepts"].values)
        question_window_aucs = np.unique(all_res["windowaucconcepts"].values)
        question_window_accs = np.unique(all_res["windowaccconcepts"].values)
        question_auc_mean, question_auc_std = np.mean(question_aucs), np.std(question_aucs, ddof=0)
        question_acc_mean, question_acc_std = np.mean(question_accs), np.std(question_accs, ddof=0)
        question_winauc_mean, question_winauc_std = np.mean(question_window_aucs), np.std(question_window_aucs, ddof=0)
        question_winacc_mean, question_winacc_std = np.mean(question_window_accs), np.std(question_window_accs, ddof=0)
        # key = dataset_name + "_" + model_name
        if print_std:
            print("_concepts:", "%.4f"%question_auc_mean + "±" + "%.4f"%question_auc_std + "," + "%.4f"%question_acc_mean + "±" + "%.4f"%question_acc_std + "," + "%.4f"%question_winauc_mean + "±" + "%.4f"%question_winauc_std + "," + "%.4f"%question_winacc_mean + "±" + "%.4f"%question_winacc_std) 
        else:
            print("_concepts:", "%.4f"%question_auc_mean + "," + "%.4f"%question_acc_mean + "," + "%.4f"%question_winauc_mean + "," + "%.4f"%question_winacc_mean) 
        final_result['question_auc_mean'] = question_auc_mean
        final_result['question_auc_std'] = question_auc_std
        final_result['question_acc_mean'] = question_acc_mean
        final_result['question_acc_std'] = question_acc_std
        final_result['question_auc'] = "%.4f"%question_auc_mean + "±" + "%.4f"%question_auc_std
        final_result['question_acc'] = "%.4f"%question_acc_mean + "±" + "%.4f"%question_acc_std
        final_result['question_winauc'] = "%.4f"%question_winauc_mean + "±" + "%.4f"%question_winauc_std
        final_result['question_winacc'] = "%.4f"%question_winacc_mean + "±" + "%.4f"%question_winacc_std
    
    except:
        print(f"don't have question tag!!!")
        return final_result

    try:
        early_aucs = np.unique(all_res["oriaucearly_preds"].values)
        early_accs = np.unique(all_res["oriaccearly_preds"].values)
        early_window_aucs = np.unique(all_res["windowaucearly_preds"].values)
        early_window_accs = np.unique(all_res["windowaccearly_preds"].values)
        early_auc_mean, early_auc_std = np.mean(early_aucs), np.std(early_aucs, ddof=0)
        early_acc_mean, early_acc_std = np.mean(early_accs), np.std(early_accs, ddof=0)
        early_winauc_mean, early_winauc_std = np.mean(early_window_aucs), np.std(early_window_aucs, ddof=0)
        early_winacc_mean, early_winacc_std = np.mean(early_window_accs), np.std(early_window_accs, ddof=0)
        # key = dataset_name + "_" + model_name
        if print_std:
            print("_early:", "%.4f"%early_auc_mean + "±" + "%.4f"%early_auc_std + "," + "%.4f"%early_acc_mean + "±" + "%.4f"%early_acc_std + "," + "%.4f"%early_winauc_mean + "±" + "%.4f"%early_winauc_std + "," + "%.4f"%early_winacc_mean + "±" + "%.4f"%early_winacc_std)
        else:
            print("_early:", "%.4f"%early_auc_mean + "," + "%.4f"%early_acc_mean + "," + "%.4f"%early_winauc_mean + "," + "%.4f"%early_winacc_mean)         
        
        final_result['early_auc_mean'] = early_auc_mean
        final_result['early_auc_std'] = early_auc_std
        final_result['early_acc_mean'] = early_acc_mean
        final_result['early_acc_std'] = early_acc_std
        final_result['early_auc'] = "%.4f"%early_auc_mean + "±" + "%.4f"%early_auc_std
        final_result['early_acc'] = "%.4f"%early_acc_mean + "±" + "%.4f"%early_acc_std
        final_result['early_winauc'] = "%.4f"%early_winauc_mean + "±" + "%.4f"%early_winauc_std
        final_result['early_winacc'] = "%.4f"%early_winacc_mean + "±" + "%.4f"%early_winacc_std
    except:
        print(f"don't have early fusion!!!")


    late_mean_aucs = np.unique(all_res["oriauclate_mean"].values)
    late_mean_accs = np.unique(all_res["oriacclate_mean"].values)
    late_mean_window_aucs = np.unique(all_res["windowauclate_mean"].values)
    late_mean_window_accs = np.unique(all_res["windowacclate_mean"].values)
    latemean_auc_mean, latemean_auc_std = np.mean(late_mean_aucs), np.std(late_mean_aucs, ddof=0)
    latemean_acc_mean, latemean_acc_std = np.mean(late_mean_accs), np.std(late_mean_accs, ddof=0)
    latemean_winauc_mean, latemean_winauc_std = np.mean(late_mean_window_aucs), np.std(late_mean_window_aucs, ddof=0)
    latemean_winacc_mean, latemean_winacc_std = np.mean(late_mean_window_accs), np.std(late_mean_window_accs, ddof=0)
    
    final_result['late_mean_auc_mean'] = late_mean_auc_mean
    final_result['late_mean_auc_std'] = late_mean_auc_std
    final_result['late_mean_acc_mean'] = late_mean_acc_mean
    final_result['late_mean_acc_std'] = late_mean_acc_std
    final_result['late_mean_auc'] = "%.4f"%late_mean_auc_mean + "±" + "%.4f"%late_mean_auc_std
    final_result['late_mean_acc'] = "%.4f"%late_mean_acc_mean + "±" + "%.4f"%late_mean_acc_std
    final_result['late_mean_winauc'] = "%.4f"%late_mean_winauc_mean + "±" + "%.4f"%late_mean_winauc_std
    final_result['late_mean_winacc'] = "%.4f"%late_mean_winacc_mean + "±" + "%.4f"%late_mean_winacc_std
    
    # key = dataset_name + "_" + model_name
    if print_std:
        print("_latemean:", "%.4f"%late_mean_auc_mean + "±" + "%.4f"%late_mean_auc_std + "," + "%.4f"%late_mean_acc_mean + "±" + "%.4f"%late_mean_acc_std + "," + "%.4f"%late_mean_winauc_mean + "±" + "%.4f"%late_mean_winauc_std + "," + "%.4f"%late_mean_winacc_mean + "±" + "%.4f"%late_mean_winacc_std)
    else:
        print("_latemean:", "%.4f"%late_mean_auc_mean + "," + "%.4f"%late_mean_acc_mean + "," + "%.4f"%late_mean_winauc_mean + "," + "%.4f"%late_mean_winacc_mean)

    late_vote_aucs = np.unique(all_res["oriauclate_vote"].values)
    late_vote_accs = np.unique(all_res["oriacclate_vote"].values)
    late_vote_window_aucs = np.unique(all_res["windowauclate_vote"].values)
    late_vote_window_accs = np.unique(all_res["windowacclate_vote"].values)
    latevote_auc_mean, latevote_auc_std = np.mean(late_vote_aucs), np.std(late_vote_aucs, ddof=0)
    latevote_acc_mean, latevote_acc_std = np.mean(late_vote_accs), np.std(late_vote_accs, ddof=0)
    latevote_winauc_mean, latevote_winauc_std = np.mean(late_vote_window_aucs), np.std(late_vote_window_aucs, ddof=0)
    latevote_winacc_mean, latevote_winacc_std = np.mean(late_vote_window_accs), np.std(late_vote_window_accs, ddof=0)
    
    final_result['latevote_auc_mean'] = latevote_auc_mean
    final_result['latevote_auc_std'] = latevote_auc_std
    final_result['latevote_acc_mean'] = latevote_acc_mean
    final_result['latevote_acc_std'] = latevote_acc_std
    final_result['latevote_auc'] = "%.4f"%latevote_auc_mean + "±" + "%.4f"%latevote_auc_std
    final_result['latevote_acc'] = "%.4f"%latevote_acc_mean + "±" + "%.4f"%latevote_acc_std
    final_result['latevote_winauc'] = "%.4f"%latevote_winauc_mean + "±" + "%.4f"%latevote_winauc_std
    final_result['latevote_winacc'] = "%.4f"%latevote_winacc_mean + "±" + "%.4f"%latevote_winacc_std
    # key = dataset_name + "_" + model_name
    if print_std:
        print("_latevote:", "%.4f"%latevote_auc_mean + "±" + "%.4f"%latevote_auc_std + "," + "%.4f"%latevote_acc_mean + "±" + "%.4f"%latevote_acc_std + "," + "%.4f"%latevote_winauc_mean + "±" + "%.4f"%latevote_winauc_std + "," + "%.4f"%latevote_winacc_mean + "±" + "%.4f"%latevote_winacc_std)
    else:
        print("_latevote:", "%.4f"%latevote_auc_mean + "," + "%.4f"%latevote_acc_mean + "," + "%.4f"%latevote_winauc_mean + "," + "%.4f"%latevote_winacc_mean)

    late_all_aucs = np.unique(all_res["oriauclate_all"].values)
    late_all_accs = np.unique(all_res["oriacclate_all"].values)
    late_all_window_aucs = np.unique(all_res["windowauclate_all"].values)
    late_all_window_accs = np.unique(all_res["windowacclate_all"].values)
    lateall_auc_mean, lateall_auc_std = np.mean(late_all_aucs), np.std(late_all_aucs, ddof=0)
    lateall_acc_mean, lateall_acc_std = np.mean(late_all_accs), np.std(late_all_accs, ddof=0)
    lateall_winauc_mean, lateall_winauc_std = np.mean(late_all_window_aucs), np.std(late_all_window_aucs, ddof=0)
    lateall_winacc_mean, lateall_winacc_std = np.mean(late_all_window_accs), np.std(late_all_window_accs, ddof=0)
    
    final_result['lateall_auc_mean'] = lateall_auc_mean
    final_result['lateall_auc_std'] = lateall_auc_std
    final_result['lateall_acc_mean'] = lateall_acc_mean
    final_result['lateall_acc_std'] = lateall_acc_std
    final_result['lateall_auc'] = "%.4f"%lateall_auc_mean + "±" + "%.4f"%lateall_auc_std
    final_result['lateall_acc'] = "%.4f"%lateall_acc_mean + "±" + "%.4f"%lateall_acc_std
    final_result['lateall_winauc'] = "%.4f"%lateall_winauc_mean + "±" + "%.4f"%lateall_winauc_std
    final_result['lateall_winacc'] = "%.4f"%lateall_winacc_mean + "±" + "%.4f"%lateall_winacc_std
    

    if print_std:
        print("_lateall:", "%.4f"%lateall_auc_mean + "±" + "%.4f"%lateall_auc_std + "," + "%.4f"%lateall_acc_mean + "±" + "%.4f"%lateall_acc_std + "," + "%.4f"%lateall_winauc_mean + "±" + "%.4f"%lateall_winauc_std + "," + "%.4f"%lateall_winacc_mean + "±" + "%.4f"%lateall_winacc_std)
    else:
        print("_lateall:", "%.4f"%lateall_auc_mean + "," + "%.4f"%lateall_acc_mean + "," + "%.4f"%lateall_winauc_mean + "," + "%.4f"%lateall_winacc_mean)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    

    parser.add_argument("--use_wandb", type=int, default=1)

    parser.add_argument("--saved_model_dir", type=str, default="")
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--wandb_entity_name", type=str, default="")
    parser.add_argument("--wandb_project_name", type=str, default="")

    parser.add_argument("--output_dir", type=str, help='model_name + dataset_name')


    args = parser.parse_args()
    
    if args.use_wandb:
        each_fold_best_ckpt_path = extract_wandb_best_ckpt_path(args)
        prediction_result = get_prediction_result(each_fold_best_ckpt_path=each_fold_best_ckpt_path)
    else:
        prediction_result = get_prediction_result(saved_model_dir=args.saved_model_dir)
    
    final_result = get_final_result(prediction_result)
    print(final_result)
    json.dump(final_result, open(os.path.join(args.output_dir, 'final_result.json'), 'w+'), indent=4, ensure_ascii=False)
    
    