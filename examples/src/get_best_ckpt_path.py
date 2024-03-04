import wandb 
import os
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--wandb_entity_name", type=str, default="")
    parser.add_argument("--wandb_project_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, help='model_name + dataset_name')


    args = parser.parse_args()
    os.environ['WANDB_API_KEY'] = args.wandb_api_key


    api = wandb.Api(timeout=180)
    project = api.project(name=args.wandb_project_name)


    sweeps = project.sweeps()
    each_fold_best_ckpt_path = {}
    
    for sweep in sweeps:
        print(sweep.name, sweep.id)
        

        new_cur_sweep = api.sweep(path=f"{args.wandb_entity_name}/{args.wandb_project_name}/{sweep.id}")
        cur_sweep_fold = new_cur_sweep.name[-1]
        cur_sweep_best_run = new_cur_sweep.best_run()
        cur_sweep_best_run_dict = cur_sweep_best_run.summary._json_dict
        
        each_fold_best_ckpt_path[cur_sweep_fold] = cur_sweep_best_run_dict
        

        
    output_file = os.path.join(args.output_dir, 'best_ckpt_path.json')
    json.dump(each_fold_best_ckpt_path, open(output_file, 'w+'), indent=4)
    
        
