import os
import argparse
import json
import copy
import torch
import pandas as pd
import sys
sys.path.insert(0, '../..')
from pykt.models import evaluate,evaluate_question,load_model
from pykt.datasets import init_test_datasets

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

#! wandb 是我账号的 key 
with open("../../configs/wandb.json") as fin:
    wandb_config = json.load(fin)

def main(params):
    if params['use_wandb'] ==1:
        import wandb
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        #! 自动创建 wandb
        wandb.init(project="wandb_predict")

    #! 在实际的代码中，是两种 fusion 的策略都有去尝试的
    #! 所以这里 fusion type 是个 list
    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    with open(os.path.join(save_dir, "config.json")) as fin:
        #! 读取 ckpt 的 config 
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]    
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len   

    with open("../../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]    
    if model_name not in ["dimkt"]:        
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)
    else:
        #! 读取各个数据集对应的 data config 
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))                

    if model.model_name == "rkt":
        testauc, testacc = evaluate(model, test_loader, model_name, rel, save_test_path)
    else:
        #! 直接普通的 evaluate, 就是给 model 和 test data loader
        testauc, testacc = evaluate(model, test_loader, model_name, save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
    if model.model_name == "rkt":
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, rel, save_test_window_path)
    else:
        #! 也是用的 evaluate 函数，只不过传的是 test_window_dataloader
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_test_window_path)
    print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    # question_testauc, question_testacc = -1, -1
    # question_window_testauc, question_window_testacc = -1, -1
    #! 这边就有 window 和 非 window 的 acc 了
    #! 上面这边都是 kc level 的 test
    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }  

    #! 这边是 test question level
    #! 也是 window 和非 window 两种
    q_testaucs, q_testaccs = -1,-1
    qw_testaucs, qw_testaccs = -1,-1
    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
        #! 因为 train with kc, test with question 是需要传 fusion type 的
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]
    
    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, model.emb_type+"_test_question_window_predictions.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc"+key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc"+key] = qw_testaccs[key]

        
    # print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
    # print(f"question_testauc: {question_testauc}, question_testacc: {question_testacc}, question_window_testauc: {question_window_testauc}, question_window_testacc: {question_window_testacc}")
    
    print(dres)
    #! 我自己加的 dump 结果下来
    json.dump(dres, open(os.path.join(save_dir, 'test_result.json'), 'w+'), indent=4)
    raw_config = json.load(open(os.path.join(save_dir,"config.json")))
    dres.update(raw_config['params'])

    if params['use_wandb'] ==1:
        wandb.log(dres)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=1)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
