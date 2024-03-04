#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
# from torch.cuda import FloatTensor, LongTensor
from torch import FloatTensor, LongTensor
import numpy as np
import json
from collections import defaultdict

class KTQueDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """

    def __init__(self, file_path, input_type, folds,concept_num,max_concepts, qtest=False, data_config=None):
        super(KTQueDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        if "questions" not in input_type or "concepts" not in input_type:
            raise("The input types must contain both questions and concepts")

        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        self.save_path_prefix = file_path + folds_str
        processed_data = self.save_path_prefix + "_qlevel.pkl"
        self.q_pair_file = self.save_path_prefix + '_q_pos_neg.json'
        self.data_config = data_config
        

        if os.path.exists(processed_data) and os.path.exists(self.q_pair_file):
            print(f"Read data from processed file: {processed_data}")
            
            self.dori = pd.read_pickle(processed_data)
            #! 只有 train 数据集才有多个 fold
            if len(folds) > 1:
                print(f"Read pos neg mapping from file: {self.q_pair_file}")
                self.q_pos_neg = json.load(open(self.q_pair_file, 'r'))
            else:
                self.q_pos_neg = None
        else:
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            self.dori, self.q_pos_neg = self.__load_data__(sequence_path, folds)
            save_data = self.dori
            pd.to_pickle(save_data, processed_data)
            if self.q_pos_neg != None:
                json.dump(self.q_pos_neg, open(self.q_pair_file, 'w+'), ensure_ascii=False)
                
            
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            if key=='cseqs':
                seqs = self.dori[key][index][:-1,:]
                shft_seqs = self.dori[key][index][1:,:]
            else:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        return dcur

    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns: 
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}

        df = pd.read_csv(sequence_path)

        df = df[df["fold"].isin(folds)].copy()#[0:1000]
        interaction_num = 0
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:

                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))

                    row_skills.append(skills)

                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)
        
        q_pair = None
      
        if len(folds) > 1:
        
            print('getting question positive and negative ...')
            qseqs = dori['qseqs']; cseqs = dori['cseqs']; rseqs = dori['rseqs']
            
            kc_q_mapping = {}
            kc_q_level_mapping = defaultdict(list)
            q_acc = {}
            q_kc_mapping = {}
            q_pair = {}
    
            for qseq, cseq, rseq in zip(qseqs, cseqs, rseqs):
                for q, c, r in zip(qseq, cseq, rseq):
                    if q == -1:
                        continue
                    c = sorted(c)
                    c_str = '_'.join(map(str, c))
                    if c_str not in kc_q_mapping:
                        kc_q_mapping[c_str] = set()
                        
                    kc_q_mapping[c_str].add(q)
                    
                    if q not in q_kc_mapping:
                        q_kc_mapping[q] = c_str
             
                    
                    if q not in q_acc:
                        q_acc[q] = {}
                        q_acc[q]['correct'] = 0
                        q_acc[q]['total'] = 0
                    
                    
                    if r == 1:
                        q_acc[q]['correct'] += 1
                        
                    q_acc[q]['total'] += 1
            
           
            pos_neg_cand = set()
            correct_ratio_level = self.data_config["correct_ratio_level"]
            correct_ratio_level.append(1)
            for k, v in q_acc.items():
                   
                if v['total'] >= self.data_config['question_frequency_threshold']:
                    pos_neg_cand.add(k)
                
                
                correct_ratio = v['correct'] / v['total']
                v['correct_ratio'] = correct_ratio
                
                
                prev_level = 0
                
                for i, cur_level in enumerate(correct_ratio_level):
                    if correct_ratio == 0:
                        v['level'] = 'A'
                        
                    elif correct_ratio > prev_level and correct_ratio <= cur_level:
                        v['level'] = chr(ord('A') + i)
                    prev_level = cur_level
                    

            
            for kc, q_id_set in kc_q_mapping.items():
                if kc not in kc_q_level_mapping:
                    kc_q_level_mapping[kc] = defaultdict(list)
                    
                for q_id in q_id_set:
                    kc_q_level_mapping[kc][q_acc[q_id]['level']].append(q_id)
            
            neg_level_mapping = self.data_config['neg_level_mapping']
            
            def get_total_ratio(q_acc, threshold):
                valid_count = 0
                total_count = 0
                for k, v in q_acc.items():
                    if v['total'] >= threshold:
                        valid_count += 1
                    total_count += 1
                
                return valid_count / total_count
                        
                    
             
         

            
            for q, v in q_acc.items():  
                if v['total'] >= self.data_config['question_frequency_threshold']:
                    continue
                    
                cur_c = q_kc_mapping[q]
                positive_q = kc_q_level_mapping[cur_c].get(v['level'], [])
                positive_q = positive_q.copy()
                
            
                positive_q.remove(q) 
                
        
                positive_q = set(positive_q) & pos_neg_cand
                positive_q = list(positive_q)
                
                
         
                if not positive_q:
                    continue
                
                neg_levels = neg_level_mapping[v['level']]
                neg_q = []
                
                for neg_level in neg_levels:
                    neg_q.extend(kc_q_level_mapping[cur_c].get(neg_level, []))
                q_pair[str(q)] = {}
                q_pair[str(q)]['positive'] = positive_q
                q_pair[str(q)]['negative'] = neg_q
        
            
        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
   
        return dori, q_pair
