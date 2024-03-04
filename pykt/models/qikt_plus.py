import os
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .que_base_model import QueBaseModel,QueEmb
from pykt.utils import debug_print
from sklearn import metrics
from torch.utils.data import DataLoader
from .loss import Loss
from .utils import MLP
from scipy.special import softmax


class MLPExperts(nn.Module):
    def __init__(self,
        dim,
        mlp_layer=1,
        num_experts=16,
        output_dim=None,
        dpo=0.5):
        super().__init__()
        
        
        self.dropout = nn.Dropout(p=dpo)
        self.num_experts = num_experts
        self.mlp_layer_num = mlp_layer
        self.output_dim = output_dim
        self.dpo = dpo
        

        experts_mlp = []
        experts_out = []

        for e in range(num_experts):
            temp_lins = nn.ModuleList([
                nn.Linear(dim, dim)
                for _ in range(mlp_layer)
            ])
            experts_mlp.append(temp_lins)
        
            temp_out = nn.Linear(dim, output_dim)
            experts_out.append(temp_out)

        self.experts_mlp = nn.ModuleList(experts_mlp)
        self.experts_out = nn.ModuleList(experts_out)
        self.layer_norm = nn.LayerNorm(dim)
    

    def forward(self, x):

        final_out = []

        for e in range(self.num_experts):
            cur_e_in = x[e]
            cur_e_mlp = self.experts_mlp[e]
            cur_e_out_layer = self.experts_out[e]
            
            for layer in cur_e_mlp:
                cur_e_in = F.relu(self.layer_norm(layer(cur_e_in)))
            
            cur_e_out = cur_e_out_layer(self.dropout(cur_e_in))
            final_out.append(cur_e_out)
        
        return torch.stack(final_out)

class Aggregate_MoE(nn.Module):
    def __init__(self,
        input_dim,
        num_experts=16,
        experts=None,):
        super().__init__()
        
        self.num_experts = num_experts
        self.experts = experts
        self.input_dim = input_dim
        
        
        self.gate = nn.Linear(input_dim, num_experts)
        
   
        
    def forward(self, inputs, **kwargs):
        
        gate_distribution = self.gate(inputs).softmax(dim=-1)

        
       
        all_expert_inputs = inputs.unsqueeze(0).repeat(self.num_experts, 1, 1, 1)
        all_expert_outputs = self.experts(all_expert_inputs)
        
       
        aggregate_outputs = torch.einsum("ebsd,bse->bsd", all_expert_outputs, gate_distribution)
        """
            for b in range(batch):
                for s in range(sequence):
                    fod d in range(embedding_dim):
                        result_sum = 0
                        for e in range(experts):
                            result_sum += all_expert_outputs[b, s, e, d] * gate_distribution[b, s, e]
                        aggregate_outputs[b, s, d] = result_sum 
                        
        """
   
        return aggregate_outputs




def get_outputs(self,emb_qc_shift,h,data,add_name="",model_type='question'):
    outputs = {}

    if model_type == 'question':
        q_all = self.out_question_all(h)
        y_question_all = torch.sigmoid(q_all)
        outputs["y_question_all"+add_name] = (y_question_all * F.one_hot(data['qshft'].long(), self.num_q)).sum(-1)
    else: 
        c_all = self.out_concept_all(h)
        y_concept_all = torch.sigmoid(c_all)
        outputs["y_concept_all"+add_name] = self.get_avg_fusion_concepts(y_concept_all,data['cshft'])

    return outputs

class QIKT_Plus_Net(nn.Module):
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',mlp_layer_num=1,other_config={}, **kwargs):
        super().__init__()
        self.model_name = "qikt+"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.mlp_layer_num = mlp_layer_num
        self.device = device
        self.other_config = other_config
        self.output_mode = self.other_config.get('output_mode','an_irt')
        self.q_pos_neg = kwargs.get('q_pos_neg', {}) 
        
        self.num_experts = self.other_config.get('num_experts', 2)
        self.neg_margin = self.other_config.get('contrastive_neg_distance_margin', 30)
        self.pos_margin = self.other_config.get('contrastive_pos_distance_margin', 23)
        self.contrastive_train_lr = self.other_config.get('contrastive_train_lr', 1e-3)
       


        self.emb_type = emb_type
        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,model_name=self.model_name,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)
       
      
        self.que_embed_copy_detach = nn.Embedding(self.que_emb.que_emb.num_embeddings, self.que_emb.que_emb.embedding_dim)
           

        self.que_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, batch_first=True)
        

        self.concept_lstm_layer = nn.LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
       
        self.dropout_layer = nn.Dropout(dropout)
        
        if self.num_experts > 1:
            print('Using Mixture of Experts Module ..')        
            self.out_question_all_experts = MLPExperts(
                    dim=self.hidden_size,
                    mlp_layer=self.mlp_layer_num,
                    num_experts=self.num_experts,
                    output_dim=num_q,
                    dpo=dropout
                )
           
            self.out_question_all = Aggregate_MoE(
                input_dim=self.hidden_size,
                num_experts=self.num_experts,
                experts=self.out_question_all_experts
            )

       
            self.out_concept_all_experts = MLPExperts(
                    dim=self.hidden_size,
                    mlp_layer=self.mlp_layer_num,
                    num_experts=self.num_experts,
                    output_dim=num_c,
                    dpo=dropout
                )
            
            self.out_concept_all = Aggregate_MoE(
                input_dim=self.hidden_size,
                num_experts=self.num_experts,
                experts=self.out_concept_all_experts
            )
            
        else:
            self.out_question_all = MLP(self.mlp_layer_num,self.hidden_size,num_q,dropout)
            self.out_concept_all = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)
        

        

    def get_avg_fusion_concepts(self,y_concept,cshft):
        """获取知识点 fusion 的预测结果
        """
        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long()==-1,False,True)
        concept_index = F.one_hot(torch.where(cshft!=-1,cshft,0),self.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1,1,max_num_concept,1)*concept_index).sum(-1)
        concept_sum = concept_sum*concept_mask#remove mask
        y_concept = concept_sum.sum(-1)/torch.where(concept_mask.sum(-1)!=0,concept_mask.sum(-1),1)
        return y_concept
    
    def to_device(self, x):
        return x.to(self.device)
    
    
    def get_positive_pairs(self, q):
        valid_mask = ~torch.eq(q, -1)
        valid_question = q.masked_select(valid_mask)
        unique_valid_questions = torch.unique(valid_question)

        shuffle_indices = torch.randperm(unique_valid_questions.size(0))
        unique_valid_questions = unique_valid_questions[shuffle_indices]
        
        pivot = []
        pos = []
        neg = []

        for unique_valid_question in unique_valid_questions:
            cur_q_id = str(unique_valid_question.item())
            if cur_q_id not in self.q_pos_neg:
                continue
            
            
            pivot_embed = self.que_emb.que_emb(unique_valid_question)
            
            
            positive_q = self.q_pos_neg[cur_q_id]['positive']
            if not positive_q:
                positive_que_embed = None
            else:
                positive_que_embed = self.que_embed_copy_detach(torch.LongTensor(positive_q).to(self.device))

            
            neg_q = self.q_pos_neg[cur_q_id]['negative']
            if neg_q:
                neg_que_embed = self.que_embed_copy_detach(torch.LongTensor(neg_q).to(self.device))
                
            else:
                neg_que_embed = None

            

            pivot.append(pivot_embed)
            pos.append(positive_que_embed)
            neg.append(neg_que_embed)

        
        return pivot, pos, neg
    
    def forward(self, q, c ,r, data=None, **kwargs):
       
        _,emb_qca,emb_qc,emb_q,emb_c = self.que_emb(q,c,r)#[batch_size,emb_size*4],[batch_size,emb_size*2],[batch_size,emb_size*1],[batch_size,emb_size*1]

        outputs = dict()
        if self.training:
            pivot, pos, neg = self.get_positive_pairs(q)
            outputs['pivot'] = pivot
            outputs['pos'] = pos
            outputs['neg'] = neg


        emb_qc_shift = emb_qc[:,1:,:]

        
        
        emb_qca_current = emb_qca[:,:-1,:]
        que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])

        que_outputs = get_outputs(self,emb_qc_shift,que_h,data,add_name="",model_type="question")
        outputs.update(que_outputs)

        
        emb_ca = torch.cat([emb_c.mul((1-r).unsqueeze(-1).repeat(1,1, self.emb_size)),
                                emb_c.mul((r).unsqueeze(-1).repeat(1,1, self.emb_size))], dim = -1)# s_t 扩展，分别对应正确的错误的情况
                                
        emb_ca_current = emb_ca[:,:-1,:]
        concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0])
        concept_outputs = get_outputs(self,emb_qc_shift,concept_h,data,add_name="",model_type="concept")
        outputs.update(concept_outputs)

        
        
        return outputs

class QIKTPlus(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',seed=0,mlp_layer_num=1,other_config={},**kwargs):
        model_name = "qikt+"
       
        debug_print(f"emb_type is {emb_type}",fuc_name="QIKT+")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = QIKT_Plus_Net(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,mlp_layer_num=mlp_layer_num,other_config=other_config,**kwargs)
        
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type

        self.loss_func = self._get_loss_func("binary_crossentropy")
        self.eval_result = {}
        self.q_pos_neg = kwargs.get('q_pos_neg', {})
    
    
    def eu_distance(self, x, y):
        return F.pairwise_distance(x, y, p=2)
    
    def contrastive_loss_func(self, pos_pivot=None, pos=None, neg_pivot=None, neg=None):
        try:
            if pos != None:
                pos_distance = self.eu_distance(pos_pivot, pos)
            if neg != None:
                neg_distance = self.eu_distance(neg_pivot, neg)
                
        except RuntimeError:
            import pdb; pdb.set_trace()
        
        
        if pos != None:
            positive_loss = F.relu(pos_distance - self.model.pos_margin).pow(2).mean()
       
        if neg != None:
            negative_loss = F.relu(self.model.neg_margin - neg_distance).pow(2).mean()
        
     
        if neg != None and pos != None:
            loss = positive_loss + negative_loss
        elif pos != None and neg == None:
            loss = positive_loss
        elif pos == None and neg != None:
            loss = negative_loss
        else:
            loss = 0
    
        return loss 
    
    def get_contrastive_loss(self, pivots, positives, negatives):
        """
            list of tensor
        """
        pos_pivot = []
        neg_pivot = []
        final_negatives = []
        final_positives = []
        
        for pivot, pos, neg in zip(pivots, positives, negatives):

            if pos != None:
                cur_pos_pivot = pivot.unsqueeze(0).repeat(pos.size(0), 1)
                pos_pivot.append(cur_pos_pivot)
                final_positives.append(pos)
            
        
            if neg != None:
                cur_neg_pivot = pivot.unsqueeze(0).repeat(neg.size(0), 1)
                neg_pivot.append(cur_neg_pivot)
                final_negatives.append(neg)
              
        if len(final_negatives) > 0 and len(final_positives) > 0:
            loss = self.contrastive_loss_func(
                pos_pivot=torch.cat(pos_pivot, dim=0), 
                pos=torch.cat(final_positives, dim=0),
                neg_pivot=torch.cat(neg_pivot, dim=0),
                neg=torch.cat(final_negatives, dim=0)
            )
        elif len(final_positives) > 0 and len(final_negatives) == 0:

            loss = self.contrastive_loss_func(
                pos_pivot=torch.cat(pos_pivot, dim=0), 
                pos=torch.cat(final_positives, dim=0),
            )
            
        elif len(final_negatives) > 0 and len(final_positives) == 0:

            loss = self.contrastive_loss_func(
                neg_pivot=torch.cat(neg_pivot, dim=0), 
                neg=torch.cat(final_negatives, dim=0),
            )
        else:
            loss = self.contrastive_loss_func()
        return loss 


        
    
    def train_one_step(self,data, process=True,return_all=False, **kwargs):
           
        outputs,data_new = self.predict_one_step(data, return_details=True, process=process, **kwargs)
        def get_loss_lambda(x):
            return self.model.other_config.get(f'loss_{x}',0)*self.model.other_config.get(f'output_{x}',0)
        
        
        contrastive_loss = self.get_contrastive_loss(outputs.get('pivot',[]), outputs.get('pos', []), outputs.get('neg', []))
        
        loss_q_all = self.get_loss(outputs['y_question_all'],data_new['rshft'],data_new['sm'])

        loss_c_all = self.get_loss(outputs['y_concept_all'],data_new['rshft'],data_new['sm'])

        loss_kt = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'])

        
            
        # loss weight
        loss_c_all_lambda = get_loss_lambda("c_all_lambda")
        loss_q_all_lambda = get_loss_lambda("q_all_lambda")
        
    
        kt_part_loss = loss_kt  + loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all
        loss = kt_part_loss + contrastive_loss*self.model.other_config.get('contrastive_loss_lambda',1)
             
        return outputs['y'],loss


    def predict(self,dataset,batch_size,return_ts=False,process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_pred_dict = {}
            for data in test_loader:
                new_data = self.batch_to_device(data,process=process)
                outputs,data_new = self.predict_one_step(data,return_details=True)
               
                for key in outputs:
                    if not key.startswith("y") or key in ['y_qc_predict']:
                        continue
                    elif key not in y_pred_dict:
                       y_pred_dict[key] = []
                    y = torch.masked_select(outputs[key], new_data['sm']).detach().cpu()#get label
                    y_pred_dict[key].append(y.numpy())
                
                t = torch.masked_select(new_data['rshft'], new_data['sm']).detach().cpu()
                y_trues.append(t.numpy())


        results = y_pred_dict
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)
        ts = np.concatenate(y_trues, axis=0)
        results['ts'] = ts
        return results

    def evaluate(self,dataset,batch_size,acc_threshold=0.5):
        results = self.predict(dataset,batch_size=batch_size)
        eval_result = {}
        ts = results["ts"]
        for key in results:
            if not key.startswith("y") or key in ['y_qc_predict']:
                pass
            else:
                ps = results[key]
                kt_auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
                prelabels = [1 if p >= acc_threshold else 0 for p in ps]
                kt_acc = metrics.accuracy_score(ts, prelabels)
                if key!="y":
                    eval_result["{}_kt_auc".format(key)] = kt_auc
                    eval_result["{}_kt_acc".format(key)] = kt_acc
                else:
                    eval_result["auc"] = kt_auc
                    eval_result["acc"] = kt_acc
        
        self.eval_result = eval_result
        return eval_result

    def predict_one_step(self,data, return_details=False,process=True,return_raw=False, **kwargs):
        data_new = self.batch_to_device(data,process=process)
        outputs = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(), data=data_new, **kwargs)
    
        
        def sigmoid_inverse(x,epsilon=1e-8):
                return torch.log(x/(1-x+epsilon)+epsilon)
        
        if self.model.output_mode=="an_irt":
            y = sigmoid_inverse(outputs['y_question_all']) + sigmoid_inverse(outputs['y_concept_all'])
            y = torch.sigmoid(y)
        else:
            # output weight
            y = outputs['y_question_all'] + outputs['y_concept_all']
        outputs['y'] = y
        if return_details:
            return outputs,data_new
        else:
            return y