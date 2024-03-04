
train(){
    python -u ${SRC_DIR}/wandb_${MODEL_NAME}_train.py \
    --dataset_name ${DATASET_NAME} \
    --fold ${FOLD} \
    --save_dir ${SAVE_DIR} \
    --num_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --batch_size ${BATCH_SIZE} \
    --dropout ${DROPOUT} \
    --seed ${SEED} \
    --mlp_layer_num ${MLP_LAYER_NUM} \
    --use_wandb 0 \
    --loss_q_all_lambda ${LOSS_Q_ALL_LAMBDA} \
    --loss_c_all_lambda ${LOSS_C_ALL_LAMBDA} \
    --loss_c_next_lambda ${LOSS_C_NEXT_LAMBDA} \
    --emb_size ${EMB_SIZE}
}

# done
eval(){
    python -u ${SRC_DIR}/get_final_five_fold_result.py \
    --saved_model_dir ${SAVE_DIR} \
    --output_dir ${SAVE_DIR} \
    --bz ${BATCH_SIZE} \
    --use_wandb 0
}





SRC_DIR=../src
DATASET_NAME=bridge2algebra2006
export CUDA_VISIBLE_DEVICES=7

MODEL_NAME=qikt
SAVE_DIR=../saved_models/${MODEL_NAME}-${DATASET_NAME}
EPOCHS=200
LR=1e-4
BATCH_SIZE=8 
EMB_SIZE=256

mkdir -p ${SAVE_DIR}


#! fold 0
#! 'loss_q_all_lambda': 2.0, 'loss_c_all_lambda': 1.5, 'loss_c_next_lambda': 1.0, 'output_q_all_lambda': 1, 'output_c_all_lambda': 1, 'output_c_next_lambda': 1
LOSS_Q_ALL_LAMBDA=2.0
LOSS_C_ALL_LAMBDA=1.5
LOSS_C_NEXT_LAMBDA=1.0
SEED=42
DROPOUT=0.5
MLP_LAYER_NUM=1
FOLD=0

train

#! fold1
#! 'loss_q_all_lambda': 1.0, 'loss_c_all_lambda': 2.0, 'loss_c_next_lambda': 2.0, 'output_q_all_lambda': 1, 'output_c_all_lambda': 1, 'output_c_next_lambda': 1
LOSS_Q_ALL_LAMBDA=1.0
LOSS_C_ALL_LAMBDA=2.0
LOSS_C_NEXT_LAMBDA=2.0
SEED=3407
DROPOUT=0.5
MLP_LAYER_NUM=1
FOLD=1


train

#! fold2
#! 'loss_q_all_lambda': 1.5, 'loss_c_all_lambda': 2.0, 'loss_c_next_lambda': 2.0, 'output_q_all_lambda': 1, 'output_c_all_lambda': 1, 'output_c_next_lambda': 1
LOSS_Q_ALL_LAMBDA=1.5
LOSS_C_ALL_LAMBDA=2.0
LOSS_C_NEXT_LAMBDA=2.0
MLP_LAYER_NUM=1
DROPOUT=0.5
SEED=3407
FOLD=2


train

#! fold3
#! 'loss_q_all_lambda': 2.0, 'loss_c_all_lambda': 2.0, 'loss_c_next_lambda': 1.5, 'output_q_all_lambda': 1, 'output_c_all_lambda': 1, 'output_c_next_lambda': 1
LOSS_Q_ALL_LAMBDA=2.0
LOSS_C_ALL_LAMBDA=2.0
LOSS_C_NEXT_LAMBDA=1.5
MLP_LAYER_NUM=1
DROPOUT=0.5
SEED=3407
FOLD=3


train

#! fold4
#! 'loss_q_all_lambda': 0.0, 'loss_c_all_lambda': 0.5, 'loss_c_next_lambda': 2.0, 'output_q_all_lambda': 1, 'output_c_all_lambda': 1, 'output_c_next_lambda': 1
LOSS_Q_ALL_LAMBDA=0.0
LOSS_C_ALL_LAMBDA=0.5
LOSS_C_NEXT_LAMBDA=2.0
MLP_LAYER_NUM=1
DROPOUT=0.5
SEED=3407
FOLD=4

train


eval


