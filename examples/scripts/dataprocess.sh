SRC_DIR=../src


nohup python -u ${SRC_DIR}/data_preprocess.py --dataset_name assist2009 >assist2009.log&

nohup python -u ${SRC_DIR}/data_preprocess.py --dataset_name algebra2005 >algebra2005.log&

nohup python -u ${SRC_DIR}/data_preprocess.py --dataset_name statics2011 >statics2011.log&
nohup python -u ${SRC_DIR}/data_preprocess.py --dataset_name assist2015 >assist2015.log&

nohup python -u ${SRC_DIR}/data_preprocess.py --dataset_name poj >poj.log&
nohup python -u ${SRC_DIR}/data_preprocess.py --dataset_name bridge2algebra2006 >bridge2algebra2006.log&

nohup python -u ${SRC_DIR}/data_preprocess.py --dataset_name ednet5w >ednet5w.log&

nohup python -u ${SRC_DIR}/data_preprocess.py --dataset_name assist2012 >assist2012.log&

