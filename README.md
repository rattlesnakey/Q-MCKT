# Setup
```shell
python setup.py install
conda env create -f environment.yaml or pip install -r requirements.txt
```




# Experiment

## Data Processing

- Follow the instruction in [documentation](https://pykt-toolkit.readthedocs.io/en/latest/datasets.html) to get datasets, then execute `cd examples && bash dataprocess.sh`



## Train & Eval
- `cd examples && bash train_q-mckt_{dataset}.shg`, set the dataset you want to train and evaluate.