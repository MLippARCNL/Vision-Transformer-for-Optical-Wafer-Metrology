## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirement.txt
```
Clone data_analysis_tools from amolf git
```bash
git clone https://git.amolf.nl/NANOIM/data_analysis_tools.git
```

## Running the experiments

First run _param_tuning.py_
```bash
python param_tuning.py
```
Check if folder ParameterTuning is created and contains .yaml files for each model. If you already have the yaml files you can skip this part. 

Go into experiments.py and set RUN and/or GENERATE_CSV to True/False if you want to get the results from scratch or load in saved results. Then run _experiments.py_.
```bash
python experiments.py
```
If the csv files exist in the data dir, experiments.py also plots the figures and loads in the data used for the tables in the paper. 

## Remarks
I have saved the original log files in the D: drive. More specifically at: \
D:\DL_DHM_logs

I changed the dataset names since the original naming convention (easy, medium, hard) did not make sense. If you encounter any errors that mention something like "name: 'easy_10' does not exist", please rename it as follows:\
'easy' --> W0 \
'medium' --> W1 \
'hard' --> W2 

So if you encounter for example 'medium_16', rename it to 'W1_16'

Depending on the Experiment type, outputs are saved under different directories. 
For Experiment 1, the file tree looks as follows:
```
.\
└── logs/
    ├── Experiment_1/                       
    │   ├── RegressionVit/                  
    │   │   ├── W0_10/                      
    │   │   │   ├── ViT_500K/               
    │   │   │   │   ├── version_0        
    │   │   │   │   ├── version_1
    │   │   │   │   ├── version_2
    │   │   │   │   ├── version_3
    │   │   │   │   └── version_4
    │   │   │   ├── ViT_1M
    │   │   │   ├── ViT_5M
    │   │   │   └── ViT_15M
    │   │   ├── W1_10
    │   │   ├── W2_10
    │   │   ├── W0_16
    │   │   ├── W1_16
    │   │   └── W2_16
    │   ├── RegressionConv
    │   └── RegressionMLP
    └── Experiment_2/
```
For Experiment 2, each model size and budget combination gets its own subdir.
```
.\
└── logs/
    ├── Experiment_1/                     
    └── Experiment_2/
        ├── RegressionVit/
        │   ├── W1_10/
        │   │   ├── ViT_500K_budget=32/      (Each Model size + budget has own dir)
        │   │   │   ├── version_0
        │   │   │   ├── version_1
        │   │   │   ├── version_2
        │   │   │   ├── version_3
        │   │   │   └── version_4
        │   │   ├── .
        │   │   ├── .
        │   │   ├── .
        │   │   ├── ViT_500K_budget=512
        │   │   ├── ViT_1M
        │   │   ├── ViT_5M
        │   │   └── ViT_15M
        │   ├── W2_10
        │   ├── W1_16
        │   └── W2_16
        ├── RegressionConv
        └── RegressionMLP
```

