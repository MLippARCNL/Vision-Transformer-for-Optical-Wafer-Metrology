import sys
import os
import optuna

import numpy as np
import matplotlib.pyplot as plt

from train import main
from multiprocessing import freeze_support
from optuna.visualization.matplotlib import (
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_timeline
)

## adding the path to my code base to the python import path
sys.path.append(r"C:\Users\dwolf\PycharmProjects\data_analysis_tools")
## change path location accordingly where you have cloned the git

def objective(trial: optuna.Trial, model_name: str, type: str, params: dict, train_file: str, epochs: int, name: str):
    params = {
        'lr': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        'dropout': trial.suggest_float("dropout", 0, 0.5),
        'scheduler': trial.suggest_categorical("schedule", ["cosine"]),
    } | params

    seed = np.random.randint(1, 10000)
    args1 = [
        'fit',
        f'--model={model_name}',
        f'--name={name}',
        f'--data.train_file={train_file}',
        f'--data.split_train={0.75}',
        f'--data.augment=True',
        f'--trainer.max_epochs={epochs}',
        f'--seed_everything={seed}',
        '--trainer.enable_model_summary=False',
        f'--type={type}',
    ]
    run = args1 + [f'--model.{param}={value}' for param, value in params.items()]
    return main(run)


if __name__ == "__main__":
    freeze_support()

    EPOCHS = 75
    TRIALS = 25
    NAME = "ParameterTuning"

    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler()

    TYPES = {
        'RegressionVit': {
            'vit_500K': {'patch_size': 32, 'hidden_dim': 96, 'n_heads': 2, 'n_layers': 4, 'mlp_factor': 2},
            'vit_1M': {'patch_size': 32, 'hidden_dim': 120, 'n_heads' : 4, 'n_layers': 6, 'mlp_factor': 2},
            'vit_5M': {'patch_size': 32, 'hidden_dim': 240, 'n_heads' : 6, 'n_layers': 8, 'mlp_factor': 3},
            'vit_15M': {'patch_size': 32, 'hidden_dim': 360, 'n_heads' : 8, 'n_layers': 10, 'mlp_factor': 4},
        },
        'RegressionConv': {
            'conv_100K': {'num_blocks': 3},
            'conv_500K': {'num_blocks': 4},
            'conv_2M': {'num_blocks': 5},
        },
        'RegressionMLP': {
            'mlp_15M': {'hidden_dim': 120, 'n_layers': 3},
            'mlp_30M': {'hidden_dim': 240, 'n_layers': 3}
        }
    }

    for MODEL in ['RegressionVit', 'RegressionConv', 'RegressionMLP']:
        for SPLIT in ['W0_10', 'W0_16']:
            for TYPE in TYPES[MODEL]:
                print(f"Hyperparameter Tuning on {TYPE} for {SPLIT} dataset")

                study = optuna.create_study(study_name=f'{NAME}_{MODEL}_{SPLIT}', direction="minimize", sampler=sampler, pruner=pruner)
                study.optimize(lambda trial: objective(trial, MODEL, TYPE, TYPES[MODEL][TYPE], SPLIT, EPOCHS, NAME),
                               n_trials=TRIALS, n_jobs=1,
                               show_progress_bar=True,
                               catch=[
                                   AssertionError,
                                   ValueError,
                               ])

                print("Number of finished trials: {}".format(len(study.trials)))
                print("Best trial:")
                trial = study.best_trial

                print("  Value: {}".format(trial.value))
                print("  Params: ")
                for key, value in trial.params.items():
                    print("    {}: {}".format(key, value))

                out =  "\n".join([f"{key}: {value}" for key, value in trial.params.items()])

                with open(f"logs/{NAME}/{MODEL}/{SPLIT}/{TYPE}/outputs.txt", "w") as f:
                    f.write(out)

                # Prepare yaml file
                params = TYPES[MODEL][TYPE] | trial.params
                os.system(f'python train.py fit --print_config --model={MODEL} ' +
                          ' '.join([f'--model.{param}={value}' for param, value in params.items()]) +
                          f' > logs/{NAME}/{MODEL}/{SPLIT}/{TYPE}.yaml')

                plot_intermediate_values(study)
                plt.savefig(f"logs/{NAME}/{MODEL}/{SPLIT}/{TYPE}/intermediate_values_{TYPE}_{SPLIT}.png")
                plot_optimization_history(study)
                plt.savefig(f"logs/{NAME}/{MODEL}/{SPLIT}/{TYPE}/plot_optimization_history_{TYPE}_{SPLIT}.png")
                plot_param_importances(study)
                plt.savefig(f"logs/{NAME}/{MODEL}/{SPLIT}/{TYPE}/plot_param_importances_{TYPE}_{SPLIT}.png")
                plot_parallel_coordinate(study)
                plt.savefig(f"logs/{NAME}/{MODEL}/{SPLIT}/{TYPE}/plot_parallel_coordinate_{TYPE}_{SPLIT}.png")
                plot_timeline(study)
                plt.savefig(f"logs/{NAME}/{MODEL}/{SPLIT}/{TYPE}/plot_parallel_timeline_{TYPE}_{SPLIT}.png")

