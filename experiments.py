import sys
sys.path.append(r"C:\Users\dwolf\PycharmProjects\data_analysis_tools")
import time
import yaml
import pandas as pd
from train import main
from multiprocessing import freeze_support
from plot import *
from models import *
from data import Data
from pathlib import Path


PARAM_DIR = 'logs/ParameterTuning/'
MODELS = ['RegressionVit', 'RegressionConv', 'RegressionMLP']
TYPES = {
    'RegressionVit': {
        'vit_500K': {'patch_size': 32, 'hidden_dim': 96, 'n_heads': 2, 'n_layers': 4, 'mlp_factor': 2},
        'vit_1M': {'patch_size': 32, 'hidden_dim': 120, 'n_heads': 4, 'n_layers': 6, 'mlp_factor': 2},
        'vit_5M': {'patch_size': 32, 'hidden_dim': 240, 'n_heads': 6, 'n_layers': 8, 'mlp_factor': 3},
        'vit_15M': {'patch_size': 32, 'hidden_dim': 360, 'n_heads': 8, 'n_layers': 10, 'mlp_factor': 4},
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

def run(name, budgets, datasets):
    for model in MODELS:

        for type in TYPES[model]:

            for dataset in datasets:

                target = 'W0_16' if '16' in dataset else 'W0_10'
                with open(f"{PARAM_DIR}/{model}/{target}/{type}.yaml", "r") as f:
                    model_args = [f'--model.{arg}={value}' for arg, value in yaml.safe_load(f)['model']['init_args'].items()]

                for budget in budgets:
                    type = f"{type}_budget={budget}" if len(budgets) > 1 else type
                    print(f"Training {model} {dataset}")

                    for repeat in range(5):
                        seed = np.random.randint(1, 10000)
                        general_args = [
                            f'fit',
                            f'--seed_everything={seed}',
                            f'--name={name}',
                            f'--model={model}',
                            f'--data.train_file={dataset}',
                            f'--data.val_file={dataset}',
                            f'--data.test_file={dataset}',
                            f'--data.budget={budget}',
                            f'--data.augment={True}',
                            f'--type={type}',
                        ]

                        match model:
                            case 'RegressionVit':
                                test_loss = main(general_args + model_args)

                            case 'RegressionMLP':
                                test_loss = main(general_args + model_args)

                            case 'RegressionConv':
                                test_loss = main(general_args + model_args)

def get_csv(log_path='D:\DL_DHM_logs'):
    models = {'RegressionMLP': RegressionMLP,
              'RegressionConv': RegressionConv,
              'RegressionVit': RegressionVit}

    for exp in ['Experiment_1', 'Experiment_2']:
        checkpoints = []
        path = Path(f'{log_path}/{exp}')
        for ckpt in path.rglob('*last*.ckpt'):
            rel = ckpt.relative_to(path).parts
            model_name, dataset, model_scale, version = rel[:4]
            checkpoints.append(
                {'Model': model_name, 'Dataset': dataset, 'ModelType': model_scale, 'Version': version, 'ckpt': ckpt})
        df = pd.DataFrame(checkpoints)

        output = []
        for dataset, df_ in df.groupby('Dataset'):
            data = Data(dataset, dataset, dataset)
            data.setup('test')
            loader = data.test_dataloader()
            for model_name, df_model in df_.groupby('ModelType'):
                for i, rows in enumerate(df_model.iterrows()):
                    model = models[df_model.Model.iloc[0]].load_from_checkpoint(rows[-1]['ckpt']).eval().to('cuda')
                    pred, gt = [], []
                    for y, x in loader:
                        x = x.to('cuda')
                        out = model(x)
                        pred.append(out.detach().cpu().item())
                        gt.append(y.item())
                    if exp[-1] == '2':
                        output.append(
                            {'Model': model_name.split('_budget=')[0], 'Budget': model_name.split('_budget=')[1], 'Dataset': dataset, 'Version': i, 'pred': pred, 'GT': gt}
                        )
                    else:
                        output.append(
                            {'Model': model_name, 'Dataset': dataset, 'Version': i, 'pred': pred, 'GT': gt}
                        )
                print('done model')

        output = pd.DataFrame(output)
        # Losses
        output['test_mae'] = output.apply(lambda row: F.l1_loss(torch.tensor(row['pred']), torch.tensor(row['GT'])), axis=1)
        output['test_mse'] = output.apply(lambda row: F.mse_loss(torch.tensor(row['pred']), torch.tensor(row['GT'])),
                                          axis=1)

        if '1' in exp:
            # Figure 5
            output[
                (output.Model == 'vit_5M') & (
                        (output.Dataset == 'W1_10') |
                        (output.Dataset == 'W1_16') |
                        (output.Dataset == 'W2_10') |
                        (output.Dataset == 'W2_16'))
                ].to_csv('data/fig_5.csv', index=False)
            output[['Model', 'Dataset', 'Version', 'test_mse', 'test_mae']].to_csv(f'data/full_data_losses.csv', index=False)
        else:
            output[['Model', 'Budget', 'Dataset', 'Version', 'test_mse', 'test_mae']].to_csv(f'data/limited_data_losses.csv', index=False)

def get_runtimes():
    models = {'RegressionMLP': RegressionMLP,
              'RegressionConv': RegressionConv,
              'RegressionVit': RegressionVit}
    outputs = []

    for model, types in TYPES.items():
        for type in types:
            net = models[model](**types[type]).eval().to('cuda')
            output = {}
            for batch in [1, 64]:
                input = torch.randn(batch, 2, 256, 256).to('cuda')
                # Warm-up
                for _ in range(10):
                    _ = net(input)

                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    for _ in range(100):
                        _ = net(input)
                torch.cuda.synchronize()
                end = time.time()

                avg_time_per_sample = (((end - start) / 100) / batch) * 1000 # s -> ms
                output[str(batch)] = avg_time_per_sample

                if batch == 64:
                    optim = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.1)
                    crit = nn.MSELoss()
                    inputs, targets = torch.randn(1024, 2, 256, 256), torch.randn(1024, 1)

                    torch.cuda.synchronize()
                    start = time.time()
                    for _ in range(100):
                        for i in range(0, 1024, 64):
                            x, yhat = inputs[i:i + 64].to('cuda'), targets[i:i + 64].to('cuda')
                            optim.zero_grad()
                            out = net(x)
                            loss = crit(out, yhat)
                            loss.backward()
                            optim.step()
                    torch.cuda.synchronize()
                    end = time.time()
                    avg_time_per_sample = (end - start) / 100   # s
                    output['train'] = avg_time_per_sample
            outputs.append({
                'Model': type, 'Seq.': output['1'], 'Batched.': output['64'], 'Train': output['train']
            })
    df_ = pd.DataFrame(outputs)
    df_.to_csv('data/runtimes.csv', index=False)



if __name__ == '__main__':
    freeze_support()
    RUN = False # If you do not have the log files
    GENERATE_CSV = False # If you have the log ('D:\DL_DHM_logs') files, but not the csv files

    if RUN:
        # Experiment 1
        run(name='Experiment_1',
            budgets=[1024],
            datasets=['W0_10', 'W1_10', 'W2_10', 'W0_16', 'W1_16', 'W2_16'])

        # Experiment_2
        run(name='Experiment_2',
            budgets=[32, 64, 128, 256, 512],
            datasets=['W1_10', 'W2_10', 'W1_16', 'W2_16'])

    if GENERATE_CSV:
        # Get csv files
        get_csv()
        get_runtimes()

    ## Thesis Plots

    # Plot: Fig 5.
    df = pd.read_csv('data/fig_5.csv')
    fig_5(df, True)


    # Plot: Fig 6.
    df = pd.read_csv('data/Experiment_1_losses.csv')
    mean_metric_over_wavefronts(df, 'test_mae')

    ## Thesis Tables

    # Table 2:
    df = pd.read_csv('data/runtimes.csv')

    # Table 3-4
    df = pd.read_csv('data/full_data_losses.csv')

    # Table 5-6
    df = pd.read_csv('data/limited_data_losses.csv')





