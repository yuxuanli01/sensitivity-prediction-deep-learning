import optuna
from optuna.trial import TrialState
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lenet import LeNet5
from loader_new4 import getTrain, getVal

def train_and_evaluate(params, net, trial):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Use the updated loader function without balanced batches
    trainloader = getTrain(norm=False, num_chan=1, batch_size=params['batch_size'])
    valloader = getVal(num_chan=1, norm=False)

    net.to(device=device)
    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'], weight_decay=params['decay'])

    early_stopping_patience = 6

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(12):  # Adjust number of epochs if necessary
        print(f"Training epoch {epoch + 1}")
        running_loss = 0.0
        total_loss = 0.0
        step = 0
        net.train()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 200 == 199:
                running_loss = 0.0
            step += 1

        print(f"Evaluating epoch {epoch + 1}")
        net.eval()
        val_steps = 0
        val_loss = 0.0
        with torch.no_grad():
            for j, data in enumerate(valloader):
                val_imgs, val_labels = data
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = net(val_imgs)
                loss = criterion2(val_outputs, val_labels.unsqueeze(1)).item()
                val_loss += loss
                val_steps += 1

        avg_val_loss = val_loss / val_steps
        print(f'[Epoch {epoch + 1}] training loss: {total_loss / step:.3f} validation loss: {avg_val_loss:.3f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        trial.report(avg_val_loss, epoch)

        if trial.should_prune() or epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered")
            print(f"Trial {trial.number} failed or pruned with params: {params}")
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

def objective(trial):
    params = { 
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'decay': trial.suggest_float('decay', 0.15, 0.4),
        'batch_size': trial.suggest_int('batch_size', 32, 64)
    }
    
    # Instantiate LeNet5 with the appropriate number of input channels
    net = LeNet5(num_chan=1)
    
    mae = train_and_evaluate(params, net, trial)

    # Print the hyperparameters after each trial
    print(f"\nTrial {trial.number} completed.")
    print(f"Hyperparameters: {params}")
    print(f"Mean Absolute Error: {mae}\n")

    return mae

if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)

    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(completed_trials) > 0:
        print("\nFinal Best Trial Overall:")
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print(f"{key}: {value}")
    else:
        print('No best trial found.')