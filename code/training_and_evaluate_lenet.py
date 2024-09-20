import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import csv

# Import LeNet5 instead of ResNet
from lenet import LeNet5
from loader_new4 import getTrain, getVal, getTest

if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Use default batch size suitable for LeNet5
    trainloader = getTrain(num_chan=1, batch_size=35)  # Default batch size for LeNet5
    valloader = getVal(num_chan=1)

    # Initialize LeNet5 model with 1 channel
    net = LeNet5(num_chan=1)
    net.to(device=device)

    # Loss function and optimizer with default parameters for LeNet5
    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.0008451675820105067,
                            weight_decay=0.1945334404427726)

    # Training loop
    with open(os.path.join("lenet_tuned_epoch12.csv"), mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', "Validation L1 Loss"])
        t0 = time.perf_counter()
        best_val = 35
        for epoch in range(12):  # Loop over the dataset multiple times for baseline (increased epochs)
            print("Training epoch " + str(epoch + 1))
            t1 = time.perf_counter()
            running_loss = 0.0
            total_loss = 0.0
            step = 0
            net.train()
            for i, data in enumerate(trainloader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                total_loss += loss.item()
                if i % 200 == 199:  # Print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
                step += 1

            print("Evaluating epoch " + str(epoch + 1))
            net.eval()
            with torch.no_grad():
                val_steps = 0
                val_loss = 0.0
                val_l1loss = 0.0
                for j, data in enumerate(valloader):
                    val_imgs, val_labels = data
                    val_imgs = val_imgs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = net(val_imgs)
                    loss = criterion(val_outputs, val_labels.unsqueeze(1)).item()
                    l1loss = criterion2(val_outputs, val_labels.unsqueeze(1)).item()
                    val_loss += loss
                    val_l1loss += l1loss
                    val_steps += 1
            speed = round(time.perf_counter() - t1)
            ave_val = val_loss / val_steps
            ave_val_l1 = val_l1loss / val_steps
            print('[Epoch %d] training loss: %.3f validation loss: %.3f validation l1 loss: %.3f time taken: %ds' %
                  (epoch + 1, total_loss / step, ave_val, ave_val_l1, speed))
            csv_writer.writerow([epoch + 1, total_loss / step, ave_val, ave_val_l1])
            if ave_val_l1 < best_val:
                model_name = 'lenet_tuned_epoch12.pt'
                torch.save(net.state_dict(), model_name)
                best_val = ave_val_l1
                best_epoch = epoch + 1
            print("current best val loss: " + str(best_val) + " epoch: " + str(best_epoch))

    print('LeNet5 Training done.')
    print("Final best val loss: " + str(best_val) + " epoch: " + str(best_epoch))



# evaluation:
    device = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()

    testloader = getTest(num_chan=1)

    net=LeNet5(num_chan=1)
    net.load_state_dict(torch.load('lenet_tuned_epoch12.pt'))
    #net = torch.load('saved_model_1_18_last.pt')
    net = net.to(device=device)
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            test_imgs, test_labels = data
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            test_outputs = net(test_imgs)
            loss = criterion(test_outputs, test_labels.unsqueeze(1)).item()
            l1loss = criterion2(test_outputs, test_labels.unsqueeze(1)).item()
    print(loss, l1loss)


    with open('evaluate_lenet_tuned_epoch12.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(test_labels.cpu().numpy(), test_outputs.squeeze(1).cpu().numpy()))
