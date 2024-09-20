import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import csv

from resnet_new import newResNet18_Simplified
from lenet import LeNet5
from loader_new4 import getTrain, getVal, getTest

if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    trainloader = getTrain(num_chan=1, batch_size=32)
    # trainloader = DataLoader(trainset, batch_size=45, shuffle=True, num_workers=2) # batch_size=43
    valloader = getVal(num_chan=1)


    ## cnn
    net = newResNet18_Simplified(num_chan=1, dropout=0.293825466125746) # dropout=0.154
    net.to(device=device)

    ## loss and optimiser
    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=7.358803984928571e-05, 
                           weight_decay=0.2952579423272889) # lr=0.00024


    ## train epoch9
    with open(os.path.join("resnet18_t1_epoch12.csv"), mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', "Validation L1 Loss"])
        t0 = time.perf_counter()
        best_val = 35
        for epoch in range(12):  # loop over the dataset multiple times # 10 epochs
            print("Training epoch "+str(epoch+1))
            t1= time.perf_counter()
            running_loss = 0.0
            total_loss = 0.0
            step = 0
            net.train()
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                total_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
                step += 1

            print("Evaluating epoch "+str(epoch+1))
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
            speed = round(time.perf_counter()-t1)
            ave_val = val_loss/val_steps
            ave_val_l1 = val_l1loss/val_steps
            print('[Epoch %d] training loss: %.3f validation loss: %.3f validation l1 loss: %.3f time taken: %ds' % (epoch+1, total_loss/step, ave_val, ave_val_l1, speed))
            csv_writer.writerow([epoch+1, total_loss/step, ave_val, ave_val_l1])
            if ave_val_l1 < best_val:
                model_name = 'resnet18_t1_epoch12.pt'
                torch.save(net.state_dict(), model_name)
                best_val = ave_val_l1
                best_epoch = epoch+1
            print("current best val loss: "+str(best_val)+" epoch: "+str(best_epoch))

    print('Resnet18 resnet18_t1_epoch12 Training done.')
    #print("Parameters: batch size="+str(batch_size)+" lr="+str(lr)+" momentum="+str(momentum)+" dropout="+str(dropout))
    print("Final best val loss: "+str(best_val)+" epoch: "+str(best_epoch))



# #11:

#     device = ('cuda' if torch.cuda.is_available() else 'cpu')

#     trainloader = getTrain(num_chan=1, batch_size=32)
#     # trainloader = DataLoader(trainset, batch_size=45, shuffle=True, num_workers=2) # batch_size=43
#     valloader = getVal(num_chan=1)


#     ## cnn
#     net = newResNet18_Simplified(num_chan=1, dropout=0.293825466125746) # dropout=0.154
#     net.to(device=device)

#     ## loss and optimiser
#     criterion = torch.nn.MSELoss()
#     criterion2 = torch.nn.L1Loss()
#     optimizer = optim.Adam(net.parameters(), lr=7.358803984928571e-05, 
#                            weight_decay=0.2952579423272889) # lr=0.00024


#     ## train
#     with open(os.path.join("resnet18_t1_epoch11.csv"), mode='w', newline='') as file:
#         csv_writer = csv.writer(file)
#         csv_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', "Validation L1 Loss"])
#         t0 = time.perf_counter()
#         best_val = 35
#         for epoch in range(11):  # loop over the dataset multiple times # 10 epochs
#             print("Training epoch "+str(epoch+1))
#             t1= time.perf_counter()
#             running_loss = 0.0
#             total_loss = 0.0
#             step = 0
#             net.train()
#             for i, data in enumerate(trainloader, 0):
#                 # get the inputs; data is a list of [inputs, labels]
#                 inputs, labels = data
#                 inputs = inputs.to(device=device)
#                 labels = labels.to(device=device)
#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward + backward + optimize
#                 outputs = net(inputs)
#                 loss = criterion(outputs, labels.unsqueeze(1))
#                 loss.backward()
#                 optimizer.step()

#                 # print statistics
#                 running_loss += loss.item()
#                 total_loss += loss.item()
#                 if i % 200 == 199:    # print every 200 mini-batches
#                     print('[%d, %5d] loss: %.3f' %
#                         (epoch + 1, i + 1, running_loss / 200))
#                     running_loss = 0.0
#                 step += 1

#             print("Evaluating epoch "+str(epoch+1))
#             net.eval()
#             with torch.no_grad():
#                 val_steps = 0
#                 val_loss = 0.0
#                 val_l1loss = 0.0
#                 for j, data in enumerate(valloader):
#                     val_imgs, val_labels = data
#                     val_imgs = val_imgs.to(device)
#                     val_labels = val_labels.to(device)
#                     val_outputs = net(val_imgs)
#                     loss = criterion(val_outputs, val_labels.unsqueeze(1)).item()
#                     l1loss = criterion2(val_outputs, val_labels.unsqueeze(1)).item()
#                     val_loss += loss
#                     val_l1loss += l1loss
#                     val_steps += 1
#             speed = round(time.perf_counter()-t1)
#             ave_val = val_loss/val_steps
#             ave_val_l1 = val_l1loss/val_steps
#             print('[Epoch %d] training loss: %.3f validation loss: %.3f validation l1 loss: %.3f time taken: %ds' % (epoch+1, total_loss/step, ave_val, ave_val_l1, speed))
#             csv_writer.writerow([epoch+1, total_loss/step, ave_val, ave_val_l1])
#             if ave_val_l1 < best_val:
#                 model_name = 'resnet18_t1_epoch11.pt'
#                 torch.save(net.state_dict(), model_name)
#                 best_val = ave_val_l1
#                 best_epoch = epoch+1
#             print("current best val loss: "+str(best_val)+" epoch: "+str(best_epoch))

#     print('Resnet18 resnet18_t1_epoch11 Training done.')
#     #print("Parameters: batch size="+str(batch_size)+" lr="+str(lr)+" momentum="+str(momentum)+" dropout="+str(dropout))
#     print("Final best val loss: "+str(best_val)+" epoch: "+str(best_epoch))






# evaluation: 12

device = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.MSELoss()
criterion2 = torch.nn.L1Loss()

testloader = getTest(num_chan=1)

net=newResNet18_Simplified(num_chan=1)
net.load_state_dict(torch.load('resnet18_t1_epoch12.pt'))
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
        #print(loss, l1loss)
print(loss, l1loss) #76.94039916992188 7.35371208190918

with open('evaluate_resnet18_t1_epoch12.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(test_labels.cpu().numpy(), test_outputs.squeeze(1).cpu().numpy()))


# # 14:

# net=newResNet18_Simplified(num_chan=1)
# net.load_state_dict(torch.load('resnet18_t1_epoch14.pt'))
# #net = torch.load('saved_model_1_18_last.pt')
# net = net.to(device=device)
# net.eval()
# with torch.no_grad():
#     for i, data in enumerate(testloader):
#         test_imgs, test_labels = data
#         test_imgs = test_imgs.to(device)
#         test_labels = test_labels.to(device)
#         test_outputs = net(test_imgs)
#         loss = criterion(test_outputs, test_labels.unsqueeze(1)).item()
#         l1loss = criterion2(test_outputs, test_labels.unsqueeze(1)).item()
#         #print(loss, l1loss)
# print(loss, l1loss) #76.94039916992188 7.35371208190918

# with open('evaluate_resnet18_t1_epoch14.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(zip(test_labels.cpu().numpy(), test_outputs.squeeze(1).cpu().numpy()))





# # lenet:

#     device = ('cuda' if torch.cuda.is_available() else 'cpu')

#     # Use default batch size suitable for LeNet5
#     trainloader = getTrain1(num_chan=1, batch_size=64)  # Default batch size for LeNet5
#     valloader = getVal1(num_chan=1)

#     # Initialize LeNet5 model with 1 channel
#     net = LeNet5(num_chan=1)
#     net.to(device=device)

#     # Loss function and optimizer with default parameters for LeNet5
#     criterion = torch.nn.MSELoss()
#     criterion2 = torch.nn.L1Loss()
#     optimizer = optim.Adam(net.parameters(), lr=0.2)  # Typical default learning rate

#     # Training loop
#     with open(os.path.join("lenet_epoch10.csv"), mode='w', newline='') as file:
#         csv_writer = csv.writer(file)
#         csv_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', "Validation L1 Loss"])
#         t0 = time.perf_counter()
#         best_val = 35
#         for epoch in range(10):  # Loop over the dataset multiple times for baseline (increased epochs)
#             print("Training epoch " + str(epoch + 1))
#             t1 = time.perf_counter()
#             running_loss = 0.0
#             total_loss = 0.0
#             step = 0
#             net.train()
#             for i, data in enumerate(trainloader, 0):
#                 # Get the inputs; data is a list of [inputs, labels]
#                 inputs, labels = data
#                 inputs = inputs.to(device=device)
#                 labels = labels.to(device=device)
#                 # Zero the parameter gradients
#                 optimizer.zero_grad()

#                 # Forward + backward + optimize
#                 outputs = net(inputs)
#                 loss = criterion(outputs, labels.unsqueeze(1))
#                 loss.backward()
#                 optimizer.step()

#                 # Print statistics
#                 running_loss += loss.item()
#                 total_loss += loss.item()
#                 if i % 200 == 199:  # Print every 200 mini-batches
#                     print('[%d, %5d] loss: %.3f' %
#                           (epoch + 1, i + 1, running_loss / 200))
#                     running_loss = 0.0
#                 step += 1

#             print("Evaluating epoch " + str(epoch + 1))
#             net.eval()
#             with torch.no_grad():
#                 val_steps = 0
#                 val_loss = 0.0
#                 val_l1loss = 0.0
#                 for j, data in enumerate(valloader):
#                     val_imgs, val_labels = data
#                     val_imgs = val_imgs.to(device)
#                     val_labels = val_labels.to(device)
#                     val_outputs = net(val_imgs)
#                     loss = criterion(val_outputs, val_labels.unsqueeze(1)).item()
#                     l1loss = criterion2(val_outputs, val_labels.unsqueeze(1)).item()
#                     val_loss += loss
#                     val_l1loss += l1loss
#                     val_steps += 1
#             speed = round(time.perf_counter() - t1)
#             ave_val = val_loss / val_steps
#             ave_val_l1 = val_l1loss / val_steps
#             print('[Epoch %d] training loss: %.3f validation loss: %.3f validation l1 loss: %.3f time taken: %ds' %
#                   (epoch + 1, total_loss / step, ave_val, ave_val_l1, speed))
#             csv_writer.writerow([epoch + 1, total_loss / step, ave_val, ave_val_l1])
#             if ave_val_l1 < best_val:
#                 model_name = 'lenet_epoch10.pt'
#                 torch.save(net.state_dict(), model_name)
#                 best_val = ave_val_l1
#                 best_epoch = epoch + 1
#             print("current best val loss: " + str(best_val) + " epoch: " + str(best_epoch))
#         torch.save(net.state_dict(), 'lenet_epoch10.pt')

#     print('LeNet5 Training done.')
#     print("Final best val loss: " + str(best_val) + " epoch: " + str(best_epoch))


















# # reduce lr on plateau:
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import time
# import os
# import csv

# from Resnet_new import newResNet18
# from Lenet import LeNet5
# from Loader_new3 import getTrain1, getVal1

# if __name__ == '__main__':
#     device = ('cuda' if torch.cuda.is_available() else 'cpu')

#     trainloader = getTrain1(num_chan=1, batch_size=47)
#     valloader = getVal1(num_chan=1)

#     ## cnn
#     net = newResNet18(num_chan=1, dropout=0.34739) # dropout=0.154
#     net.to(device=device)

#     ## loss and optimizer
#     criterion = torch.nn.MSELoss()
#     criterion2 = torch.nn.L1Loss()
#     optimizer = optim.Adam(net.parameters(), lr=0.000039262, weight_decay=0.22) # lr=0.00024

#     # Add ReduceLROnPlateau scheduler
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

#     ## train
#     with open(os.path.join("resnet18_1_trial4_10.csv"), mode='w', newline='') as file:
#         csv_writer = csv.writer(file)
#         csv_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', "Validation L1 Loss"])
#         t0 = time.perf_counter()
#         best_val = 20
#         for epoch in range(10):  # loop over the dataset multiple times # 10 epochs
#             print("Training epoch " + str(epoch + 1))
#             t1 = time.perf_counter()
#             running_loss = 0.0
#             total_loss = 0.0
#             step = 0
#             net.train()
#             for i, data in enumerate(trainloader, 0):
#                 inputs, labels = data
#                 inputs = inputs.to(device=device)
#                 labels = labels.to(device=device)
#                 optimizer.zero_grad()

#                 # forward + backward + optimize
#                 outputs = net(inputs)
#                 loss = criterion(outputs, labels.unsqueeze(1))
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()
#                 total_loss += loss.item()
#                 if i % 200 == 199:  # print every 200 mini-batches
#                     print('[%d, %5d] loss: %.3f' %
#                           (epoch + 1, i + 1, running_loss / 200))
#                     running_loss = 0.0
#                 step += 1

#             print("Evaluating epoch " + str(epoch + 1))
#             net.eval()
#             with torch.no_grad():
#                 val_steps = 0
#                 val_loss = 0.0
#                 val_l1loss = 0.0
#                 for j, data in enumerate(valloader):
#                     val_imgs, val_labels = data
#                     val_imgs = val_imgs.to(device)
#                     val_labels = val_labels.to(device)
#                     val_outputs = net(val_imgs)
#                     loss = criterion(val_outputs, val_labels.unsqueeze(1)).item()
#                     l1loss = criterion2(val_outputs, val_labels.unsqueeze(1)).item()
#                     val_loss += loss
#                     val_l1loss += l1loss
#                     val_steps += 1
#             speed = round(time.perf_counter() - t1)
#             ave_val = val_loss / val_steps
#             ave_val_l1 = val_l1loss / val_steps
#             print('[Epoch %d] training loss: %.3f validation loss: %.3f validation l1 loss: %.3f time taken: %ds' % (
#                 epoch + 1, total_loss / step, ave_val, ave_val_l1, speed))
#             csv_writer.writerow([epoch + 1, total_loss / step, ave_val, ave_val_l1])

#             # Step the scheduler
#             scheduler.step(ave_val)

#             if ave_val_l1 < best_val:
#                 model_name = 'resnet18_1_trial4_10.pt'
#                 torch.save(net.state_dict(), model_name)
#                 best_val = ave_val_l1
#                 best_epoch = epoch + 1
#             print("current best val loss: " + str(best_val) + " epoch: " + str(best_epoch))
#         torch.save(net.state_dict(), 'resnet18_1_last_trial4_10.pt')

#     print('Resnet18 1_trial4_10 Training done.')
#     print("Final best val loss: " + str(best_val) + " epoch: " + str(best_epoch))
