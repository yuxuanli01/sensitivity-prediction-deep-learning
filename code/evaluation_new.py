import torch
import csv

from Resnet_new import newResNet18
from Lenet import LeNet5
from Loader_new3 import getTest1

device = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.MSELoss()
criterion2 = torch.nn.L1Loss()

testloader = getTest1(num_chan=1)

net=LeNet5(num_chan=1)
net.load_state_dict(torch.load('lenet_epoch10.pt'))
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
print(loss, l1loss) # 45.953372955322266 5.196100234985352


with open('evaluate_lenet_epoch10.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(test_labels.cpu().numpy(), test_outputs.squeeze(1).cpu().numpy()))


net=newResNet18(num_chan=1)
net.load_state_dict(torch.load('resnet18_setting1_10.pt'))
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

with open('evaluate_resnet_setting1_10.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(test_labels.cpu().numpy(), test_outputs.squeeze(1).cpu().numpy()))

#
# testloader = getTest(num_chan=3)
#
# net=LeNet5(num_chan=3)
# net.load_state_dict(torch.load('lenet_3.pt'))
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
# print(loss, l1loss)
#
# with open('outputs_lenet_3.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(zip(test_labels.cpu().numpy(), test_outputs.squeeze(1).cpu().numpy()))
#
# net=ResNet18(num_chan=3)
# net.load_state_dict(torch.load('resnet18_3.pt'))
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
# print(loss, l1loss)
#
# with open('outputs_resnet_3.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(zip(test_labels.cpu().numpy(), test_outputs.squeeze(1).cpu().numpy()))
'''
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
print(val_loss/val_steps, val_l1loss/val_steps)

next_val = next(iter(valloader))
next_imgs, next_labels = next_val
next_imgs = next_imgs.to(device)
next_labels = next_labels.to(device)
print(next_labels)
print(net(next_imgs))

'''
