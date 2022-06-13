import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as transforms 
from dataloader_v2 import CustomDataset
from model import CNNModel, CNNModel_Small
from utils import check_accuracy

BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-2
NUM_CLASSES = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
transform_train = transforms.Compose([transforms.ToTensor(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      
                                      ])
'''                           

# Load data 
'''
train_dataset = CustomDataset(csv_file = "/Users/hcagri/Documents/ComputerVision/projects/ebrar/data/annotations.csv",
                        root_dir = '/Users/hcagri/Documents/ComputerVision/projects/ebrar/data/new', 
                        transform=transform_train
                        )
'''
train_dataset = CustomDataset(root_dir = 'data\data_latest', 
                              transform=None
                            )

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNNModel_Small(NUM_CLASSES).to(device)

'''
state_dict = torch.load('pretrained/checkpoint_37.pth', map_location='cpu')
del state_dict['fc.3.weight']
del state_dict['fc.3.bias']

model.load_state_dict(state_dict, strict=False)

for param in model.conv_net.parameters():
    param.requires_grad = False
'''
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

model = model.float()

for epoch in range(EPOCHS):
    loop = tqdm(train_loader)
    epoch_loss = []

    for data, labels in loop:
        # Get data to cuda if possible
        data = data.to(device=device)
        labels = labels.to(device=device)
        
        
        # forward
        score = model(data.float())
        loss = criterion(score, labels)
        
        epoch_loss.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
      
    if epoch+1 == 15:
        optimizer.param_groups[0]['lr'] = 0.0002
    if epoch+1 == 30:
        optimizer.param_groups[0]['lr'] = 0.00002
    
    if (epoch+1)%5 == 0:
        torch.save(model.state_dict(), f'checkpoints/checkpoint_{epoch}.pth')
    print("\n")
    mean_loss = sum(epoch_loss) / len(epoch_loss)
    print("Mean Loss of Epoch = {:.2f}".format(mean_loss))
    train_acc = check_accuracy(train_loader, model, device)
    print("\n")
    


