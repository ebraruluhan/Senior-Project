import pandas as pd 
import torch
import os
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import img_convert

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir, str(self.annotations.iloc[index, 0])+'.jpg')
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label


if __name__ == '__main__':
    # Load data 
    transform_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      transforms.ToTensor()
                                      ])

    dataset = CustomDataset(csv_file = "/Users/hcagri/Documents/ComputerVision/projects/ebrar/data/annotations.csv",
                            root_dir = '/Users/hcagri/Documents/ComputerVision/projects/ebrar/data/images', 
                            transform=transform_train)

    #train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    
    print(labels.shape)

    #test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)


