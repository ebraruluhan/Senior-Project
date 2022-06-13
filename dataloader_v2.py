import torch
import os
from torch.utils.data import Dataset
from skimage import io
from skimage.color import rgba2rgb
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils import img_convert
import PIL.Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        img_names = os.listdir(root_dir)
        for name in img_names:
            root, ext = os.path.splitext(name)
            if ext != 'png':
                img_names.remove(name)
        
        self.img_names = img_names

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self,index):
        name = self.img_names[index]
        img_path = os.path.join(self.root_dir, name)
        image = torch.tensor(rgba2rgb(io.imread(img_path)))

        root, _ = os.path.splitext(name)
        y_label = torch.tensor(int(root.split('_')[-1]))
        y_label = y_label.type(torch.LongTensor)
        image = image.permute(2,0,1)
        if self.transform:
            image = self.transform(image)

        return image, y_label


if __name__ == '__main__':
    # Load data 
    transform_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      transforms.ToTensor()
                                      ])

    dataset = CustomDataset(
                            root_dir = '/Users/hcagri/Documents/DeepLearning/projects/ebrar/data/data_latest', 
                            transform=transform_train)

    #train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    
    print(labels.shape)

    #test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)


