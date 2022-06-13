import torch 
from dataloader_v2 import CustomDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import CNNModel, CNNModel_Small, MyModel
import matplotlib.pyplot as plt
from utils import img_convert
import time


def inference(model, image, device):

    with torch.no_grad():
        
        image = image.to(device=device)
        scores_class = model(image)
        _, prediction_label = scores_class.max(1)
            
    return prediction_label

transform_test = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor()
                                      ])

test_dataset = CustomDataset(
                            root_dir = 'data/data_latest', 
                            transform=None)

test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

model = CNNModel_Small(5)
model.load_state_dict(torch.load('checkpoints/checkpoint_39.pth', map_location='cpu'))
model.eval()
model = model.float()

imgs, targets = next(iter(test_loader))

label_names = ['Blue square', 'Blue triangle', 'Blue circle', 
               'Green square', 'Red Square']


summ = 0
for img,target in zip(imgs,targets):
    img_i = img.unsqueeze(0)
    s = time.time()
    label = inference(model, img_i.float(), 'cpu')
    if label == target:
        summ += 1
    e = time.time()

    print("Time for one forward pass: ",100*(e-s), " ms")
    img = img_convert(img)
    plt.imshow(img)
    plt.title(f"Predicted: {label_names[label]} | Real: {label_names[target]}")
    plt.savefig("prediction.png")
    # plt.show()


print("Accuracy: ", 100*summ/len(targets))