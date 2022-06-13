import torch 
from model import CNNModel_Small, CNNModel
import matplotlib.pyplot as plt
from utils import img_convert
import time 
from picamera import PiCamera
import numpy as np


def predict(model, image, device):
    image = image.permute(0, 3,1,2).float()
    with torch.no_grad():
        
        image = image.to(device=device)
        scores_class = model(image)
        print(scores_class)
        _, prediction_label = scores_class.max(1)
            
    return prediction_label

label_names = ['Blue square', 'Blue triangle', 'Blue circle', 
               'Green square', 'Red Square']

model = CNNModel(5)
model.load_state_dict(torch.load('checkpoints/checkpoint_transferlearning.pth', map_location='cpu'))
model.eval()
model = model.float()

camera = PiCamera()
camera.resolution = (32,32)

for i in range(50):
    # Take image from PiCamera
    s = time.time()
    img = np.empty((32, 32, 3), dtype=np.uint8)
    camera.capture(img, 'rgb')

    p_label = predict(model, torch.tensor(img).unsqueeze(0), 'cpu')
    print(label_names[p_label])
    plt.imshow(img)
    plt.title(f"Predicted: {label_names[p_label]}")
    plt.savefig("prediction.png")
    e = time.time()
    time.sleep(1)
    print("Time: ", 100*(e-s))


