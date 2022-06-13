import torch
from tqdm import tqdm


def img_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  return image

def check_accuracy(loader, model, device='cpu'):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        loop = loader
        for x, labels in loop:
            x = x.to(device=device)
            labels = labels.to(device=device)

            scores_class = model(x.float())
            _, predictions = scores_class.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

        accuracy = num_correct/num_samples
            # loop.set_postfix(acc=100*accuracy.item())

    print("Final Accuracy: {:.2f}".format(100*accuracy))
    model.train()
    return accuracy