import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.CIFAR10 import CIFAR10
from models.cnn_simple import CNNSimple

if __name__ == "__main__":
    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    testset = CIFAR10(root="../data", train=False, transform=transformer)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    net = CNNSimple()
    net.load_state_dict(torch.load("../models/cnn_simple_12-01-23-18_39.pth"))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )
