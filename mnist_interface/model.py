import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import time


class Net(nn.Module):
    '''
    model definition
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Dropout(),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_num_correct(preds, labels):
    '''
    returns number of predictions that match their respective labels
    '''
    return preds.argmax(dim=1).eq(labels).sum().item()


def transform_input(img):
    '''
    transforms image
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return transform(img)


def train(model, train_set, train_set_loader, opt, epoch, writer):
    # Training
    model.train()

    total_correct = 0

    for images, labels in tqdm(train_set_loader):
        opt.zero_grad()
        preds = model(images)
        loss = F.nll_loss(preds, labels)
        loss.backward()
        opt.step()

        total_correct += get_num_correct(preds, labels)

    accuracy = total_correct/len(train_set)

    print(f"Train set:- Loss: {loss.item()}, Accuracy: {accuracy}.")

    # Training viz
    writer.add_scalar("Train/Loss", loss.item(), epoch)
    writer.add_scalar("Train/Accuracy", accuracy, epoch)


def test(model, test_set, test_set_loader, opt, epoch, writer):
    # Testing
    model.eval()

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(test_set_loader):
            outputs = model(images)
            test_loss += F.nll_loss(outputs, labels, reduction='sum').item()
            total_correct += get_num_correct(outputs, labels)
        test_loss /= len(test_set)

    accuracy = total_correct/len(test_set)

    print(f"Test set:- Avg. Loss: {test_loss}, Accuracy:{accuracy}.")

    # Testing viz
    writer.add_scalar("Test/Loss", test_loss, epoch)
    writer.add_scalar("Test/Accuracy", accuracy, epoch)


def main():
    # Parameters
    EPOCHS = 10
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_TEST = 1000
    LRN_RATE = 0.001

    # Creates dataset
    train_set = datasets.MNIST('/data/', train=True, download=False,
                               transform=transform_input)
    test_set = datasets.MNIST('/data/', train=False, download=False,
                              transform=transform_input)
    train_set_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_set_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    # Declares Tensorboard writer
    model_name = f"mnist-cnn-{time.time()}"
    writer = SummaryWriter(f"runs/{model_name}")
    print(f"\nTensorboard is recording into folder: runs/{model_name}.")

    # Visualizes first batch of train set images
    dataiter = iter(train_set_loader)
    images = dataiter.next()[0]
    grid = utils.make_grid(images)
    writer.add_image('Dataset/Images', grid, 0)
    writer.close()

    # Creates model and optimizer
    model = Net()
    opt = optim.Adam(model.parameters(), lr=LRN_RATE)
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}.")
        train(model, train_set, train_set_loader, opt, epoch, writer)
        test(model, test_set, test_set_loader, opt, epoch, writer)
        writer.close()

    # Saves model state
    torch.save(model.state_dict(), os.getcwd() +
               f"/mnist_interface/saved_models/{model_name}.pt")


if __name__ == "__main__":
    main()
