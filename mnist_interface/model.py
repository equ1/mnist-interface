import os
import torch
import torch.nn as nn
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
        self.layer4 = nn.Sequential(
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def transform_input(img):
    '''
    transforms image
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((28, 28)), transforms.Normalize((0.1307,), (0.3081,))])
    return transform(img)


def visualize_metric(model_name, tags, values, epoch):
    # Declares Tensorboard writer
    writer = SummaryWriter(f"runs/{model_name}")

    # Adds scalars to Tensorboard viz
    for tag, value in zip(tags, values):
        writer.add_scalar(tag, value, epoch)

    writer.close()


def data_generator(transform_input, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST):
    # Creates dataset
    train_set = datasets.MNIST('/data/', train=True, download=False,
                               transform=transform_input)
    test_set = datasets.MNIST('/data/', train=False, download=False,
                              transform=transform_input)
    train_set_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_set_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    return (train_set, test_set, train_set_loader, test_set_loader)


def train(model, train_set, train_set_loader, opt, criterion, epoch, model_name, save_model, visualize):
    # Training
    model.train()

    total_train_loss = 0
    total_correct = 0

    for i, (images, labels) in enumerate(tqdm(train_set_loader)):
        opt.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        total_train_loss += loss.item()

        loss.backward()
        opt.step()

        # Calculates train accuracy
        outputs_probs = nn.functional.softmax(
            outputs, dim=1)  # gets probabilities
        for idx, preds in enumerate(outputs_probs):
            # if label with max probability matches true label
            if labels[idx] == torch.argmax(preds.data):
                total_correct += 1

    train_loss = total_train_loss/(i+1)
    train_accuracy = total_correct/len(train_set)

    print(f"Train set:- Loss: {train_loss}, Accuracy: {train_accuracy}.")

    # Training viz
    if visualize:
        visualize_metric(model_name, ["Train/Loss", "Train/Accuracy"],
                         [train_loss, train_accuracy], epoch)

    # Saves model state
    if save_model:
        torch.save(model.state_dict(), os.getcwd() +
                   f"/mnist_interface/saved_models/{model_name}.pt")


def test(model, test_set, test_set_loader, opt, criterion, epoch, model_name, visualize):
    # Testing
    model.eval()

    total_test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_set_loader)):
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            outputs_probs = nn.functional.softmax(outputs, dim=1)
            for idx, preds in enumerate(outputs_probs):
                if labels[idx] == torch.argmax(preds.data):
                    total_correct += 1

    test_loss = total_test_loss/(i+1)
    test_accuracy = total_correct/len(test_set)

    print(f"Test set:- Loss: {test_loss}, Accuracy:{test_accuracy}.")

    # Testing viz
    if visualize:
        visualize_metric(model_name, ["Test/Loss", "Test/Accuracy"],
                         [test_loss, test_accuracy], epoch)


def main():
    # Parameters
    EPOCHS = 3
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_TEST = 64
    LRN_RATE = 0.003
    model_name = f"mnist-cnn-{time.time()}"

    # Creates train/test split
    train_set, test_set, train_set_loader, test_set_loader = data_generator(
        transform_input, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

    # Creates model
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=LRN_RATE)
    criterion = nn.NLLLoss()
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}.")
        train(model, train_set, train_set_loader,
              optimizer, criterion, epoch, model_name, save_model=True, visualize=True)
        test(model, test_set, test_set_loader,
             optimizer, criterion, epoch, model_name, visualize=True)


if __name__ == "__main__":
    main()
