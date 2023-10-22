import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
import sys
from os import path
from PIL import Image
import pickle

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Sequential(
            # input:1*28*28, output: 32*24*24
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            # input:32*24*24, output: 32*12*12
            nn.MaxPool2d(2))
        # input:32*12*12, output:
        self.conv2 = nn.Sequential(
            # input:32*12*12, output: 64*8*8
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            # input:64*8*8, output: 64*4*4
            nn.MaxPool2d(2)
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x1=x
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output,x1

# Define the CNN architecture
class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        # input:3*32*32, output: 32*16*16
        self.conv1 = nn.Sequential(
            # input:3*32*32, output: 32*32*32
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2, # keep same shape
            ),
            # Batch Normalization
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # input:3*32*32, output: 32*16*16
            nn.MaxPool2d(2),
        )
        # input:32*16*16, output: 64*8*8
        self.conv2 = nn.Sequential(
            # input:32*16*16, output: 64*16*16
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # input:64*16*16, output: 64*8*8
            nn.MaxPool2d(2),
        )
        # input:64*8*8, output: 128*4*4
        self.conv3 = nn.Sequential(
            # input:64*8*8, output: 128*8*8
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # input:128*8*8, output: 128*4*4
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 50)
        self.out = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Image size: 32x32, 3 channels
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.out(x)
        return output, x1


# Define the training function
def train(dataset):
    # dataset loading
    if dataset == 'mnist':
        BATCH_SIZE = 64
        EPOCHS = 5
        train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(),download=True)
        test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())
    elif dataset == 'cifar':
        # Define batch size and number of epochs
        BATCH_SIZE = 128
        EPOCHS = 40
        train_data = torchvision.datasets.CIFAR10(root='./data.cifar10', train=True,transform=torchvision.transforms.ToTensor(), download=True)
        test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False,transform=torchvision.transforms.ToTensor())
    else:
        raise ValueError("Invalid dataset choice. Please choose from 'mnist' or 'cifar'.")

    # Mini-batch configuration
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)

    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    # Model Initialization
    if dataset == 'mnist':
        model = MnistCNN()

    elif dataset == 'cifar':
        model = CifarCNN()

    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)
    # Loss function: cross entropy
    loss_func = torch.nn.CrossEntropyLoss()

    print('\nEpoch  Batch   Train loss   Train acc%   Test loss   Test acc%')

    for epoch in range(EPOCHS):
        # Use mini-batch update parameters
        for batch, (input, target) in enumerate(train_loader):
            # Set model
            model.train()
            # Forward pass for this batch's inputs
            output, x1 = model(input)
            loss = loss_func(output, target)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Weights update
            optimizer.step()

            # Output training results for every 200 batches
            if batch % 200 == 0:
                # Set model in evaluation mode
                model.eval()
                # Test set evaluation
                test_loss = 0
                test_correct = 0
                for data, target in test_loader:
                    # Forward pass
                    output, x1 = model(data)
                    # Also use the Cross Entropy Loss criterion for the test loss
                    criterion = nn.CrossEntropyLoss()
                    test_loss = criterion(output, target)
                    # Get the index of the max log-probability
                    pred = output.data.max(1, keepdim=True)[1]
                    test_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # Compute the total test loss and accuracy
                test_loss /= len(test_loader.dataset)
                test_acc = float(100. * test_correct) / float(len(test_loader.dataset))

                # Train set evaluation
                train_loss = 0
                train_correct = 0
                for data, target in train_loader:
                    # Forward pass
                    output, x1 = model(data)
                    criterion = nn.CrossEntropyLoss()
                    train_loss = criterion(output, target)
                    pred = output.data.max(1, keepdim=True)[1]
                    train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # Compute the total train loss and accuracy
                train_loss /= len(train_loader.dataset)
                train_acc = float(100. * train_correct) / float(len(train_loader.dataset))

                # Print this batch's results
                print(
                    '  {}      {}\t{:.8f}      {:.2f}%\t{:.8f}    {:.2f}%'
                        .format(epoch + 1, batch, train_loss, train_acc,
                                test_loss, test_acc))

        # At the end of each epoch, if the performance better than threshold(75%), save the model. Cover the model if there is better performance
        best_acc = 75
        if test_acc > best_acc:
            best_acc = test_acc
            if dataset == 'mnist':
                torch.save(model, 'model/mnist.pt')
            elif dataset == 'cifar':
                torch.save(model, 'model/cifar.pt')

def test(img_path):


    mnist = torch.load('model/mnist.pt')
    cifar = torch.load('model/cifar.pt')

    #Mnist
    # Load test image
    img1 = Image.open(img_path).convert('L')
    # Resize it to the Mnist size
    img_r = img1.resize((28, 28))
    # Convert it to a tensor
    trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
    img_r_t = trans(img_r).unsqueeze(0)

    # test on the input image
    mnist.eval()
    result, x1 = mnist(img_r_t)
    result = result.data[0].tolist()
    # Softmax for each class
    result_ind = result.index(max(result))
    print('Prediction result: {}'.format(result_ind))

    # Generate the first CONV1 layer visualization
    plt.figure(figsize = (6, 6))
    # Conv1 has 32 channels, so there are 32 figures
    for i in range(32):
        ax = plt.subplot(6, 6, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        img_conv = torchvision.transforms.ToPILImage()(x1[0][i])
        plt.imshow(img_conv, cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('CONV_rslt_mnist.png')
    plt.show()

    #Cifar
    # Load test image
    img2 = Image.open(img_path).convert('RGB')
    # Resize it to the CIFAR size
    img_r = img2.resize((32, 32))
    # Convert it to a tensor
    trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
    img_r_t = trans(img_r).unsqueeze(0)

    # Load the label names from the CIFAR database
    label_names = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
                  4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

    # test on the input image
    cifar.eval()
    result, x1 = cifar(img_r_t)
    result = result.data[0].tolist()
    # Softmax for each class
    result_ind = result.index(max(result))
    result_lab = label_names[result_ind]
    print('Prediction result: {}'.format(result_lab))

    # Generate the first CONV1 layer visualization
    plt.figure(figsize = (6, 6))
    # Conv1 has 32 channels, so there are 32 figures
    for i in range(32):
        ax = plt.subplot(6, 6, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        img_conv = torchvision.transforms.ToPILImage()(x1[0][i])
        plt.imshow(img_conv, cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('CONV_rslt_cifar.png')
    plt.show()

if __name__ == "__main__":
    task = str(sys.argv[1])
    if task == 'train':
        if str(sys.argv[2]) =='--mnist':
            train('mnist')
        elif str(sys.argv[2]) =='--cifar':
            train('cifar')
    elif task == 'test':
        test(str(sys.argv[2]))