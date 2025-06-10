import os
import random

from Global.Utils import FacesDataset
from Global.visualisation import Plot, plot_example
from Simplest_autoencoder import Autoencoder
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# # Download and load MNIST training and testing dataset
mnist_train = datasets.MNIST(root=r'../Global/data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root=r'../Global/data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

dim_in = len(mnist_train[0][0].squeeze().flatten())
dim_out = dim_in

# set up the model and criterion
device = "cpu" #torch.device("mps" if torch.mps.is_available() else "cpu")
print("running on {}".format(device))
dimensions = [dim_in, dim_in//2, dim_in//4, 64, dim_out//4, dim_out//2, dim_out]
dimensions = [int(x) for x in dimensions]
print(dimensions)
model = Autoencoder(dimensions).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.0005)

# set up the training loop
plot = Plot()
plot_num = [random.randint(0, len(train_loader)) for _ in range(1)]
for epoch in range(30):

    def Train_loop():
        train_loss = 0
        for x, _ in tqdm(train_loader):
            optimizer.zero_grad()  # for correct weight update

            # model forward
            x = x.to(device).squeeze().flatten(start_dim = 1)  # preparing the input
            z = model(x)
            loss = criterion(z, x)  # compute loss

            loss.backward()  # update gradients
            optimizer.step()  # taking an Adam step

            train_loss += loss.item()
        train_loss /= len(train_loader)
        return train_loss
    train_loss = Train_loop()

    def Validation_loop():
        validation_loss = 0
        for i, (x, _) in tqdm(enumerate(test_loader)):
            optimizer.zero_grad()  # for correct weight update

            # model forward
            x = x.to(device).squeeze().flatten(start_dim = 1)  # preparing the input
            z = model(x)
            loss = criterion(z, x)  # compute loss
            validation_loss += loss.item()

        validation_loss /= len(test_loader)
        return test_loader

    validation_loss = Validation_loop()



    #plot example
    (x, _) = mnist_test[plot_num[0]]
    x = x.to(device).squeeze().flatten()  # preparing the input
    z = model(x)
    plot_example(x, z)

    os.makedirs(f"./models", exist_ok=True)
    torch.save({"state_dict":model.state_dict(),
                "dimensions":dimensions,},
               f"./models/Autoencoder_{len(dimensions)-2}_layers.pth")

    plot.update(train_loss, validation_loss)
plot.show()