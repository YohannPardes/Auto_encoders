import random
import torch
from MNIST.Simplest_autoencoder import Autoencoder
from visualisation import img_from_tensor, plot_example
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# loading pretrained model
checkpoint = torch.load(r'../MNIST/models/Autoencoder_5_layers.pth')
model = Autoencoder(checkpoint["dimensions"])
model.load_state_dict(checkpoint['state_dict'])
encoder = model.encoder
decoder = model.decoder

# picking 2 samples from mnist
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
random_samples = [mnist_train[random.randint(0, len(mnist_train) - 1)] for _ in range(3)]
random_samples = [random_sample[0].squeeze().flatten() for random_sample in random_samples]

# extracting latent vectors from the encoder
latent_vector_1 = encoder.forward(random_samples[0])
latent_vector_2 = encoder.forward(random_samples[1])
latent_vector_3 = encoder.forward(random_samples[2])

#decode back to the full image
out_1 = decoder.forward(latent_vector_1)
out_2 = decoder.forward(latent_vector_2)
out_3 = decoder.forward(latent_vector_3)
plot_example(out_1, out_2)

# plot the path from the first and second vector in the latent space
def move_from_a_to_b(a, b, steps=50, save=r"data/Animation"):
    """given two vector in the latent space
    render the steps from vector a to b in the latent space
    """
    difference = b - a
    step_vector = difference / steps
    for i in range(steps):
        a += step_vector
        out = decoder.forward(a)
        img = img_from_tensor(out)
        plt.imshow(img, cmap='gray')
        if save:
            plt.savefig(save + "/" + 'step' + str(i) + '.png')
        plt.show()
move_from_a_to_b(latent_vector_1, latent_vector_2)
# move_from_a_to_b(latent_vector_2, latent_vector_3)