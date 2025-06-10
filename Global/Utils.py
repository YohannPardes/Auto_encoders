from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_lfw_people
class MyDataLoader(Dataset):

    def __init__(self):
        super().__init__()
        ### useless for now ###

# Custom PyTorch Dataset for Autoencoders
class FacesDataset(Dataset):
    def __init__(self, min_faces_per_person=0, resize=0.5):
        lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
        self.images = lfw.images
        self.transform = transforms.Compose([
            transforms.ToTensor(),                # [H, W] → [1, H, W]
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img)
        return img

# Create DataLoader
def get_faces_dataloader(batch_size=32):
    dataset = FacesDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

