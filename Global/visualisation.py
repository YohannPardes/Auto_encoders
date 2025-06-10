from matplotlib import pyplot as plt

def img_from_tensor(tensor, shape=(28, 28)):
    tensor = tensor.reshape(*shape)

    # Handle different tensor shapes
    if len(tensor.shape) == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # Remove channel dimension if it's 1
        elif tensor.shape[0] == 3:
            # if it has 3 channels it is assumed to be a grayscale image where each channel is equal
            tensor = tensor.mean(dim=0)
        else:
            raise ValueError("Tensor shape is not valid for grayscale image")
    elif len(tensor.shape) != 2:
        raise ValueError("Tensor shape is not valid for grayscale image")

    # Ensure tensor is on CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert to NumPy array
    image = tensor.cpu().detach().numpy()

    return image

def plot_example(x, z):

    x_img = img_from_tensor(x, shape=(28, 28))
    z_img = img_from_tensor(z, shape=(28, 28))

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image in the first subplot
    axes[0].imshow(x_img, cmap='gray')
    axes[0].axis('off')  # Turn off axis labels

    # Display the second image in the second subplot
    axes[1].imshow(z_img, cmap='gray')
    axes[1].axis('off')  # Turn off axis labels

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

class Plot:
    def __init__(self, label1="Train", label2="Validation"):
        self.x_data = []
        self.y1_data = []
        self.y2_data = []
        self.index = 0
        self.label1 = label1
        self.label2 = label2

    def update(self, val1, val2):
        self.index += 1
        self.x_data.append(self.index)
        self.y1_data.append(val1)
        self.y2_data.append(val2)

    def show(self):
        plt.figure()
        plt.plot(self.x_data, self.y1_data, marker='o', label=self.label1)
        plt.plot(self.x_data, self.y2_data, marker='s', label=self.label2)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Final Plot: Train vs Validation loss")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    pass