import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
from keras.datasets import fashion_mnist

# Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST Model with Wandb Logging")
    parser.add_argument("-wp", "--wandb_project", type=str, default="da24m029-da6401-assignment1", help="Project name for Wandb tracking")
    parser.add_argument("-we", "--wandb_entity", type=str, default="da24m029-indian-institute-of-technology-madras", help="Wandb Entity for tracking")
    
    # Placeholder for future params (e.g., epochs, batch size, etc.)
    return parser.parse_args()

# Main Function
def main():
    args = get_args()

    # Initialize Wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name="fashion_mnist_samples")

    # Load Fashion-MNIST dataset
    (x_train, y_train), _ = fashion_mnist.load_data()

    # Class labels in Fashion-MNIST
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Select one image per class
    samples = []
    for class_idx in range(10):
        idx = np.where(y_train == class_idx)[0][0]
        samples.append((x_train[idx], class_names[class_idx]))

    # Plot images in a 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Fashion-MNIST Sample Images", fontsize=14)

    for ax, (img, label) in zip(axes.flatten(), samples):
        ax.imshow(img, cmap="gray")
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()

    # Log plot to Wandb
    wandb.log({"Fashion-MNIST Samples": wandb.Image(fig)})

    # Show plot
    plt.show()

    # Finish Wandb run
    wandb.finish()

# Run script
if __name__ == "__main__":
    main()
