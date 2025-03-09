import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
from keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder

# Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST Model with Wandb Logging")
    parser.add_argument("-wp", "--wandb_project", type=str, default="da24m029-da6401-assignment1", help="Project name for Wandb tracking")
    parser.add_argument("-we", "--wandb_entity", type=str, default="da24m029-indian-institute-of-technology-madras", help="Wandb Entity for tracking")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("-nhl", "--num_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of neurons per hidden layer")
    
    return parser.parse_args()

# Activation Functions and Derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size):
        self.layers = [input_size] + [hidden_size] * hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.01 for i in range(len(self.layers)-1)]
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights) - 1):
            X = sigmoid(np.dot(X, self.weights[i]) + self.biases[i])
            activations.append(X)
        output = softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        activations.append(output)
        return activations

    def backward(self, activations, y_true):
        grads_w, grads_b = [], []
        loss_grad = activations[-1] - y_true  # Softmax-CrossEntropy Derivative
        
        for i in range(len(self.weights)-1, -1, -1):
            grads_w.append(np.dot(activations[i].T, loss_grad))
            grads_b.append(np.sum(loss_grad, axis=0, keepdims=True))
            if i > 0:
                loss_grad = np.dot(loss_grad, self.weights[i].T) * sigmoid_derivative(activations[i])
        
        return grads_w[::-1], grads_b[::-1]  # Reverse to match weight order

    def update_weights(self, grads_w, grads_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]

# Training Function
def train_nn(args, X_train, y_train):
    input_size = X_train.shape[1]
    output_size = 10  # Fashion-MNIST has 10 classes
    nn = NeuralNetwork(input_size, args.num_layers, args.hidden_size, output_size)

    num_samples = X_train.shape[0]
    losses = []

    for epoch in range(args.epochs):
        total_loss = 0
        for i in range(0, num_samples, args.batch_size):
            X_batch = X_train[i:i + args.batch_size]
            y_batch = y_train[i:i + args.batch_size]

            activations = nn.forward(X_batch)
            grads_w, grads_b = nn.backward(activations, y_batch)
            nn.update_weights(grads_w, grads_b, args.learning_rate)

            batch_loss = -np.sum(y_batch * np.log(activations[-1] + 1e-9)) / args.batch_size
            total_loss += batch_loss

        avg_loss = total_loss / (num_samples / args.batch_size)
        losses.append(avg_loss)
        wandb.log({"Epoch": epoch + 1, "Loss": avg_loss})

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

# Main Function
def main():
    args = get_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name="fashion_mnist_training")

    # Load Fashion-MNIST dataset
    (x_train, y_train), _ = fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # Flatten and normalize

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    # Train the neural network
    train_nn(args, x_train, y_train)

    wandb.finish()

# Run script
if __name__ == "__main__":
    main()
