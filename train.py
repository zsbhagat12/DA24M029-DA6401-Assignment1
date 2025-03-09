import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
from keras.datasets import fashion_mnist

# Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="Train Neural Network with Wandb Logging")

    # Arguments as per Code Specifications
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name for Wandb tracking")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb Entity for tracking")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for momentum/NAG")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for RMSprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 for Adam/Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 for Adam/Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight Initialization")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of neurons per hidden layer")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function")

    # Additional Arguments
    parser.add_argument("--sweep", action="store_true", help="Run Wandb sweep instead of normal training")

    return parser.parse_args()


# Activation Functions 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity(x):
    return x

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Activation Function Dictionary
activation_functions = {
    "identity": identity,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "ReLU": relu
}

# Derivative Functions
def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    return x * (1 - x)

activation_derivatives = {
    "identity": lambda x: np.ones_like(x),
    "sigmoid": sigmoid_derivative,
    "tanh": lambda x: 1 - np.tanh(x) ** 2,
    "ReLU": relu_derivative
}

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# One-Hot Encoding Function using NumPy
def one_hot_numpy(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))  # Create a zero matrix
    one_hot[np.arange(y.shape[0]), y] = 1  # Set appropriate indices to 1
    return one_hot

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, optimizer, args):
        self.layers = [input_size] + [hidden_size] * hidden_layers + [output_size]

        if args.weight_init == "random":
            self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.01 for i in range(len(self.layers)-1)]
        elif args.weight_init == "Xavier":
            self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1 / self.layers[i]) for i in range(len(self.layers)-1)]
        
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]

        # Optimizer parameters
        self.optimizer = optimizer
        self.args = args

        # Momentum-based optimizers
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]

        # RMSProp / Adam / Nadam parameters
        self.s_w = [np.zeros_like(w) for w in self.weights]
        self.s_b = [np.zeros_like(b) for b in self.biases]
        self.t = 1  # Time step for Adam/Nadam

    def forward(self, X):
        activations = [X]
        activation_func = activation_functions[self.args.activation]
        
        for i in range(len(self.weights) - 1):
            X = activation_func(np.dot(X, self.weights[i]) + self.biases[i])
            activations.append(X)

        output = softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        activations.append(output)
        return activations


    def backward(self, activations, y_true):
        grads_w, grads_b = [], []
        loss_grad = activations[-1] - y_true  # Softmax-CrossEntropy Derivative

        activation_derivative = activation_derivatives[self.args.activation]

        for i in range(len(self.weights)-1, -1, -1):
            grads_w.append(np.dot(activations[i].T, loss_grad))
            grads_b.append(np.sum(loss_grad, axis=0, keepdims=True))
            if i > 0:
                loss_grad = np.dot(loss_grad, self.weights[i].T) * activation_derivative(activations[i])

        return grads_w[::-1], grads_b[::-1] # Reverse to match weight order

    def update_weights(self, grads_w, grads_b):
        lr = self.args.learning_rate
        beta1, beta2, eps = self.args.beta1, self.args.beta2, self.args.epsilon

        for i in range(len(self.weights)):
            if self.optimizer == "sgd":
                self.weights[i] -= lr * grads_w[i]
                self.biases[i] -= lr * grads_b[i]

            elif self.optimizer == "momentum":
                self.v_w[i] = self.args.momentum * self.v_w[i] - lr * grads_w[i]
                self.v_b[i] = self.args.momentum * self.v_b[i] - lr * grads_b[i]
                self.weights[i] += self.v_w[i]
                self.biases[i] += self.v_b[i]

            elif self.optimizer == "nag":
                v_prev_w, v_prev_b = self.v_w[i], self.v_b[i]
                self.v_w[i] = self.args.momentum * self.v_w[i] - lr * grads_w[i]
                self.v_b[i] = self.args.momentum * self.v_b[i] - lr * grads_b[i]
                self.weights[i] += -self.args.momentum * v_prev_w + (1 + self.args.momentum) * self.v_w[i]
                self.biases[i] += -self.args.momentum * v_prev_b + (1 + self.args.momentum) * self.v_b[i]

            elif self.optimizer == "rmsprop":
                self.s_w[i] = beta1 * self.s_w[i] + (1 - beta1) * (grads_w[i] ** 2)
                self.s_b[i] = beta1 * self.s_b[i] + (1 - beta1) * (grads_b[i] ** 2)
                self.weights[i] -= lr * grads_w[i] / (np.sqrt(self.s_w[i]) + eps)
                self.biases[i] -= lr * grads_b[i] / (np.sqrt(self.s_b[i]) + eps)

            elif self.optimizer == "adam" or self.optimizer == "nadam":
                self.v_w[i] = beta1 * self.v_w[i] + (1 - beta1) * grads_w[i]
                self.v_b[i] = beta1 * self.v_b[i] + (1 - beta1) * grads_b[i]
                self.s_w[i] = beta2 * self.s_w[i] + (1 - beta2) * (grads_w[i] ** 2)
                self.s_b[i] = beta2 * self.s_b[i] + (1 - beta2) * (grads_b[i] ** 2)

                v_w_corr = self.v_w[i] / (1 - beta1 ** self.t)
                v_b_corr = self.v_b[i] / (1 - beta1 ** self.t)
                s_w_corr = self.s_w[i] / (1 - beta2 ** self.t)
                s_b_corr = self.s_b[i] / (1 - beta2 ** self.t)

                self.weights[i] -= lr * v_w_corr / (np.sqrt(s_w_corr) + eps)
                self.biases[i] -= lr * v_b_corr / (np.sqrt(s_b_corr) + eps)

        self.t += 1  # Update time step

# Loss Functions
def compute_loss(y_true, y_pred, loss_type="cross_entropy"):
    if loss_type == "cross_entropy":
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]
    elif loss_type == "mean_squared_error":
        return np.mean((y_true - y_pred) ** 2)

# Training Function
def train_nn(args, X_train, y_train, X_val, y_val):
    input_size = X_train.shape[1]
    output_size = 10
    nn = NeuralNetwork(input_size, args.num_layers, args.hidden_size, output_size, args.optimizer, args)

    num_samples = X_train.shape[0]

    for epoch in range(args.epochs):
        total_loss = 0
        for i in range(0, num_samples, args.batch_size):
            X_batch = X_train[i:i + args.batch_size]
            y_batch = y_train[i:i + args.batch_size]

            activations = nn.forward(X_batch)
            grads_w, grads_b = nn.backward(activations, y_batch)
            nn.update_weights(grads_w, grads_b)

            batch_loss = compute_loss(y_batch, activations[-1], loss_type=args.loss)
            total_loss += batch_loss

        avg_loss = total_loss / (num_samples / args.batch_size)

        # Validation Loss Calculation
        val_activations = nn.forward(X_val)
        val_loss = compute_loss(y_val, val_activations[-1], loss_type=args.loss)

        wandb.log({"Epoch": epoch + 1, "Loss": avg_loss, "val_loss": val_loss})

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")


sweep_config = {
    "method": "random",  # Random search
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "Xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
    }
}
def sweep_train():
    with wandb.init() as run:
        config = run.config

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = x_train.reshape(x_train.shape[0], -1) / 255.0, x_test.reshape(x_test.shape[0], -1) / 255.0
        y_train, y_test = one_hot_numpy(y_train, num_classes=10), one_hot_numpy(y_test, num_classes=10)

        # Create validation split (10%)
        val_size = int(0.1 * len(x_train))
        X_val, y_val = x_train[:val_size], y_train[:val_size]
        X_train, y_train = x_train[val_size:], y_train[val_size:]

        args = argparse.Namespace(**config)
        train_nn(args, X_train, y_train, X_val, y_val)

# Class labels for Fashion-MNIST
fashion_mnist_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Function to plot one sample image per class
def plot_fashion_mnist_classes():
    (x_train, y_train), _ = fashion_mnist.load_data()

    plt.figure(figsize=(10, 5))
    
    for label in range(10):
        idx = np.where(y_train == label)[0][0]  # Get the first occurrence of each class
        plt.subplot(2, 5, label + 1)
        plt.imshow(x_train[idx], cmap="gray")
        plt.title(fashion_mnist_labels[label])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Main Function
def main():
    args = get_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name="fashion_mnist_training")

    plot_fashion_mnist_classes()  # Plot Q1 answer before training

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train.reshape(x_train.shape[0], -1) / 255.0, x_test.reshape(x_test.shape[0], -1) / 255.0
    y_train, y_test = one_hot_numpy(y_train, num_classes=10), one_hot_numpy(y_test, num_classes=10)

    # Create validation split (10%)
    val_size = int(0.1 * len(x_train))
    X_val, y_val = x_train[:val_size], y_train[:val_size]
    X_train, y_train = x_train[val_size:], y_train[val_size:]

    train_nn(args, X_train, y_train, X_val, y_val)
    wandb.finish()

# Initialize Sweep
if __name__ == "__main__":
    args = get_args()

    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=sweep_train, count=250)
    else:
        main()

