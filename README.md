WanDB Report: https://wandb.ai/da24m029-indian-institute-of-technology-madras/da24m029-da6401-assignment1/reports/DA24M029-DA6401-Assignment-1--VmlldzoxMTgzODEwMQ?accessToken=sdmgpmcqs6h6v8946h3pt18vzd325lv50o2skdrr37m9mdg23ynxh0v86bd3teqg 

GitHub Link: https://github.com/zsbhagat12/DA24M029-DA6401-Assignment1

train.py and README.md are the only files. 

Make sure the required libraries are installed.  

Just Run for example `python train.py -wp da24m029-da6401-assignment1 -we da24m029-indian-institute-of-technology-madras --sweep` for sweep functionality.  

Run for example `python train.py -wp da24m029-da6401-assignment1 -we da24m029-indian-institute-of-technology-madras ` for single model train and its plots (confusion, class labels) which will run for default values or custom values if they are passed. 

Can add parameters with their arg flags and values as and when required. 

Add `-l mean_squared_error` for digressing from default value and answering q8.

Similarly, `-d mnist` to be added for q10. 

# Code Organization for Neural Network with WandB Logging

## 1. Argument Parsing
- **`get_args()`**: Parses command-line arguments (e.g., dataset, optimizer, hyperparameters).

## 2. Data Handling
- **`load_dataset(dataset_name)`**: Loads and preprocesses **MNIST/Fashion-MNIST** dataset.
- **`one_hot_numpy(y, num_classes)`**: Converts labels to **one-hot encoding**.

## 3. Activation Functions & Derivatives
- **Activation Functions**:
  - `sigmoid(x)`, `identity(x)`, `tanh(x)`, `relu(x)`

- **Derivatives** (for backpropagation):
  - `sigmoid_derivative(x)`, `relu_derivative(x)`, etc.

- **Softmax Function**:
  - `softmax(x)`: Implements **Softmax activation**.

## 4. Neural Network Class (`NeuralNetwork`)
- **`__init__()`**: Initializes weights, biases, and optimizer parameters.
- **`forward(X)`**: Implements **forward pass**.
- **`backward(activations, y_true)`**: Computes gradients using **backpropagation**.
- **`update_weights(grads_w, grads_b)`**: Updates weights using selected **optimizer**.

## 5. Loss Functions
- **`compute_loss(y_true, y_pred, loss_type)`**: Computes loss for **cross-entropy / MSE**.

## 6. Model Training
- **`train_nn(args, X_train, y_train, X_val, y_val)`**: Trains the neural network.
- **`compute_confusion_matrix(y_true, y_pred, labels)`**: Generates **confusion matrix**.
- **`plot_confusion_matrix(cm, labels, title)`**: Plots **heatmap** of the confusion matrix.

## 7. Hyperparameter Tuning (WandB Sweeps)
- **`sweep_train()`**: Runs WandB hyperparameter sweeps.
- **`sweep_config`**: Defines search space for sweeps.

## 8. Data Visualization
- **`plot_classes()`**: Displays one sample per **class**.
- **`plot_confusion_matrix()`**: Visualizes the **confusion matrix**.

## 9. Main Execution
- **`main()`**: Handles dataset loading, training, and WandB logging.
- **`if __name__ == "__main__"`**:
  - Runs `sweep_train()` if **sweep mode is enabled**.
  - Otherwise, executes `main()`.
