import numpy as np
import matplotlib.pyplot as plt

# Load dataset
x_train = np.load('X_train.npy')
x_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

# Neural Network class
class NeuralNetwork:
    def __init__(self, layer_dims, activation='relu', learning_rate=0.01, epochs=100):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Set the activation function and its derivative
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Unsupported activation function. Choose 'relu', 'sigmoid', or 'tanh'.")
        self.parameters = self.initialize_parameters()
        
    def initialize_parameters(self):
        parameters = {}
        np.random.seed(42)
        for l in range(1, len(self.layer_dims)):
            parameters[f"W{l}"] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            parameters[f"b{l}"] = np.zeros((self.layer_dims[l], 1))
        return parameters

    def forward_propagation(self, X):
        cache = {"A0": X.T}
        A = X.T
        for l in range(1, len(self.layer_dims)):
            Z = np.dot(self.parameters[f"W{l}"], A) + self.parameters[f"b{l}"]
            A = self.activation(Z) if l < len(self.layer_dims) - 1 else sigmoid(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A
        return A, cache

    def backward_propagation(self, Y, cache):
        gradients = {}
        m = Y.shape[0]
        A_final = cache[f"A{len(self.layer_dims) - 1}"]
        dA = -(np.divide(Y.T, A_final) - np.divide(1 - Y.T, 1 - A_final))
        
        for l in reversed(range(1, len(self.layer_dims))):
            dZ = dA * (sigmoid_derivative(cache[f"Z{l}"]) if l == len(self.layer_dims) - 1 else self.activation_derivative(cache[f"Z{l}"]))
            dW = (1 / m) * np.dot(dZ, cache[f"A{l - 1}"].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.parameters[f"W{l}"].T, dZ)
            
            gradients[f"dW{l}"] = dW
            gradients[f"db{l}"] = db
        return gradients

    def update_parameters(self, gradients):
        for l in range(1, len(self.layer_dims)):
            self.parameters[f"W{l}"] -= self.learning_rate * gradients[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * gradients[f"db{l}"]

    def fit(self, X, Y):
        for epoch in range(self.epochs):
            A_final, cache = self.forward_propagation(X)
            gradients = self.backward_propagation(Y, cache)
            self.update_parameters(gradients)

    def predict(self, X):
        A_final, _ = self.forward_propagation(X)
        return (A_final.T > 0.5).astype(int)

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1


# Reshape labels
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Define the architectures and activation functions
architectures = [
    {"layer_dims": [8, 16, 16, 16, 1], "activation": "relu"},
    {"layer_dims": [8, 16, 16, 1], "activation": "relu"},
    {"layer_dims": [8, 16, 1], "activation": "relu"},
    {"layer_dims": [8, 8, 1], "activation": "relu"},
    {"layer_dims": [8, 4, 1], "activation": "relu"},
    {"layer_dims": [8, 4, 1], "activation": "tanh"}
]

# Store metrics for each architecture
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Train and evaluate each architecture
for i, arch in enumerate(architectures, start=1):
    nn = NeuralNetwork(
        layer_dims=arch["layer_dims"],
        activation=arch["activation"],
        learning_rate=0.01,
        epochs=500
    )
    nn.fit(x_train, y_train)
    y_pred_test = nn.predict(x_test)
    
    accuracy, precision, recall, f1 = calculate_metrics(y_test.flatten(), y_pred_test.flatten())
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    
    # Print metrics
    print(f"NN{i} - Activation: {arch['activation']}, Layer Dims: {arch['layer_dims']}")
    print(f"    Accuracy: {accuracy:.2f}")
    print(f"    Precision: {precision:.2f}")
    print(f"    Recall: {recall:.2f}")
    print(f"    F1 Score: {f1:.2f}")

# Plot the results
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Upper left: Accuracy
axs[0, 0].plot(range(1, 7), accuracy_list, marker='o', label="Accuracy", color='blue')
axs[0, 0].set_xticks(range(1, 7))
axs[0, 0].set_xticklabels(["NN1", "NN2", "NN3", "NN4", "NN5", "NN6"])
axs[0, 0].set_ylabel("Accuracy")
axs[0, 0].set_title("Accuracy vs Architectures")
axs[0, 0].legend()

# Upper right: Precision
axs[0, 1].plot(range(1, 7), precision_list, marker='o', label="Precision", color='green')
axs[0, 1].set_xticks(range(1, 7))
axs[0, 1].set_xticklabels(["NN1", "NN2", "NN3", "NN4", "NN5", "NN6"])
axs[0, 1].set_ylabel("Precision")
axs[0, 1].set_title("Precision vs Architectures")
axs[0, 1].legend()

# Bottom left: Recall
axs[1, 0].plot(range(1, 7), recall_list, marker='o', label="Recall", color='red')
axs[1, 0].set_xticks(range(1, 7))
axs[1, 0].set_xticklabels(["NN1", "NN2", "NN3", "NN4", "NN5", "NN6"])
axs[1, 0].set_ylabel("Recall")
axs[1, 0].set_title("Recall vs Architectures")
axs[1, 0].legend()

# Bottom right: F1 Score
axs[1, 1].plot(range(1, 7), f1_list, marker='o', label="F1 Score", color='purple')
axs[1, 1].set_xticks(range(1, 7))
axs[1, 1].set_xticklabels(["NN1", "NN2", "NN3", "NN4", "NN5", "NN6"])
axs[1, 1].set_ylabel("F1 Score")
axs[1, 1].set_title("F1 Score vs Architectures")
axs[1, 1].legend()

# Save the plot
plt.tight_layout()
plt.savefig("nn_architecture_metrics_dataset_2.png", dpi=300)  # Save as a PNG file
plt.show()

