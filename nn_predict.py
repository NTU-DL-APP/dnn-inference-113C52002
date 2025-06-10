import numpy as np
import json

# === Activation functions ===
def relu(x):
    # TODO: Implement the Rectified Linear Unit
    return np.maximum(0, x)

def softmax(x):
    # TODO: Implement the SoftMax function
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    out = e / np.sum(e, axis=-1, keepdims=True)
    return out if x.ndim == 2 else out[np.newaxis, :]


# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Batch Normalization (inference mode) ===
def BatchNormalization(x, gamma, beta, moving_mean, moving_variance, epsilon=1e-3):
    x_norm = (x - moving_mean) / np.sqrt(moving_variance + epsilon)
    return gamma * x_norm + beta

# === Dropout (inference: skip) ===
def Dropout(x, rate):
    # During inference, dropout is not applied. Just return x.
    return x



# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer.get('weights', [])

        if ltype == "Flatten":
            x = flatten(x)
        
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)

            activation = cfg.get("activation")
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)

        elif ltype == "BatchNormalization":
            gamma = weights[wnames[0]]  # gamma
            beta = weights[wnames[1]]   # beta
            moving_mean = weights[wnames[2]]
            moving_var = weights[wnames[3]]
            epsilon = cfg.get("epsilon", 1e-3)
            x = BatchNormalization(x, gamma, beta, moving_mean, moving_var, epsilon)

        elif ltype == "Dropout":
            rate = cfg.get("rate", 0.5)
            x = Dropout(x, rate)  # No effect during inference

    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
    
