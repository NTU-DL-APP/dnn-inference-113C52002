import json
import numpy as np
from nn_predict import nn_inference
import gzip
import numpy as np

def load_fashion_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        buffer = f.read(num_images * rows * cols)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows, cols) / 255.0  # 正規化
        return data

def load_fashion_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num_labels = int.from_bytes(f.read(4), 'big')
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

# 讀模型架構和權重
with open('./model/fashion_mnist.json', 'r') as f:
    model_arch = json.load(f)

weights_npz = np.load('./model/fashion_mnist.npz')
weights = {k: weights_npz[k] for k in weights_npz.files}

# 讀資料
images = load_fashion_mnist_images('data/fashion/t10k-images-idx3-ubyte.gz')
labels = load_fashion_mnist_labels('data/fashion/t10k-labels-idx1-ubyte.gz')

# 推論
preds = nn_inference(model_arch, weights, images)
pred_labels = np.argmax(preds, axis=1)

# 準確率
acc = np.mean(pred_labels == labels)
print(f'Accuracy: {acc:.4f}')