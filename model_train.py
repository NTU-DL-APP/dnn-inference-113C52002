import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
import json
import numpy as np
import os

model_path = './model/model.h5'
output_json = './model/fashion_mnist.json'
output_npz = './model/fashion_mnist.npz'

# 確保資料夾存在
os.makedirs('./model', exist_ok=True)

# 載入資料
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 編譯 & 訓練
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.1)

# 評估測試集
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 儲存模型為 .h5
model.save('model/model.h5')

# 載入模型
model = load_model(model_path)

# 儲存架構為 json
layers = []
for i, layer in enumerate(model.layers):
    cfg = layer.get_config()
    if isinstance(layer, tf.keras.layers.Flatten):
        layers.append({
            "name": f"flatten_{i}",
            "type": "Flatten",
            "config": {},
            "weights": []
        })
    elif isinstance(layer, tf.keras.layers.Dense):
        act = cfg['activation']
        W, b = layer.get_weights()
        w_name = f"W_{i}"
        b_name = f"b_{i}"
        layers.append({
            "name": f"dense_{i}",
            "type": "Dense",
            "config": {"activation": act},
            "weights": [w_name, b_name]
        })
    else:
        print("Warning: Unsupported layer type")

with open(output_json, 'w') as f:
    json.dump(layers, f, indent=2)

# 儲存權重為 npz
weights = {}
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'get_weights'):
        w = layer.get_weights()
        if len(w) == 2:
            weights[f"W_{i}"] = w[0]
            weights[f"b_{i}"] = w[1]
np.savez(output_npz, **weights)

print("模型成功轉換為 JSON + NPZ")