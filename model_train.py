import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
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
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# 編譯 & 訓練 & 自動停止過擬合階段
opt = Adam(learning_rate=0.0003)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, validation_split=0.1, callbacks=[early_stop])

# 評估測試集
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 儲存模型為 .h5
model.save('model/model.h5')

# 載入模型
model = load_model(model_path)

# 儲存架構為 json
layers = []
weights = {}

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
        weights[w_name] = W
        weights[b_name] = b

    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        gamma, beta, mean, var = layer.get_weights()
        gamma_name = f"gamma_{i}"
        beta_name = f"beta_{i}"
        mean_name = f"mean_{i}"
        var_name = f"var_{i}"
        layers.append({
            "name": f"bn_{i}",
            "type": "BatchNormalization",
            "config": {"epsilon": cfg.get("epsilon", 1e-3)},
            "weights": [gamma_name, beta_name, mean_name, var_name]
        })
        weights[gamma_name] = gamma
        weights[beta_name] = beta
        weights[mean_name] = mean
        weights[var_name] = var

    elif isinstance(layer, tf.keras.layers.Dropout):
        rate = cfg.get("rate", 0.5)
        layers.append({
            "name": f"dropout_{i}",
            "type": "Dropout",
            "config": {"rate": rate},
            "weights": []  # Dropout has no weights
        })

    else:
        print(f"Warning: Unsupported layer type: {layer}")

# 寫入 JSON
with open(output_json, 'w') as f:
    json.dump(layers, f, indent=2)

# 寫入 NPZ
np.savez(output_npz, **weights)
print("模型成功轉換為 JSON + NPZ")