import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout

# 假设降维后的嵌入（例如使用 PCA 得到）：
# 每个样本为 50 维特征
reduced_embeddings = np.random.rand(1000, 50)  # 示例数据，1000 个样本，50 维特征
labels = np.random.randint(0, 2, size=(1000,))  # 示例标签（0 或 1）

# 规范化输入形状：CNN 需要 (样本数, 时间步, 特征数)
X = reduced_embeddings[..., np.newaxis]  # 在最后一维增加通道数
y = tf.keras.utils.to_categorical(labels, num_classes=2)  # 独热编码

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义 CNN 模型
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 二分类输出
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
