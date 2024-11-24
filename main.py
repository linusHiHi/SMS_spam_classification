from keras import Sequential
from keras.src.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(128, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
