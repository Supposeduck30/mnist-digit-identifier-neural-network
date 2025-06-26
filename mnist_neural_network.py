import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random

# Load and normalize data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build simple neural network
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x_train_small = x_train[:10000]
y_train_small = y_train[:10000]

print("Training...")
model.fit(x_train_small, y_train_small, epochs=5, validation_split=0.1, batch_size=16, verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Pick one random image of each digit 1–9
images = []
labels = []
for digit in range(1, 10):
    indices = list(np.where(y_test == digit)[0])
    random_idx = random.choice(indices)
    images.append(x_test[random_idx])
    labels.append(y_test[random_idx])

# Predict on those 9 images
images_np = np.array(images)
predictions = model.predict(images_np)

# Plot grid with color-coded confidence
plt.figure(figsize=(9, 9))
confidences = []

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')

    pred = np.argmax(predictions[i])
    conf = np.max(predictions[i]) * 100
    true = labels[i]
    confidences.append(conf)

    if conf >= 90:
        color = 'green'
    elif conf >= 70:
        color = 'orange'
    else:
        color = 'red'

    plt.title(f"Pred: {pred} ({conf:.1f}%)\nActual: {true}", fontsize=10, pad=15, color=color)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.subplots_adjust(top=0.88)
plt.suptitle("Model predictions on digits 1–9 with confidence %", fontsize=16, y=0.98)
plt.show()

# New: Plot average model confidence for true class on all test images (digits 0–9)
pred_test = model.predict(x_test, verbose=0)  # shape: (10000, 10)

# Calculate average confidence the model assigns to the correct label for each digit
avg_conf = [
    np.mean(pred_test[y_test == d, d]) * 100
    for d in range(10)
]

# Plot bar chart of average confidence per digit
lowest_idx = int(np.argmin(avg_conf))

plt.figure(figsize=(10, 4))
bars = plt.bar(range(10), avg_conf, color='skyblue')
bars[lowest_idx].set_color('red')  # Highlight the lowest-confidence digit

# Add labels
for i, c in enumerate(avg_conf):
    plt.text(i, c + 1, f"{c:.1f}%", ha='center', fontsize=9)

plt.xticks(range(10))
plt.ylim(0, 105)
plt.xlabel("Digit")
plt.ylabel("Average confidence (%)")
plt.title("Average Model Confidence per Digit on the Full MNIST Test Set\n(red = lowest)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
