"""
Baseline CNN Training - EXACT CODE YOU RAN
This reproduces your minimal CNN with 90% accuracy
"""

import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ----------------------
# 1️⃣ Load Data - EXACTLY AS YOU DID
# ----------------------
data_path = "data/train"
classes = sorted(os.listdir(data_path))
num_classes = len(classes)

X = []
y = []

for idx, cls in enumerate(classes):
    cls_path = os.path.join(data_path, cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        img = Image.open(img_path).convert("L")  # convert to grayscale
        img_array = np.array(img) / 255.0  # normalize
        X.append(img_array)
        y.append(idx)

X = np.array(X).reshape(-1, 64, 64, 1)  # add channel dimension
y = to_categorical(y, num_classes=num_classes)

# ----------------------
# 2️⃣ Split into train/validation - EXACTLY AS YOU DID
# ----------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Number of classes: {num_classes}")

# ----------------------
# 3️⃣ Build Minimal CNN - EXACTLY AS YOU DID
# ----------------------
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------
# 4️⃣ Train the Model - EXACTLY AS YOU DID
# ----------------------
print("\nTraining minimal CNN (5 epochs)...")
history = model.fit(X_train, y_train, 
                    epochs=5, 
                    batch_size=32, 
                    validation_data=(X_val, y_val),
                    verbose=1)

# ----------------------
# 5️⃣ Save the Model
# ----------------------
model.save("models/baseline/minimal_cnn.h5")
print("\n✅ Model saved as models/baseline/minimal_cnn.h5")

# Save training history
import pickle
with open("training/history/baseline_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("✅ Training history saved")

# Print results
final_val_acc = history.history['val_accuracy'][-1]
print(f"\n📊 Final validation accuracy: {final_val_acc:.2%}")

# Plot training history
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig("results/baseline/training_history.png", dpi=100, bbox_inches='tight')
print("✅ Training plots saved to results/baseline/training_history.png")
plt.show()