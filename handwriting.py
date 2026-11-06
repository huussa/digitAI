from pathlib import Path
import tensorflow as tf


# ==========================================
#                  SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"

EPOCHS = 20

# ==========================================
#           LOAD MNIST DATASET
# ==========================================

print("Loading MNIST dataset...")

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Convert labels to One-Hot Encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Add channel dimension
# Before: (60000, 28, 28)
# After:  (60000, 28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


# ==========================================
#               CREATE MODEL
# ==========================================
def create_model():

    model = tf.keras.Sequential([

        tf.keras.Input(shape=(28, 28, 1)),

        # Data Augmentation
        tf.keras.layers.RandomTranslation(
            height_factor=0.15,
            width_factor=0.15
        ),

        # =========================
        # Block 1
        # =========================
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),
        tf.keras.layers.Dropout(0.25),

        # =========================
        # Block 2
        # =========================
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),
        tf.keras.layers.Dropout(0.25),

        # =========================
        # Block 3
        # =========================
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        # =========================
        # Classification
        # =========================
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )

    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==========================================
#   LOAD EXISTING MODEL OR CREATE NEW ONE
# ==========================================
if MODEL_PATH.exists():

    print("\n" + "=" * 50)
    print("Existing model found!")
    print("Loading model and continuing training...")
    print("=" * 50 + "\n")

    model = tf.keras.models.load_model(MODEL_PATH)

else:

    print("\n" + "=" * 50)
    print("No existing model found.")
    print("Creating a new model...")
    print("=" * 50 + "\n")

    model = create_model()

# ==========================================
#           SHOW MODEL STRUCTURE
# ==========================================
model.summary()

# ==========================================
#                TRAIN MODEL
# ==========================================
print("\nStarting training...")
print(f"Training for {EPOCHS} additional epochs.\n")

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    validation_data=(x_test, y_test)
)

# ==========================================
#               EVALUATE MODEL
# ==========================================
print("\nEvaluating model on test data...\n")

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=2
)

print("\n" + "=" * 50)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss * 100:.4f}%")
print("=" * 50)

# ==========================================
#                SAVE MODEL
# ==========================================

model.save(MODEL_PATH)

print(f"\nModel saved successfully to:")
print(MODEL_PATH)