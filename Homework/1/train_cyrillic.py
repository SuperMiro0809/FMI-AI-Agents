# import os
# import cv2
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, models

# # ------------ CONFIG (adjust if needed) ------------
# BASE_DIR = "cmnist"
# IMAGES_SUBDIR = "allgr"        # folder with PNGs (allbi / allgr / allrgb)
# TRAIN_CSV = "train.csv"
# VAL_CSV = "validation.csv"
# TEST_CSV = "test.csv"
# LABEL_COLUMN = "balanced42"    # which label column to use
# IMG_SIZE = 28
# NUM_CLASSES = 42               # 42 Cyrillic classes
# # ---------------------------------------------------

# def load_split(csv_name, max_samples=None):
#     """
#     Loads images & labels from a CSV file and corresponding PNG images.
#     """
#     csv_path = os.path.join(BASE_DIR, csv_name)
#     images_path = os.path.join(BASE_DIR, IMAGES_SUBDIR)

#     df = pd.read_csv(csv_path)
#     if max_samples:
#         df = df.head(max_samples)

#     filenames = df["filename"].values
#     labels = df[LABEL_COLUMN].astype("int64").values

#     images = []
#     for name in filenames:
#         img = cv2.imread(os.path.join(images_path, name), cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise RuntimeError(f"Cannot load image: {name}")

#         if img.shape != (IMG_SIZE, IMG_SIZE):
#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

#         images.append(img)

#     x = np.array(images, dtype="float32") / 255.0
#     x = np.expand_dims(x, -1)     # (N, 28, 28, 1)
#     y = labels

#     return x, y


# def build_model():
#     """
#     Simple CNN for 28x28 grayscale images.
#     """
#     model = models.Sequential([
#         layers.Conv2D(32, 3, activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
#         layers.MaxPooling2D(2),
#         layers.Conv2D(64, 3, activation="relu"),
#         layers.MaxPooling2D(2),
#         layers.Flatten(),
#         layers.Dense(128, activation="relu"),
#         layers.Dropout(0.4),
#         layers.Dense(NUM_CLASSES, activation="softmax")
#     ])

#     model.compile(
#         optimizer="adam",
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )
#     return model


# def main():
#     # Small subsets to avoid freezing issues
#     x_train, y_train = load_split(TRAIN_CSV, max_samples=2000)
#     x_val, y_val = load_split(VAL_CSV, max_samples=500)
#     x_test, y_test = load_split(TEST_CSV, max_samples=500)

#     model = build_model()
#     model.summary()

#     model.fit(
#         x_train, y_train,
#         validation_data=(x_val, y_val),
#         batch_size=64,
#         epochs=3,
#         verbose=1
#     )

#     loss, acc = model.evaluate(x_test, y_test, verbose=1)
#     print(f"Test accuracy: {acc:.4f}")

#     model.save("cyrillic_letters_cnn.h5")
#     print("Saved model as 'cyrillic_letters_cnn.h5'")


# if __name__ == "__main__":
#     main()

import os
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

DATA_PATH = r"cmnist"
IMG_DIR = os.path.join(DATA_PATH, "allgr")
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV  = os.path.join(DATA_PATH, "test.csv")

IMG_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 10
NUM_CLASSES = 81

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    filenames = [
        os.path.join(IMG_DIR, fname)
        for fname in df["filename"].values
    ]
    labels = df["letters81"].values

    return filenames, labels


def preprocess_image(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Invert (white background → black)
    img = 1.0 - img

    return img, label


def make_dataset(filenames, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.shuffle(1000)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def create_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),

        layers.Dense(256, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():
    print("\n=== CYRILLIC CNN TRAINING ===\n")

    # Load data
    train_files, train_labels = load_dataset(TRAIN_CSV)
    test_files, test_labels = load_dataset(TEST_CSV)

    train_ds = make_dataset(train_files, train_labels, training=True)
    test_ds  = make_dataset(test_files, test_labels, training=False)

    # Model
    model = create_model()
    model.summary()

    # Train
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds
    )

    # Evaluate
    loss, acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {acc*100:.2f}%")

    # Save
    model.save("cnn_cyrillic.keras")
    print("Model saved: cnn_cyrillic.keras")


if __name__ == "__main__":
    main()
