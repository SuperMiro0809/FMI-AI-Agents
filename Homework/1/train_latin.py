import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers

# image size is 28x28, grayscale (1 channel)
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHANNELS = 1
NUM_CLASSES = 26  # letters a–z


def load_emnist_dataset():
    """
    Load EMNIST Letters and prepare it for training.
    """

    (train_data, test_data) = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        batch_size=-1        # <– this is the trick
    )

    x_train, y_train = train_data
    x_test, y_test = test_data

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    # normalize images to [0,1]
    x_train /= 255.0
    x_test /= 255.0

    # labels: 1..26 -> 0..25
    y_train = y_train - 1
    y_test = y_test - 1

    return x_train, y_train, x_test, y_test


def build_cnn_model():
    """
    Simple CNN for 28x28 grayscale letter images.
    """

    model = keras.Sequential([
        layers.Conv2D(
            32, (3, 3), activation="relu",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    epochs = 12
    batch_size = 128

    print("Loading EMNIST dataset...")
    x_train, y_train, x_test, y_test = load_emnist_dataset()

    print("Building CNN model...")
    model = build_cnn_model()
    model.summary()

    print("Starting training...")
    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test)
    )

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save the trained model
    model.save("latin_letters_cnn.h5")
    print('Model saved as "latin_letters_cnn.h5"')


if __name__ == "__main__":
    main()
