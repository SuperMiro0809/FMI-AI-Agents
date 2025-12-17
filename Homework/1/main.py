import os
import cv2
import numpy as np
import tensorflow as tf


LATIN_MODEL_PATH = "latin_letters_cnn.h5"
CYRILLIC_MODEL_PATH = "cyrillic_letters_cnn.keras"
CANVAS_SIZE = 640
IMG_SIZE = 28
LINE_THICKNESS = 12

CYRILLIC_LETTERS = [
    "А", "Б", "В", "Г", "Д", "Е", "Ж", "З",
    "И", "Й", "К", "Л", "М", "Н", "О", "П",
    "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч",
    "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я",
    "а", "б", "в", "г", "д", "е", "ж", "з",
    "и", "й"
]


drawing = False
last_x, last_y = None, None
canvas_gray = None
prediction_text = ""


def mouse_draw(event, x, y, flags, param):
    global drawing, last_x, last_y, canvas_gray

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas_gray, (last_x, last_y), (x, y),
                 255, LINE_THICKNESS, cv2.LINE_AA)
        last_x, last_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(canvas_gray, (last_x, last_y), (x, y),
                 255, LINE_THICKNESS, cv2.LINE_AA)


def preprocess_canvas_for_model(gray_canvas):
    resized = cv2.resize(gray_canvas, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    resized = resized.astype("float32") / 255.0
    resized = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return resized


def load_model_and_mapping(lang: str):
    if lang == "en":
        if not os.path.exists(LATIN_MODEL_PATH):
            raise FileNotFoundError(f"Latin model not found: {LATIN_MODEL_PATH}")
        model = tf.keras.models.load_model(LATIN_MODEL_PATH)

        def idx_to_label(i: int) -> str:
            if 0 <= i < 26:
                return chr(ord("a") + i)
            return f"class {i}"

        window_title = "Draw LATIN letter (P=predict, C=clear, ESC=exit)"

    else:
        if not os.path.exists(CYRILLIC_MODEL_PATH):
            raise FileNotFoundError(f"Cyrillic model not found: {CYRILLIC_MODEL_PATH}")
        model = tf.keras.models.load_model(CYRILLIC_MODEL_PATH)

        def idx_to_label(i: int) -> str:
            if 0 <= i < len(CYRILLIC_LETTERS):
                print(CYRILLIC_LETTERS[i + 1])
                # return CYRILLIC_LETTERS[i]
            return f"class {i}"

        window_title = "Draw CYRILLIC letter (P=predict, C=clear, ESC=exit)"

    return model, idx_to_label, window_title


def main():
    global canvas_gray, prediction_text

    lang = input("Choose language ('en' for Latin, 'bg' for Cyrillic): ").strip().lower()
    if lang not in ("en", "bg"):
        print("Invalid choice, defaulting to 'en' (Latin).")
        lang = "en"

    model, idx_to_label, window_title = load_model_and_mapping(lang)

    canvas_gray = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    prediction_text = ""

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, mouse_draw)

    print("Controls:")
    print("  - Left mouse button: draw")
    print("  - P: predict")
    print("  - C: clear")
    print("  - ESC: exit")

    while True:
        display = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

        if prediction_text:
            cv2.rectangle(display, (0, 0), (CANVAS_SIZE, 40), (0, 0, 0), -1)
            cv2.putText(display, prediction_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        cv2.imshow(window_title, display)
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (ord("c"), ord("C")):
            canvas_gray[:] = 0
            prediction_text = ""
        elif key in (ord("p"), ord("P")):
            inp = preprocess_canvas_for_model(canvas_gray)
            probs = model.predict(inp, verbose=0)[0]
            class_idx = int(np.argmax(probs))
            conf = float(probs[class_idx] * 100.0)
            label = idx_to_label(class_idx)
            prediction_text = f"{label} ({conf:.1f}%)"

            top3_idx = probs.argsort()[-3:][::-1]
            print("Top-3 classes:", [(int(i), float(probs[i])) for i in top3_idx])

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
