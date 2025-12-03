import cv2
from pyzbar.pyzbar import decode

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2056)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1329)

    if(cap.isOpened()):
        while(cv2.waitKey(1) != ord('q')):
            ret, frame = cap.read()

            qr_codes = decode(frame)
            for qr in qr_codes:
                (x, y, w, h) = qr.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                barcodeData = qr.data.decode("utf-8")
                barcodeType = qr.type

                text = "{} ({})".format(barcodeData, barcodeType)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
 
            cv2.imshow('webCam', frame)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print('webCam is failed to use')