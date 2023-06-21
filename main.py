from detector import FasterRCNNDetector
import cv2


def main():

    detector = FasterRCNNDetector(model_path='./trained_model/model_frcnn.hdf5')

    img = cv2.imread('test on imgs/input_imgs/00110596.jpg')
    detector.detect_on_image(img)


if __name__ == '__main__':
    main()