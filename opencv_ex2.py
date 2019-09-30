import cv2
import imutils
from skimage.measure import compare_ssim

ratio2 = 3
kernel_size = 3
lowThreshold = 30

ONE_PATH = r"one.png"
ZERO_PATH = r"zero.png"


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def get_digits_from_image(image_path, matrix_size):

    oneObj = cv2.imread(ONE_PATH, cv2.IMREAD_GRAYSCALE)
    zeroObj = cv2.imread(ZERO_PATH, cv2.IMREAD_GRAYSCALE)
    imgObj = cv2.imread(image_path)#, cv2.IMREAD_GRAYSCALE)
    imgObj = cv2.cvtColor(imgObj, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", imgObj)
    thresh = cv2.threshold(imgObj, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow("thresh", thresh)

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_lines2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts2 = cv2.findContours(detected_lines2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(imgObj, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))

    cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
    for c in cnts2:
        cv2.drawContours(imgObj, [c], -1, (255, 255, 255), 2)

    result2 = 255 - cv2.morphologyEx(255 - imgObj, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    # cv2.imshow('result2', result2)

    thresh = cv2.threshold(result2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow('thresh', thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # print(x, y, w, h)
        # if the contour is sufficiently large, it must be a digit
        if w >= 5 and (10 <= h <= 20):
            digitCnts.append(c)

    digitCnts.sort(key=lambda x: get_contour_precedence(x, imgObj.shape[1]))
    final_matrix = []
    temp_list = []
    counter = 0
    for i, c in enumerate(digitCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        roi = imgObj[y-2: y+h+2, x-2: x+2+w]
        roi = cv2.resize(roi, (20, 20))
        # cv2.imwrite("Output/image_{}.png".format(i), roi)

        (score1, diff1) = compare_ssim(roi, oneObj, full=True)
        # diff1 = (diff1 * 255).astype("uint8")
        # print("SSIM1: {}".format(score1))

        (score0, diff0) = compare_ssim(roi, zeroObj, full=True)
        # diff0 = (diff0 * 255).astype("uint8")
        # print("SSIM0: {}".format(score0))
        if score0 > score1:
            number = 0
        else:
            number = 1

        if counter < matrix_size[1]:
            temp_list.append(number)
            counter += 1
        else:
            final_matrix.append(temp_list)
            temp_list = [number]
            counter = 1

        # print("{}: x:{}\ty:{}\tw:{}\th:{}".format(i, x, y, w, h))
        # cv2.rectangle(imgObj, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 255, 0), 1)
        # cv2.putText(imgObj, str(i + 1), (x - 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        # print("{}: {}".format(counter, number))
        # cv2.imshow('final', imgObj)
        # cv2.waitKey()
    final_matrix.append(temp_list)

    print(*final_matrix, sep="\n")
    cv2.waitKey()


if __name__ == '__main__':

    imagePath = r"C:\Users\UHSE1215\Desktop\Umesh_data\Learn\PyImageSearch\OpenCV\binary_matrix.png"
    matrixSize = (7, 7)

    get_digits_from_image(imagePath, matrixSize)
    