import numpy as np
import cv2

def convolution(image, kernel):

    # Flip kernel horizontally and vertically
    kernel = np.fliplr(np.flipud(kernel))

    each_additional_padding = len(kernel) // 2
    total_additional_padding = each_additional_padding * 2

    # create new padded image and set pixels to zero
    image_with_padding = np.zeros(
        (image.shape[0] + total_additional_padding, image.shape[1] + total_additional_padding))
    final_image = np.zeros_like(image)

    # fill new image with each channel
    image_with_padding[each_additional_padding:-each_additional_padding, each_additional_padding:-each_additional_padding] = image
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            # perform matrix operations on image and kernel
            final_image[j, i] = (kernel * image_with_padding[j:j + len(kernel), i:i + len(kernel)]).sum()

    return final_image


def generate_gaussian_smoothing(img, size, sigma):
    x = np.arange(0, size, dtype=np.uint16)
    y = np.arange(0, size, dtype=np.uint16)[:, np.newaxis]

    x = x - (size // 2)
    y = y - (size // 2)

    # Calculate Gaussian kernel
    e = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-e)

    # normalize kernel
    kernel /= kernel.sum()

    return convolution(img, kernel)


def image_gradient(img):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = sobel_x.T

    sobel_x_img = convolution(img, sobel_x)
    # cv2.imshow('sobel_x_img', sobel_x_img)

    sobel_y_img = convolution(img, sobel_y)
    # cv2.imshow('sobel_y_img', sobel_y_img)

    magnitude_img = np.hypot(sobel_x_img, sobel_y_img)
    magnitude_img = magnitude_img / magnitude_img.max() * 255
    final_mag_img = magnitude_img.astype(np.uint8)
    theta = np.degrees(np.arctan2(sobel_y_img, sobel_x_img))
    return final_mag_img, theta


def nonmaxima_suppress(mag, theta):
    nms = np.zeros_like(mag)
    theta[theta < 0] += 180
    f = 0
    b = 0
    for i in range(0, mag.shape[0] - 1):
        for j in range(0, mag.shape[1] - 1):
            if 0 <= theta[i][j] <= 45:
                mid = 45 / 2
                if theta[i][j] < mid:
                    # 0 degree line
                    f = mag[i, j + 1]
                    b = mag[i, j - 1]
                else:
                    # 45 degree line
                    f = mag[i - 1, j + 1]
                    b = mag[i + 1, j - 1]
            elif 45 < theta[i][j] <= 90:
                mid = (45 + 90) / 2
                if theta[i][j] < mid:
                    # 45 degree line
                    f = mag[i - 1, j + 1]
                    b = mag[i + 1, j - 1]
                else:
                    # 90 degree line
                    f = mag[i - 1, j]
                    b = mag[i + 1, j]
            elif 90 < theta[i][j] <= 135:
                mid = (90 + 135) / 2
                if theta[i][j] < mid:
                    # 90 degree line
                    f = mag[i - 1, j]
                    b = mag[i + 1, j]
                else:
                    # 135 degree line
                    f = mag[i - 1, j - 1]
                    b = mag[i + 1, j + 1]
            elif 135 < theta[i][j] <= 180:
                mid = (135 + 180) / 2
                if theta[i][j] < mid:
                    # 135 degree line
                    f = mag[i - 1, j - 1]
                    b = mag[i + 1, j + 1]
                else:
                    # 180 degree line
                    f = mag[i, j - 1]
                    b = mag[i, j + 1]

            if (mag[i, j] >= f) and (mag[i, j] >= b):
                nms[i, j] = mag[i, j]
            else:
                nms[i, j] = 0

    return nms


def edge_linking(mag, high, low):
    high *= 255
    low *= 255
    final = np.zeros_like(mag)
    for i in range(1, mag.shape[0] - 1):
        for j in range(1, mag.shape[1] - 1):
            if mag[i, j] < low:
                final[i, j] = 0
            elif mag[i, j] >= high:
                final[i, j] = mag[i, j]
            else:

                # Get 8 neighboring pixels
                neighbor_pixels = mag[i - 1:i + 2, j - 1:j + 2]

                # check if neighbors are strong
                if neighbor_pixels.max() >= high:
                    final[i, j] = mag[i, j]
                else:
                    final[i, j] = 0

    return final


def main():
    # import image
    img = cv2.imread('lena_gray.png', -1) / 255
    # img = cv2.imread('test.png', -1) / 255

    s = generate_gaussian_smoothing(img, 3, 1)
    # cv2.imshow('gaussian', s)
    mag, theta = image_gradient(s)
    # cv2.imshow('gradient_mag', mag)
    nms = nonmaxima_suppress(mag, theta)
    # cv2.imshow('NMS', nms)
    e = edge_linking(nms, 0.3, 0.15)
    cv2.imshow('edge_linking', e)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
