import numpy as np
import cv2


def convolution_bgr(image, kernel):

    each_additional_padding = len(kernel) // 2
    total_additional_padding = each_additional_padding * 2

    # create new padded image and set pixels to zero
    image_padded = np.zeros((image.shape[0] + total_additional_padding, image.shape[1] + total_additional_padding))

    # split image into its channels
    b, g, r = cv2.split(image)
    padded_channels = [np.zeros_like(b), np.zeros_like(g), np.zeros_like(r)]
    channels = [b, g, r]

    # cycle through every channel
    for c in range(3):

        # fill new image with each channel
        image_padded[each_additional_padding:-each_additional_padding, each_additional_padding:-each_additional_padding] = channels[c]
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):

                # perform matrix operations on image and kernel
                padded_channels[c][y, x] = (kernel * image_padded[y:y + len(kernel), x:x + len(kernel)]).sum()

    # merge all channels to get final image
    final_image = cv2.merge((padded_channels[0], padded_channels[1], padded_channels[2]))

    return final_image


def convolution(image, kernel):

    # Flip kernel horizontally and vertically
    kernel = np.fliplr(np.flipud(kernel))

    return convolution_bgr(image, kernel)


def correlation(image, kernel):

    return convolution_bgr(image, kernel)


def median_filter(image, kernel_size):

    each_additional_padding = kernel_size // 2
    total_additional_padding = each_additional_padding * 2

    # create new padded image and set pixels to zero
    image_padded = np.zeros((image.shape[0] + total_additional_padding, image.shape[1] + total_additional_padding))

    # split image into its channels
    b, g, r = cv2.split(image)
    padded_channels = [np.zeros_like(b), np.zeros_like(g), np.zeros_like(r)]
    channels = [b, g, r]

    # cycle through every channel
    for c in range(3):

        # fill new image with each channel
        image_padded[each_additional_padding:-each_additional_padding, each_additional_padding:-each_additional_padding] = channels[c]
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):

                # perform matrix operations on image and kernel
                padded_channels[c][y, x] = (np.median(image_padded[y:y + kernel_size, x:x + kernel_size]))

    # merge all channels to get final image
    final_image = cv2.merge((padded_channels[0], padded_channels[1], padded_channels[2]))

    return final_image


def main():

    # import image
    img = cv2.imread('lena.png', 1)

    kernel1 = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])

    kernel2 = np.array([[1 / 9, 1 / 9, 1 / 9],
              [1 / 9, 1 / 9, 1 / 9],
              [1 / 9, 1 / 9, 1 / 9]])

    kernel3 = [[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
               [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
               [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
               [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
               [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]]

    kernel_gaussian_3 = np.array([[0.077847, 0.123317, 0.077847],
                                  [0.123317, 0.195346, 0.123317],
                                  [0.077847, 0.123317, 0.077847]])

    kernel_gaussian_5 = [[0.036894, 0.039167, 0.039956, 0.039167, 0.036894],
               [0.039167, 0.041581, 0.042418, 0.041581, 0.039167],
               [0.039956, 0.042418, 0.043272, 0.042418, 0.039956],
               [0.039167, 0.041581, 0.042418, 0.041581, 0.039167],
               [0.036894, 0.039167, 0.039956, 0.039167, 0.036894]]

    image_mean = convolution(img, kernel3)
    cv2.imshow('image', image_mean)

    # image_mean = correlation(img, kernel3)
    # cv2.imshow('image', image_mean)

    # image_sharpen = convolution(img, kernel1 - kernel2)
    # cv2.imshow('image', image_sharpen)

    # image_gaussian = convolution(img, kernel_gaussian_5)
    # cv2.imshow('image', image_gaussian)

    # image_median = median_filter(img, 10)
    # cv2.imshow('image', image_median)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
