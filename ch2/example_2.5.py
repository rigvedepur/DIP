import cv2
import imutils
import numpy as np


def grayscale_and_resize(image):
    image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return imutils.resize(image, height=500)

def create_noisy_images(image, num_of_images = 10, noise_sigma = 25):
    image = image.astype('float64')

    n_images = []
    for i in range(num_of_images):

        noise = np.random.randn(image.shape[0], image.shape[1]) * noise_sigma
        noisy_image = image + noise
        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        n_images.append(noisy_image.astype(np.uint8))

    return n_images

def generate_averaged_image(noisy_image_list):

    sum_image = np.zeros(noisy_image_list[0].shape, 'float64')
    for image in noisy_image_list:
            image = np.float64(image)
            sum_image += image
    avg_image = sum_image / float(len(noisy_image_list))
    cv2.normalize(avg_image, avg_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    avg_image = avg_image.astype(np.uint8)

    return avg_image


if __name__ == '__main__':
    file = r"C:\Users\REpur143897\Documents\Python\DIP\data\sombrero-galaxy -original.tif"
    image = cv2.imread(file)
    image = grayscale_and_resize(image)
    cv2.imshow('Original Image', image)
    noisy_images = create_noisy_images(image, 100)
    cv2.imshow('Sample Noisy Image', noisy_images[0])
    averaged_image = generate_averaged_image(noisy_images)
    cv2.imshow('Averaged Image', averaged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




