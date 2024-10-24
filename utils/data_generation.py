import numpy as np
import cv2

def rotate_image(image, angle):
    """Rotate image by given angle"""
    img = image.reshape(28, 28)
    center = (14, 14)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (28, 28))
    return rotated.reshape(784)


def add_noise(image, noise_level):
    """Add random noise to image"""
    noise = np.random.normal(0, noise_level, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)

