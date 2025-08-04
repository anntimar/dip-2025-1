import cv2 as cv
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    # TODO: Implement noise removal here (e.g., median filtering)
    denoised = cv.medianBlur(image, 3)
    denoised = cv.medianBlur(denoised, 3)
    return denoised  # Replace this with your filtering implementation

if __name__ == "__main__":
    noisy_image = cv.imread("head.png", cv.IMREAD_GRAYSCALE)
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    cv.imwrite("head_filtered.png", denoised_image)
