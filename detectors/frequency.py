import cv2
import numpy as np

def measure_high_frequency_artifacts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    high_freq_score = np.mean(magnitude_spectrum)
    return high_freq_score
