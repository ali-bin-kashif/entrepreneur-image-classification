import cv2
import pywt
import numpy as np


# Haar cascade pre-trained classifiers
face_cascade = cv2.CascadeClassifier('opencv_harcascades\\haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('opencv_harcascades\\haarcascade_profileface.xml')
face_cascade_alt = cv2.CascadeClassifier('opencv_harcascades\\haarcascade_frontalface_alt.xml')

def get_visible_faces(img, is_path=False):
    """
    Detects visible faces in an image and extracts them as separate image regions.

    Parameters:
    -----------
    img : str or np.array
        The image to process, either as a file path (if a string) or an already loaded image array.

    Returns:
    --------
    list of np.array
        A list of face regions extracted from the input image, where each face is represented as a color image (ROI).
    """

    # Load the image if a file path is provided
    if is_path:
        img = cv2.imread(img)
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the pre-trained Haar Cascade classifier
    faces = face_cascade_alt.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

    # List to store extracted face images
    face_images = []

    # Loop through all detected faces and extract the regions of interest (ROI)
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face (optional)
        face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Crop and extract the region of interest for both grayscale and color images
        roi_gray = gray[y:y+h, x:x+w]   # Grayscale face region (optional for further processing)
        roi_color = face_img[y:y+h, x:x+w]  # Color face region

        # Append the extracted color face region to the list
        face_images.append(roi_color)
    
    return face_images


# Credits: This function was adapted from stackoverflow
def w2d(img, mode='haar', level=1):

    """
    Applies a 2D discrete wavelet transform to an image.

    Parameters:
        img (numpy array): The input image.
        mode (str, optional): The wavelet mode to use. Defaults to 'haar'.
        level (int, optional): The decomposition level. Defaults to 1.

    Returns:
        numpy array: The reconstructed image after applying the wavelet transform.
    """

    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H





def pre_process_image(img, image_size, is_path=False):
    """
    Pre-processes an image by optionally reading it from a file path, applying wavelet decomposition,
    resizing the original and wavelet-decomposed images, and combining them into a single array.

    Parameters:
    -----------
    img : str or np.array
        The image to process, either as a file path (if is_path=True) or an already loaded image array.
    image_size : int
        The size to which both the original and wavelet-decomposed images should be resized.
    is_path : bool, optional
        Flag indicating if `img` is a file path (default is False). If True, the image will be read from the given path.

    Returns:
    --------
    np.array
        A vertically stacked array of the resized original image and wavelet-decomposed image.
    """

    # Load the image if a file path is provided
    if is_path:
        img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply wavelet decomposition on the image using the w2d function
    img_har = w2d(img, 'db1', 5)
    
    # Resize the original image and the wavelet-decomposed image
    img_scaled = cv2.resize(img_gray, (image_size, image_size))
    img_har_scaled = cv2.resize(img_har, (image_size, image_size))
    # print(img_scaled.size, img_har_scaled.size)
    
    # Reshape and combine both images into a single vertical array
    combined_img = np.vstack((
        img_scaled.reshape(image_size * image_size, 1),  # Flatten and stack the RGB image
        img_har_scaled.reshape(image_size * image_size, 1)   # Flatten and stack the wavelet-decomposed image
    ))

    return combined_img.reshape(img_scaled.size + img_har_scaled.size)

