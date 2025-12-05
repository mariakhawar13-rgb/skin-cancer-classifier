import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(uploaded_file_or_PILimage, img_size=(224,224)):
    """
    Return uint8 numpy array in shape (h,w,3).
    We DON'T rescale to 0-1 here because the model includes preprocess_input.
    """
    if hasattr(uploaded_file_or_PILimage, "read"):
        # file-like (uploaded)
        img = image.load_img(uploaded_file_or_PILimage, target_size=img_size)
    else:
        # PIL.Image
        img = uploaded_file_or_PILimage.resize(img_size)
    arr = image.img_to_array(img)
    arr = arr.astype("uint8")
    return arr

def load_class_names():
    return [
        "actinic keratosis",
        "basal cell carcinoma",
        "dermatofibroma",
        "melanoma",
        "nevus",
        "pigmented benign keratosis",
        "seborrheic keratosis",
        "squamous cell carcinoma",
        "vascular lesion"
    ]
