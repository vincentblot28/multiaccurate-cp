import random
import re


def find_image_number(img_name):
    city_name = img_name.split('_')[0]
    if len(re.findall(r"\d{2}", city_name)) > 0:
        return int(re.findall(r"\d{2}", city_name)[0])
    else:
        return int(re.findall(r"\d", city_name)[0])


def get_image_set(img_name, test_number, cal_number, percentage_val):
    """
    Determines the set (train, val, calibration and test) for a given image
    based its number.

    Parameters:
    image_number (int): The number of the image.
    test_number (int): The maximum number of images in the test set.
    cal_number (int): The minimum number of images in the calibration set.
    percentage_val (float): The percentage of images in the validation set.

    Returns:
    str: The machine learning set for the image. Possible values are "test", "cal", "val", or "train".
    """
    image_number = find_image_number(img_name)
    if image_number <= test_number:
        ml_set = "test"
    elif image_number >= cal_number:
        ml_set = "cal"
    else:
        if random.random() < percentage_val:
            ml_set = "val"
        else:
            ml_set = "train"
    return ml_set
