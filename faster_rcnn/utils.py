import numpy as np
import cv2
import colorsys

def format_img_size(img, cfg):
    """ formats the image size based on config """
    img_min_side = float(cfg.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, cfg):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    img /= cfg.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2

def _create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def _create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id or class in detection (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = _create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def draw_boxes_and_label_on_image_cv2(img, class_label_map, class_boxes_map):
    """
    this method using cv2 to show boxes on image with various class labels
    :param img:
    :param class_label_map: {1: 'Car', 2: 'Pedestrian'}
    :param class_boxes_map: {1: [box1, box2..], 2: [..]}, in every box is [bb_left, bb_top, bb_width, bb_height, prob]
    :return:
    """
    for c, boxes in class_boxes_map.items():
        for box in boxes:
            assert len(box) == 5, 'class_boxes_map every item must be [bb_left, bb_top, bb_width, bb_height, prob]'
            # checking box order is bb_left, bb_top, bb_width, bb_height
            # make sure all box should be int for OpenCV
            bb_left = int(box[0])
            bb_top = int(box[1])
            bb_width = int(box[2])
            bb_height = int(box[3])

            # prob will round 2 digits
            prob = round(box[4], 2)
            unique_color = _create_unique_color_uchar(c)
            cv2.rectangle(img, (bb_left, bb_top), (bb_width, bb_height), unique_color, 2)

            text_label = '{} {}'.format(class_label_map[c], prob)
            (ret_val, base_line) = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            text_org = (bb_left, bb_top - 0)

            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line - 5),
                          (text_org[0] + ret_val[0] + 5, text_org[1] - ret_val[1] + 5), unique_color, 2)
            # this rectangle for fill text rect
            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line - 5),
                          (text_org[0] + ret_val[0] + 5, text_org[1] - ret_val[1] + 5),
                          unique_color, -1)
            cv2.putText(img, text_label, text_org, cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    return img