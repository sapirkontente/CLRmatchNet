import cv2
import os
import os.path as osp
import torch.nn as nn

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def add_conf(conf, color, delta_pos, img):
    # Add a title to the line
    title_text = conf
    text_position = (100, 80+delta_pos)  # Position of the text

    # Define the text color (BGR format)
    text_color = color  

    # Define the background color (BGR format)
    background_color = (255, 255, 255)  # White

    # Choose the font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8    

    # Get the size of the text box
    (text_width, text_height), baseline = cv2.getTextSize(title_text, font, font_scale, thickness=2)

    # Calculate the position for the background rectangle
    background_position = (text_position[0], text_position[1] - text_height)

    # Create the background rectangle
    cv2.rectangle(img, background_position, (background_position[0] + text_width, background_position[1] + text_height), background_color, thickness=cv2.FILLED)

    # Add the text to the image
    img = cv2.putText(img, title_text, text_position, font, font_scale, text_color, thickness=2)

    return img


def imshow_lanes(img, lanes, conf, show=False, out_file=None, width=4):
    lanes_xys = []
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)\
            
    lanes_xys.sort(key=lambda xys : xys[0][0])

    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
            img = add_conf(str(round(conf[idx].item(),2)), COLORS[idx], idx*40, img)


    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)