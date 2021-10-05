import os
import requests
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
import cv2 as cv
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.patches as patches


colors_dict = {
    'capacete' : 'yellow',
    'sem-capacete' : 'blue'
}


def setup_local(path='./stlocal'):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def download_file(url, path="./stlocal"):
    file_name = os.path.join(setup_local(path), os.path.basename(url))

    if not os.path.exists(file_name):
        r = requests.get(url)
        with open(file_name, 'wb') as fp:
            fp.write(r.content)

    return file_name


def url2pil(input_img):
    response = requests.get(input_img)
    return Image.open(BytesIO(response.content))


def np2pil(input_img, flag_bgr2rgb=True):
    if flag_bgr2rgb:
        input_img = bgr2rgb(input_img)
    return Image.fromarray(input_img)


def bgr2rgb(input_img):
    return cv.cvtColor(input_img, cv.COLOR_BGR2RGB)


def imshow(input_img, crop=(), flag_bgr2rgb=True):
    plt.figure()
    if isinstance(input_img, str):
        img = url2pil(input_img)
    elif isinstance(input_img, np.ndarray):
        img = np2pil(input_img, flag_bgr2rgb)
    else:
        img = input_img
    if len(crop) == 4:
        img = img.crop((crop[0], crop[1], crop[0]+crop[2], crop[1]+crop[3]))
    plt.axis("off")
    plt.imshow(img)
    return img


def print_detection(cap, frame, predictor, project_id, publish_iteration_name, freq_process, det=[], flag_print=False):
    _, encoded_image = cv.imencode('.png', frame)
    content2 = encoded_image.tobytes()
    im = np2pil(frame)

    if flag_print:
        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(im)
        plt.axis("off")

    results = predictor.detect_image(project_id, publish_iteration_name, content2)

    for prediction in results.predictions:
        if prediction.probability >= 0.5:
            left = int(prediction.bounding_box.left*im.size[0])
            top = int(prediction.bounding_box.top*im.size[1])
            width = int(prediction.bounding_box.width*im.size[0])
            height = int(prediction.bounding_box.height*im.size[1])
            idx = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            c = 1 * (prediction.tag_name == 'capacete')
            if c==1:
                det.append(f'{int(idx/freq_process)},{c},{left-5},{top-5},{width+10},{height+10},{prediction.probability:.4f},{idx}')
            
            if flag_print:
                # Create a Rectangle patch
                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=2,
                    edgecolor=colors_dict[prediction.tag_name], facecolor='none', ls=":"
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
    if flag_print:
        plt.show()
    
    return det


def cvt_rect2circle(start_point, end_point):
    center_point = (
        (start_point[0]+end_point[0])//2,
        (start_point[1]+end_point[1])//2
    )
    diamater = (
        ((start_point[0] - end_point[0]) ** 2) +
        ((start_point[1] - end_point[1]) ** 2)
    ) ** 0.5
    return center_point, int(np.ceil(diamater/2))


def circle_helmet(x, im, color=(23, 209, 248), label=None, line_thickness=2, box_type="circle"):
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    center_point, radius = cvt_rect2circle(c1, c2)
    radius = int(0.85 * radius)
    cv.circle(im, center_point, radius, color, thickness=tl, lineType=cv.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(label, 0, fontScale=tl / 2.9, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + 13, c1[1] - t_size[1] + 22
        cv.rectangle(im, (c1[0]-100, c1[1]-24), (c2[0]-90, c2[1]), color, -1, cv.LINE_AA)  # filled
        cv.putText(im, label, (c1[0]-90 , c1[1]-2 ), 0, tl / 2.9, [151, 81, 33], thickness=tf, lineType=cv.LINE_AA)


def interpolate_detections(det_df):
    det_df = det_df.reset_index()

    pos_max = int(det_df.idx.max())
    pos_min = int(det_df.idx.min())

    if pos_max - pos_min + 1 != det_df.shape[0]:
        x = np.linspace(pos_min, pos_max, num=pos_max-pos_min+1, endpoint=True)

        inter_df = pd.DataFrame()
        inter_df["idx"] = np.int64(x)
        inter_df["track_id"] = det_df["track_id"].unique()[0]

        for c in ["left", "top", "width", "height"]:
            f = interp1d(
                det_df["idx"].values, det_df[c].values, kind='cubic'
            )
            inter_df[c] = f(x)
    else:
        inter_df = det_df.loc[:, [
            "idx", "track_id", "left", "top", "width", "height"
        ]]

    return inter_df