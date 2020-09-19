# -*- coding: utf-8 -*-
"""
# Project : Face Recognition

- Obtain a set of image thumbnails of faces to constitute "positive" training samples.
- Obtain a set of image thumbnails of non-faces to constitute "negative" training samples.
- Extract HOG features from these training samples.
- Train a linear SVM classifier on these samples.
- For an "unknown" image, pass a sliding window across the image, using the model to evaluate whether that window contains a face or not.
- If detections overlap, combine them into a single window.

@copyright Hugo Paigneau
"""


# ---- Librairies ---- #
import numpy as np
import cv2
import itertools
from skimage import io , util, data, transform, feature
import pandas as pd
from skimage.color import rgb2gray
from multiprocessing import Pool


# ---- Global Functions ---- #
def get_image(list_files, im_number):
    choosen_img = list_files[im_number]
    random_im = cv2.imread(choosen_img)
    random_img = cv2.cvtColor(random_im, cv2.COLOR_BGR2RGB)
    return random_img


def IoU(box1, box2, generative=False):
    """
    Return IoU of 2 box with Bounding Rectangle format (x,y,w,h)
    """

    im_number, ycorner1, xcorner1, height1, width1 = box1
    im_number, ycorner2, xcorner2, height2, width2 = box2

    if generative:
        # Check if box1 in box2
        if ycorner1 > ycorner2 and xcorner1 > xcorner2 and (height1+ycorner1) < (height2+ycorner2) and (width1+xcorner1) < (width2+xcorner2):
            return 1
        # Check if box2 in box1
        if ycorner2 > ycorner1 and xcorner2 > xcorner1 and (height2+ycorner2) < (height1+ycorner1) and (width2+xcorner2) < (width1+xcorner1):
            return 1

    w_intersection = min(xcorner1 + width1, xcorner2 + width2) - max(xcorner1, xcorner2)
    h_intersection = min(ycorner1 + height1, ycorner2 + height2) - max(ycorner1, ycorner2)
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0

    I = w_intersection * h_intersection
    U = width1 * height1 + width2 * height2 - I # Union = Total Area - I

    return I / U


def IoUScore(box1, box2, generative=False):
    """
    Return IoU of 2 box with Bounding Rectangle format (x,y,w,h)
    """
    id, ycorner1, xcorner1, height1, width1, score1, imNumber = box1
    id2, imNumber, ycorner2, xcorner2, height2, width2 = box2

    if generative:
        #Check if box1 in box2
        if ycorner1 > ycorner2 and xcorner1 > xcorner2 and (height1+ycorner1) < (height2+ycorner2) and (width1+xcorner1) < (width2+xcorner2):
            return 1
        #Check if box2 in box1
        if ycorner2 > ycorner1 and xcorner2 > xcorner1 and (height2+ycorner2) < (height1+ycorner1) and (width2+xcorner2) < (width1+xcorner1):
            return 1

    w_intersection = min(xcorner1 + width1, xcorner2 + width2) - max(xcorner1, xcorner2)
    h_intersection = min(ycorner1 + height1, ycorner2 + height2) - max(ycorner1, ycorner2)
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
      return 0

    I = w_intersection * h_intersection
    U = width1 * height1 + width2 * height2 - I # Union = Total Area - I

    return I / U


def IoUDup(box1, box2, final=False, generative=False):
    """
    Return IoU of 2 box with Bounding Rectangle format (x,y,w,h)
    """
    ycorner1, xcorner1, height1, width1, score1 = box1
    ycorner2, xcorner2, height2, width2, score2 = box2

    if generative:
      #Check if box1 in box2
        if ycorner1 > ycorner2 and xcorner1 > xcorner2 and (height1+ycorner1) < (height2+ycorner2) and (width1+xcorner1) < (width2+xcorner2):
            return 1
        #Check if box2 in box1
        if ycorner2 > ycorner1 and xcorner2 > xcorner1 and (height2+ycorner2) < (height1+ycorner1) and (width2+xcorner2) < (width1+xcorner1):
            return 1

    w_intersection = min(xcorner1 + width1, xcorner2 + width2) - max(xcorner1, xcorner2)
    h_intersection = min(ycorner1 + height1, ycorner2 + height2) - max(ycorner1, ycorner2)
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0

    I = w_intersection * h_intersection
    U = width1 * height1 + width2 * height2 - I # Union = Total Area - I

    # Check if area of one is 90% in other
    if final:
        if (width1 * height1) / (width2 * height2) > 2.5 or (width2 * height2) / (width1 * height1) > 2.5:
            covera = 100 * I / (width2 * height2)
            coverb = 100 * I / (width1 * height1)
            if covera > 80 or coverb > 80:
                return 2

    return I / U


def remove_duplicates(detections, final=False):
    """
    remove duplicates for all detection's boundind boxs
    """
    glob_del_ind = []
    for pair in itertools.combinations(range(len(detections)), r=2):
        if final:
            iou = IoUDup(detections.loc[pair[0]], detections.loc[pair[1]], final=True)
        else:
            iou = IoUDup(detections.loc[pair[0]], detections.loc[pair[1]])
        if pair[0] in glob_del_ind or pair[1] in glob_del_ind:
            pass
        if iou == 2:
            glob_del_ind.append(pair[1] if (detections.loc[pair[0]].height * detections.loc[pair[0]].width) > (detections.loc[pair[1]].height * detections.loc[pair[1]].width) else pair[0])
        elif iou > 0.40:
            glob_del_ind.append(pair[1] if detections.loc[pair[0]].score > detections.loc[pair[1]].score else pair[0])

    return detections[~detections.index.isin(glob_del_ind)]


def resize_sampling(image, list_bboxs, scale=1.5, min_size=(200, 200)):
    """
    returns images scaled with a certain ratio
    """
    yield image, list_bboxs
    while True:
        width = int(image.shape[1] / scale)
        height = int(image.shape[0] / scale)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        copy_list_bboxs = list_bboxs.copy()

        # Resize bbox
        for box_idx in range(len(copy_list_bboxs)):
            curr_im_number, curr_ycorner, curr_xcorner, curr_height, curr_width = copy_list_bboxs.iloc[box_idx]
            new_xcorner = int(np.round(curr_xcorner / scale))
            new_ycorner = int(np.round(curr_ycorner / scale))
            new_height = int(np.round(curr_height / scale))
            new_width = int(np.round(curr_width / scale))
            copy_list_bboxs.iloc[box_idx] = [curr_im_number, new_ycorner, new_xcorner, new_height, new_width]

        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image, copy_list_bboxs


def resize_sampling_s(image, scale=1.5, min_size=(140, 140)):
    """
    returns images scaled with a certain ratio
    """
    yield image, (image.shape[0], image.shape[1])
    while True:

        width = round(image.shape[1] / scale)
        height = round(image.shape[0] / scale)
        dim = (width, height) #x,y
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image, (dim[1], dim[0])


def sliding_window(image, x_step, y_step, patch_size):
    """
    Slides a box across the image with a given x_step and y_step
    """
    for y in range(0, image.shape[0], y_step):
        for x in range(0, image.shape[1], x_step):
            # yield the current window
            yield (x, y, image[y:y + patch_size[0], x:x + patch_size[1]])


def sliding_windowb(img, patch_size=(120,90),
                   xstep=3, ystep=3, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, xstep):
        for j in range(0, img.shape[1] - Ni, ystep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


def get_face_on_image(box, img):
    """
    On a given image, identify a face by its BoundingRectangle and isolate it
    @parameter box : Bounding Rectangle that indentify the face
    @parameter img : Numpy format image
    """
    img_shape = img.shape
    imNumber, ycorner, xcorner, height, width = box
    face = img[ycorner:ycorner+height,xcorner:xcorner+width]
    return face


def addMirror(list_to_mirror):
    list_mirror = [cv2.flip(curr_im, 1) for curr_im in list_to_mirror]
    return np.array([im for im in itertools.chain(list_mirror, list_to_mirror)])


def test_photo(img, clf, xstep=6, ystep=6, debug=False):
    """
    For a given image,
    predict face on current image and returns its bounding boxs, may be plotable if plot=True
    """

    img_dim = img.shape
    sample = img.copy()
    predictions = pd.DataFrame([], columns=['y_corner', 'x_corner', 'height', 'width', 'score'])
    if debug:
        print("\n -- GLOBAL IMAGE SHAPE : \n")
        print(img_dim)
    for resized, curr_shape in resize_sampling_s(sample, scale=1.05):
        if debug:
            print('\n    |- New image resized ---')
            print("\n          ---- Current img shape : \n")
            print(curr_shape)
        indices, patches = zip(*sliding_windowb(resized, xstep=xstep, ystep=ystep))
        patches_hog = np.array([feature.hog(
            rgb2gray(util.img_as_float(cv2.resize(patch, (45, 60), interpolation=cv2.INTER_AREA))), block_norm='L2-Hys')
                                for patch in patches])

        # 1 = a detection
        labels = clf.predict(patches_hog)

        # Scores
        scores = clf.decision_function(patches_hog)

        # select detected only
        detection_boxs = np.asarray(indices)[np.where(labels == 1)]
        detection_labels = labels[np.where(labels == 1)]
        detection_score = scores[np.where(scores > 0)]

        # order
        detections = [[location[0], location[1], 120, 90, score] for score, location in
                      sorted(zip(detection_score, detection_boxs), key=lambda pair: pair[0], reverse=True)]
        # Remove IoU
        df_detections = pd.DataFrame(detections, columns=['y_corner', 'x_corner', 'height', 'width', 'score'])
        df_detections.reset_index(inplace=True, drop=True)
        df_detections = remove_duplicates(df_detections)

        # scale to img
        scale_val = img_dim[0] / curr_shape[0]
        if debug:
            print("\n          scale value : \n")
            print(scale_val)
        df_detections = df_detections.apply(lambda x: round(x * scale_val) if x.name not in 'score' else x)
        if debug:
            print("\n          ----- Detection has been resized in : ----------\n")
            print(df_detections.head())
        predictions = pd.concat([predictions, df_detections])

    # Sort pred list
    predictions = predictions.sort_values(by=['score'], ascending=False)
    predictions.reset_index(inplace=True, drop=True)
    print(predictions)
    predictions = remove_duplicates(predictions, final=True)
    return predictions


def scan_images(images, begin, xstep, ystep, clf):
    """
    Scan a batch of images
    """
    predictions = pd.DataFrame([], columns=['y_corner', 'x_corner', 'height', 'width', 'score', 'imNumber'])

    for i in range(len(images)):
        curr_preds = test_photo(get_image(images, i), clf=clf , xstep=xstep, ystep=ystep)
        curr_preds['imNumber'] = begin + i + 1
        predictions = pd.concat([predictions, curr_preds])

    return predictions


def scan_images_multiprocessed(images, clf, processes=8, xstep=6, ystep=6):
    """
    Scan a batch of images with multiple processes
    """

    print('Begin predincting images...\n')
    pool = Pool(processes=processes)
    results = []
    for i in range(0, processes):
        begin = i * int(len(images) / processes)
        if i == processes - 1:
            end = len(images)
        else:
            end = (i + 1) * int(len(images) / processes)

        results.append(pool.apply_async(scan_images, (images[begin:end], begin, xstep, ystep, clf)))

    predictions = pd.DataFrame([], columns=['y_corner', 'x_corner', 'height', 'width', 'score', 'imNumber'])
    for result in results:
        res = result.get()
        predictions = pd.concat([predictions, res])

    return predictions


def stats(detec, labels):

    precision_over_time = []
    recall_over_time = []
    glob_TP_ind = []
    glob_FP_ind = []

    for idx in range(1000):
        framed = detec[detec.imNumber == idx+1]
        framed = framed.reset_index()
        y_test = labels[labels.imNumber == idx+1]
        y_test = y_test.reset_index()
        list_detected_index = []
        for pred_index in range(len(framed)):
            prediction_index = int(framed.loc[pred_index]['index'])
            for test_index in range(len(y_test)):
                iou = IoUScore(framed.loc[pred_index], y_test.loc[test_index])
                detected_index = int(y_test.loc[test_index]['index'])
                if iou > 0.5:
                    if detected_index in list_detected_index:
                        glob_FP_ind.append(prediction_index)
                    else:
                        # Adding true positive index
                        glob_TP_ind.append(prediction_index)
                        # setting test index as detected
                        list_detected_index.append(int(detected_index))

            preci = len(glob_TP_ind) / (prediction_index + 1)
            reca = len(glob_TP_ind) / (len(labels))
            precision_over_time.append(preci)
            recall_over_time.append(reca)

    precision = precision_over_time[-1]
    recall = recall_over_time[-1]
    f_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f_score, precision_over_time, recall_over_time, glob_TP_ind, glob_FP_ind

