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
from skimage import io , util, data, transform, feature
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage import util
from skimage.color import rgb2gray
from sklearn.naive_bayes import GaussianNB # For HOG
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from func_utils import get_face_on_image, addMirror, resize_sampling, sliding_window, IoU,\
    get_image, scan_images_multiprocessed, stats
import pickle
from glob import glob
import itertools
import random


# Generate positive data
def generatePositive():
    """
    Generate a dataset of positive
    """
    train_positive = [get_face_on_image(label_train[label_train.imNumber == imNumberr + 1].iloc[box_idx],
                                        get_image(train_files, imNumberr)) \
                      for imNumberr in range(len(train_files)) \
                      for box_idx in range(len(label_train[label_train.imNumber == imNumberr + 1]))
                      ]

    train_positive = [cv2.resize(img, (45, 60), interpolation=cv2.INTER_AREA) for img in train_positive]
    return train_positive


def generateNegative(debug=False):
        """
        Generate a Negative dataset by sampling and extracting patch on the train images.
        Only one rule applied to not generate a negative on a face : IoU with corresponding
        Bounding Rectangle must be less than 2
        """
        # loop over the sliding window for each layer of the pyramid
        patche_list = []
        for idx_image in range(len(train_files)):

            curr_bboxs = label_train[label_train.imNumber == idx_image+1]
            curr_img = get_image(train_files, idx_image)

            for resized, list_bboxs in resize_sampling(curr_img, curr_bboxs, scale=1.5):
                for (x, y, patch) in sliding_window(image=resized, x_step=40, y_step=40, patch_size=(60, 45)):
                    # if the window does not meet our desired window size, ignore it
                    if patch.shape[0] != 60 or patch.shape[1] != 45:
                        continue
                    # Check for IoU in image
                    patch_box = [idx_image+1, y, x, 60, 45]
                    valid = True
                    for box_idx in range(len(list_bboxs)):
                        if IoU(list_bboxs.iloc[box_idx], patch_box, generative=True) > 0.25:
                            # On ne prend pas le négatif
                            valid = False
                    if valid:
                        patche_list.append(patch)

                    if debug:
                        clone = resized.copy()
                        # Create figure and axes
                        fig,ax = plt.subplots(1)
                        ax.imshow(clone)
                        rect = patches.Rectangle((x, y), 60, 45, linewidth=2, edgecolor='g', facecolor='none')
                        ax.add_patch(rect)
                        plt.show()
        return patche_list


def get_false_positive_to_neg(fp_indexs):

        new_negs = np.zeros((len(fp_indexs), 60, 45, 3))
        new_negs = [transform.resize(list_img[tr.loc[ind].imNumber - 1][int(tr.loc[ind].y_corner):int(tr.loc[ind].y_corner + tr.loc[ind].height),
                                     int(tr.loc[ind].x_corner):int(tr.loc[ind].x_corner + tr.loc[ind].width)],
                                     (60, 45), mode="constant", anti_aliasing=True) for ind in fp_indexs]

        new_negs = [feature.hog(rgb2gray(util.img_as_float(img)), block_norm='L2-Hys') for img in new_negs]

        return new_negs


if __name__ == '__main__':

    # ---- Loading datas ---- #
    print('Loading data...\n')
    train_files = glob("data/train/*.jpg")

    label_train = pd.read_csv('./data/label_train.txt', delimiter='\s+', header=None)
    label_train = label_train.rename(columns={0: 'imNumber', 1: 'ycorner', 2: 'xcorner', 3: 'height', 4: 'width'})

    list_img = [get_image(train_files, im_number) for im_number in range(len(train_files))]

    print('Generating positiv data...\n')
    list_pos = generatePositive()
    list_pos = addMirror(list_pos)

    # Generate negative data
    print('Generating negativ data...\n')
    list_neg = generateNegative()
    list_neg = addMirror(list_neg)
    random.shuffle(list_neg)
    med = len(list_neg)/2
    list_neg = list_neg[:80000]

    # ---- Data builder ---- #
    print('Extracting HoG feature...\n')
    X_train = np.array([feature.hog(rgb2gray(util.img_as_float(im)), block_norm='L2-Hys') for im in itertools.chain(list_pos, list_neg)])
    y_train = np.zeros(X_train.shape[0])
    y_train[:list_pos.shape[0]] = 1
    c = list(zip(X_train, y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)

    # ---- Baseline Model Training ---- #
    print('Building intermediate model...\n')
    cross_val_score(GaussianNB(), X_train, y_train)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3],
                         'C': [10, 100, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 5, 10, 100]}]

    grid = GridSearchCV(SVC(), tuned_parameters)
    grid.fit(X_train, y_train)

    intermediate_clf = SVC(kernel="rbf", gamma="scale")
    intermediate_clf.max_iter = 3000

    print("Cross validating classifier...")
    cv_results = cross_validate(intermediate_clf, X_train, y_train, cv=3, return_train_score=False)
    print("Result of cross_validation :")
    print(cv_results["test_score"])

    intermediate_clf.fit(X_train, y_train)

    train_predicted = scan_images_multiprocessed(train_files, clf=intermediate_clf)

    precision, recall, f_score, precision_over_time, recall_over_time, glob_TP_ind, glob_FP_ind = stats(train_predicted.reset_index(drop=True))

    # plot the precision-recall curves
    fig = plt.figure(figsize=(16,10))
    plt.plot(recall_over_time, precision_over_time, marker='.', label='Results')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    # ---- Saving data ---- #
    print('Saving intermediate model results...\n')
    folder_path = "results/"
    train_predicted.reset_index(drop=True).to_pickle(folder_path + "detections_train_inter.pkl")

    # ----  Final model ---- #
    tr = pd.read_pickle(folder_path + "detections_train_inter.pkl")
    glob_FP_ind = np.array(tr[~tr.index.isin(glob_TP_ind)].index)

    print('Transforming False positiv to negative data...\n')

    negs_to_add = get_false_positive_to_neg(glob_FP_ind)

    X_train_with_FP = np.concatenate((X_train, negs_to_add), axis=0)
    y_train_with_FP = np.concatenate((y_train, np.full(len(negs_to_add), 0)))
    random_permut = np.random.permutation(X_train_with_FP.shape[0])
    X_train_with_FP = X_train_with_FP[random_permut]
    y_train_with_FP = y_train_with_FP[random_permut]

    print('Building final model...\n')
    final_clf = SVC(C=10000, kernel="rbf", gamma="scale")
    final_clf.max_iter = 5000

    final_clf.fit(X_train_with_FP, y_train_with_FP)

    cv_results_f = cross_validate(final_clf, X_train_with_FP, y_train_with_FP,
                                  cv=3,
                                  return_train_score=False)

    print("Result of cross_validation :")
    print(cv_results_f["test_score"])

    print('\nSaving model...\n')
    filename = 'final_model.sav'
    pickle.dump(final_clf, open(filename, 'wb'))
    print('model saved.')
