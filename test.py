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
from func_utils import scan_images_multiprocessed, get_image
from glob import glob
import numpy as np
import pickle

# ---- Testing ---- #

if __name__ == '__main__':

    print('Loading model....\n')

    final_clf = pickle.load(open('final_model.sav', 'rb'))

    print('Loading images....\n')
    test_files = glob("test/*.jpg")
    list_img_test = [get_image(test_files, im_number) for im_number in range(len(test_files))]

    print('Predicting images....\n')
    test_predicted = scan_images_multiprocessed(test_files, clf=final_clf, xstep=4, ystep=4)

    # to int for columns
    test_predicted = test_predicted.reset_index(drop=True)
    test_predicted.astype({'imNumber': 'int64', 'y_corner': 'int64', 'x_corner': 'int64', 'height': 'int64',
                            'width': 'int64', 'score': 'float64'}).dtypes
    test_predicted = test_predicted[['imNumber', 'y_corner', 'x_corner', 'height', 'width', 'score']]

    print('Prediction complete. \n')

    folder_path = "results/"

    np.savetxt(folder_path + "test_results_WOAH.txt", test_predicted.values, fmt='%i %i %i %i %i %1.4f')
    print('Results stored in result folder.')
