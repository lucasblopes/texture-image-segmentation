#!/usr/bin/env python3

import os
import cv2
import numpy as np
import shutil
import logging
from sklearn.cluster import KMeans

logging.basicConfig(
    format='%(levelname)s: %(message)s',
    level=logging.INFO
)


def build_filter_bank(scales=[4, 8, 16], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):

    filters = []
    for scale in scales:
        # Kernel size proportional to scale
        ksize = int(6 * scale)
        if ksize % 2 == 0:
            ksize += 1

        # Gabor filters
        for theta in orientations:
            kernel = cv2.getGaborKernel(
                ksize=(ksize, ksize),
                sigma=scale,
                theta=theta,
                lambd=10,
                gamma=0.5,
                psi=0
            )
            filters.append(kernel)

        # Circular filter: Laplacian of Gaussian
        gauss_1d = cv2.getGaussianKernel(ksize, scale)
        gauss_2d = gauss_1d @ gauss_1d.T
        log_kernel = cv2.Laplacian(gauss_2d, cv2.CV_64F)
        filters.append(log_kernel)

    logging.info(f'Built filter bank with {len(filters)} filters')
    return filters


def compute_filter_responses(gray_img, filters):

    responses = []
    for idx, kernel in enumerate(filters):
        resp = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
        responses.append(np.abs(resp))
        logging.debug(f'Applied filter {idx+1}/{len(filters)}')
    return responses


def segment_texture(responses, num_segments=4, window_size=16):

    h, w = responses[0].shape
    features = []
    positions = []

    # Extract mean response per window
    for y in range(0, h, window_size):
        for x in range(0, w, window_size):
            patch_vals = [resp[y:y+window_size, x:x+window_size].mean() for resp in responses]
            features.append(patch_vals)
            positions.append((y, x))

    features = np.array(features)
    logging.info(f'Clustering {features.shape[0]} windows into {num_segments} segments')
    kmeans = KMeans(n_clusters=num_segments, random_state=0)
    labels = kmeans.fit_predict(features)

    # Build segmentation map
    seg_map = np.zeros((h, w), dtype=np.uint8)
    for (y, x), lbl in zip(positions, labels):
        color_val = int(lbl * 255 / (num_segments - 1))
        seg_map[y:y+window_size, x:x+window_size] = color_val

    return seg_map, kmeans


def extract_global_descriptor(responses):

    desc = np.array([resp.mean() for resp in responses])
    return desc


def main():
    image_dir = './images'
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    logging.info('Starting texture segmentation pipeline')
    filters = build_filter_bank()

    # Gather images
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    descriptors = []
    paths = []

    # Process each image
    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        logging.info(f'Loading image {fname}')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Filter responses
        logging.info('Computing filter responses')
        responses = compute_filter_responses(gray, filters)

        # Global descriptor
        desc = extract_global_descriptor(responses)
        descriptors.append(desc)
        paths.append(img_path)

        # Segment by texture
        logging.info('Segmenting texture regions')
        seg_map, _ = segment_texture(responses)
        seg_out = os.path.join(output_dir, f'seg_{fname[:-4]}.png')
        cv2.imwrite(seg_out, seg_map)
        logging.info(f'Saved segmentation map to {seg_out}')

    # Cluster images based on global descriptors
    n_clusters = 4
    X = np.vstack(descriptors)
    logging.info(f'Clustering {len(descriptors)} images into {n_clusters} groups')
    img_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = img_kmeans.labels_

    # Create group directories and copy
    group_base = os.path.join(output_dir, 'groups')
    for i in range(n_clusters):
        os.makedirs(os.path.join(group_base, f'cluster_{i}'), exist_ok=True)

    for img_path, lbl in zip(paths, labels):
        dst = os.path.join(group_base, f'cluster_{lbl}', os.path.basename(img_path))
        shutil.copy(img_path, dst)
        logging.info(f'Copied {os.path.basename(img_path)} to cluster_{lbl}')

    logging.info('All processing complete')


if __name__ == '__main__':
    main()
