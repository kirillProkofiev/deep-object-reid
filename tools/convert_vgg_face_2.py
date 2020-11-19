import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2 as cv
import numpy as np
import os
import re

import argparse

from torchreid.data.datasets.image.lfw import FivePointsAligner


class VGGFace2_raw(Dataset):
    """VGGFace2 Dataset compatible with PyTorch DataLoader."""
    def __init__(self, images_root_path, image_list_path, landmarks_folder_path='',
                 transform=None, landmarks_training=False, dest_size=200):
        self.image_list_path = image_list_path
        self.images_root_path = osp.join(images_root_path, 'train')
        self.identities = {}
        assert dest_size > 10
        self.dest_size = dest_size

        self.landmarks_file = None
        self.detections_file = None
        if osp.isdir(landmarks_folder_path):
            if 'train' in image_list_path:
                bb_file_name = 'loose_landmark_train.csv'
                landmarks_file_name = 'loose_bb_train.csv'
            elif 'test' in image_list_path:
                bb_file_name = 'loose_landmark_test.csv'
                landmarks_file_name = 'loose_bb_test.csv'
            else:
                bb_file_name = 'loose_landmark_all.csv'
                landmarks_file_name = 'loose_bb_all.csv'
            self.landmarks_file = open(osp.join(landmarks_folder_path, bb_file_name), 'r')
            self.detections_file = open(osp.join(landmarks_folder_path, landmarks_file_name), 'r')
        self.have_landmarks = not self.landmarks_file is None
        self.landmarks_training = landmarks_training
        if self.landmarks_training:
            assert self.have_landmarks is True

        self.samples_info = self._read_samples_info()

        self.transform = transform

    def _read_samples_info(self):
        """Reads annotation of the dataset"""
        samples = []

        with open(self.image_list_path, 'r') as f:
            last_class_id = -1
            images_file_lines = f.readlines()

            if self.have_landmarks:
                detections_file_lines = self.detections_file.readlines()[1:]
                landmarks_file_lines = self.landmarks_file.readlines()[1:]

                assert len(detections_file_lines) == len(landmarks_file_lines)
                assert len(images_file_lines) == len(detections_file_lines)

            for i in tqdm(range(len(images_file_lines))):
                sample = images_file_lines[i].strip()
                sample_id = int(sample.split('/')[0][1:])
                frame_id = int(sample.split('/')[1].split('_')[0])
                if sample_id in self.identities:
                    self.identities[sample_id].append(len(samples))
                else:
                    last_class_id += 1
                    self.identities[sample_id] = [len(samples)]
                if not self.have_landmarks:
                    samples.append((osp.join(self.images_root_path, sample), last_class_id, frame_id))
                else:
                    _, bbox = detections_file_lines[i].split('",')
                    bbox = [max(int(coord), 0) for coord in bbox.split(',')]
                    _, landmarks = landmarks_file_lines[i].split('",')
                    landmarks = [float(coord) for coord in landmarks.split(',')]
                    samples.append((osp.join(self.images_root_path, sample), last_class_id, sample_id, bbox, landmarks))

        return samples

    def get_num_classes(self):
        """Returns total number of identities"""
        return len(self.identities)

    def __len__(self):
        """Returns total number of samples"""
        return len(self.samples_info)

    def __getitem__(self, idx):
        """Returns sample (image, class id, image id) by index"""
        img = cv.imread(self.samples_info[idx][0], cv.IMREAD_COLOR)
        if self.landmarks_training:
            landmarks = self.samples_info[idx][-1]
            bbox = self.samples_info[idx][-2]
            img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            landmarks = [(float(landmarks[2*i]-bbox[0]) / bbox[2],
                          float(landmarks[2*i + 1]-bbox[1])/ bbox[3]) for i in range(len(landmarks)//2)]
            data = {'img': img, 'landmarks': np.array(landmarks)}
            if self.transform:
                data = self.transform(data)
            return data

        if self.have_landmarks:
            landmarks = self.samples_info[idx][-1]
            img = FivePointsAligner.align(img, landmarks, d_size=(200, 200), normalized=False)

        if self.transform:
            img = self.transform(img)
        # path, image, class_id, instance_id
        return (self.samples_info[idx][0], img, self.samples_info[idx][1], self.samples_info[idx][2])


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', type=str, default='', help='path to input data', required=True)
    parser.add_argument('--output-dir', type=str, default='', help='path to output data root', required=True)
    parser.add_argument('--output-size', type=int, default=200)
    parser.add_argument('--output-ext', type=str, default='png')

    args = parser.parse_args()

    assert osp.isdir(args.output_dir)
    dataset = VGGFace2_raw(args.data_root, osp.join(args.data_root, 'meta/all_list.txt'),
                                           osp.join(args.data_root, 'bb_landmark'),
                                           dest_size=args.output_size)

    with open(osp.join(args.output_dir, 'all_list.txt'), 'w') as list_f:
        for item in tqdm(dataset, 'Processing data'):
            path = item[0]
            image = item[1]
            tokens = path.split(osp.sep)
            folder = osp.join(args.output_dir, tokens[-2])
            img_name = osp.splitext(tokens[-1])[0] + '.' + args.output_ext
            if not osp.isdir(folder):
                os.mkdir(folder)
            cv.imwrite(osp.join(folder, img_name), image)
            list_f.write(osp.join(tokens[-2], img_name) + '\n')
        print('Finished!')

if __name__ == '__main__':
    main()