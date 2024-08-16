from __future__ import division, print_function

import os
import random
import codecs
import cv2
import numpy as np
from sample_preprocessor import preprocess


class Sample:
    def __init__(self, gt_text, file_path):
        self.gt_text = gt_text
        self.file_path = file_path


class Batch:
    def __init__(self, gt_texts, imgs):
        self.imgs = np.stack(imgs, axis = 0)
        self.gt_texts = gt_texts


class DataLoader:
    def __init__(self, file_path, batch_size, img_size, max_text_len):
        """
        Loader for the dataset
        :param file_path: File path of the image
        :param batch_size: Batch size
        :param img_size: Size of the image
        :param max_text_len: Maximum text length
        """
        self.data_augmentation = False
        self.cur_idx = 0
        self.batch_size = batch_size
        self.img_size = img_size
        self.samples = []

        with codecs.open(file_path + "full.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        chars = set()
        print(lines[5])
        for line in lines:
            if not line or line[0] == '#':
                continue
            line_split = line.strip().split(' ')
            if line_split[0] == '\ufeff':
                continue
            file_name = file_path + line_split[0]

            # Ground Truth text starts at column 1
            gt_text = self.truncate_label(' '.join(line_split[1]), max_text_len)
            chars = chars.union(set(list(gt_text)))

            # Check if image not empty
            if not os.path.getsize(file_name):
                continue
            self.samples.append(Sample(gt_text, file_name)) # This can be a dictionary

        # Split into training, validation and testing sets
        n1, n2 = int(0.8*len(self.samples)), int(0.9*len(self.samples))
        self.train_samples = self.samples[:n1]
        self.validation_samples = self.samples[n1:n2]
        self.test_samples = self.samples[n2:]

        # Put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.test_words = [x.gt_text for x in self.test_samples]
        self.valid_words = [x.gt_text for x in self.validation_samples]

        # Number of randomly chosen samples per epoch
        self.num_train_samples_per_epoch = 10000

        self.train_set()

        # List of chars in the dataset
        self.char_list = sorted(list(chars))

    @staticmethod
    def truncate_label(text, max_text_len):
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def train_set(self):
        """
        Switch to randomly chosen subset of training set
        :return: None
        """
        self.data_augmentation = True
        self.cur_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples[:self.num_train_samples_per_epoch]

    def validation_set(self):
        """
        Switch to validation set
        :return:
        """
        self.data_augmentation = False
        self.cur_idx = 0
        random.shuffle(self.validation_samples)
        self.samples = self.validation_samples

    def test_set(self):
        """
        Switch to testing set
        :return:
        """
        self.data_augmentation = False
        self.cur_idx = 0
        random.shuffle(self.test_samples)
        self.samples = self.test_samples

    def get_iterator_info(self):
        """
        Current batch index and total number of batches
        :return:
        """
        return self.cur_idx // self.batch_size + 1, len(self.samples) // self.batch_size

    def has_next(self):
        return self.cur_idx + self.batch_size <= len(self.samples)

    def get_next(self):
        batch_range = range(self.cur_idx, self.cur_idx + self.batch_size)
        gt_texts = [self.samples[i].gt_text for i in batch_range]
        imgs = [preprocess(cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE), self.img_size, self.data_augmentation) for i in batch_range]
        self.cur_idx += self.batch_size
        return Batch(gt_texts, imgs)