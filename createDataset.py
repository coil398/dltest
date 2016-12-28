import os
import six.moves.cPickle as pickle
import numpy as np
import cv2

from progressbar import ProgressBar


class FaceDataset:

    def __init__(self, dirs):
        self.data = None
        self.target = None
        self.n_types_target = -1
        self.dump_name = 'faces'
        self.image_size = 112
        print(dirs)
        self.dirs = dirs

    def read_images_in_dirs(self):
        for i, d in enumerate(self.dirs):
            pb = ProgressBar(min_value=0, max_value=len(os.listdir(d))).start()
            for j, f in enumerate(os.listdir(d)):
                pb.update(j)
                file_path = d + '/' + f
                image = cv2.imread(file_path, 0)
                image = image / 255
                self.data.append(image)
                self.target.append(i)
            pb.finish()
        self.data = np.array(self.data, np.float128)
        self.data = self.data.reshape(
            self.data.shape[0], self.image_size, self.image_size, 1)
        self.target = np.array(self.target, np.int32)
        # self.dump_dataset()

    def load_data_target(self):
        self.target = []
        self.data = []
        self.read_images_in_dirs()

    def dump_dataset(self):
        print('dumping')
        pickle.dump((self.data, self.target), open(
            self.dump_name + ".pkl", "w+b"), 2)
        print('done')

    def load_dataset(self):
        self.data, self.target = pickle.load(
            open(self.dump_name + ".pkl", "r+b"))


if __name__ == '__main__':
    dataset = FaceDataset(['male', 'female'])
    dataset.load_data_target()
