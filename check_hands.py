
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
import scipy.io
import os
from tqdm import tqdm
import numpy as np
import cv2


# class that defines and loads the kangaroo dataset
class HandsDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, data_type='train'):
        # define one class
        self.add_class("dataset", 1, "hand")
        # define data locations
        images_dir = dataset_dir + data_type + '_data/images/'
        annotations_dir = dataset_dir + data_type + '_data/annotations/'
        img_names = [name.split('.')[0] for name in os.listdir(images_dir)]
        for img_idx, img_name in enumerate(tqdm(img_names)):
            if img_name == '':
                continue
            self.add_image(source='dataset',
                           image_id=img_idx,
                           path=images_dir + img_name + '.jpg',
                           annotation=annotations_dir + img_name + '.mat')

    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes = self.coords_from_mat(path)
        box_array = np.array([[(x, y) for y, x in box] for box in boxes])
        path = info['path']
        img = cv2.imread(path)
        height = img.shape[0],
        width = img.shape[1],
        # create one array for all masks, each on a different channel
        masks = np.zeros([height, width, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i_box, box in enumerate(box_array):
            e_mask = np.zeros_like(img[:, :, 1])
            masks[:, :, i_box] = cv2.fillPoly(e_mask, [box.astype(np.int32)], 1)
            class_ids.append(self.class_names.index('hand'))
        return masks, np.asarray(class_ids, dtype='int32')

    # load an image reference  <---- This in the second of the 2 required functions
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def coords_from_mat(mat_filepath):
        mat = scipy.io.loadmat(mat_filepath)
        coords = []
        i = 0
        for e in mat['boxes'][0]:
            coords.append(list())
            c = 0
            for d in e[0][0]:
                if c > 3:
                    break
                coords[i].append((d[0][0], d[0][1]))
                c += 1
            i += 1
        return coords


# train set
train_set = HandsDataset()
train_set.load_dataset('./datasets/hands/')
train_set.prepare()
# define image id
image_id = 1
# load the image
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)
