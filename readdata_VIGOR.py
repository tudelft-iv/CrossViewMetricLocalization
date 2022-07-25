import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    # please modify the root
    root = '/scratch/zxia/datasets/VIGOR'
    
    def __init__(self, area, train_test):
        self.area = area
        self.train_test = train_test
        self.sat_size = [512, 512]  # [320, 320] or [512, 512]
        self.grd_size = [320, 640]  # [320, 640]  # [224, 1232]
        label_root = 'splits'
        
        if self.area == 'same':
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        elif self.area == 'cross':
            self.train_city_list = ['NewYork', 'Seattle']
            if self.train_test == 'train':
                self.test_city_list = ['NewYork', 'Seattle'] 
            elif self.train_test == 'test':
                self.test_city_list = ['SanFrancisco', 'Chicago'] 
                
        # load sat list, the training and test set both contain all satellite images
        self.train_sat_list = []
        self.train_sat_index_dict = {}
        idx = 0
        for city in self.train_city_list:
            train_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(train_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.train_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.train_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', train_sat_list_fname, idx)
        self.train_sat_list = np.array(self.train_sat_list)
        self.train_sat_data_size = len(self.train_sat_list)
        print('Train sat loaded, data size:{}'.format(self.train_sat_data_size))

        self.test_sat_list = []
        self.test_sat_index_dict = {}
        self.__cur_sat_id = 0  # for test
        idx = 0
        for city in self.test_city_list:
            test_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(test_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.test_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.test_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', test_sat_list_fname, idx)
        self.test_sat_list = np.array(self.test_sat_list)
        self.test_sat_data_size = len(self.test_sat_list)
        print('Test sat loaded, data size:{}'.format(self.test_sat_data_size))
        
        # load grd training list and test list. 
        self.train_list = []
        self.train_label = []
        self.train_sat_cover_dict = {}
        self.train_delta = []
        idx = 0
        for city in self.train_city_list:
            # load train panorama list
            if self.area == 'same':
                train_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_train.txt')
            if self.area == 'cross':
                train_label_fname = os.path.join(self.root, label_root, city, 'pano_label_balanced.txt')
            with open(train_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.train_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.train_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.train_label.append(label)
                    self.train_delta.append(delta)
                    if not label[0] in self.train_sat_cover_dict:
                        self.train_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.train_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print('InputData::__init__: load ', train_label_fname, idx)
        
        # split the original training set into training and validation sets
        if self.train_test == 'train':            
            self.train_list, self.val_list, self.train_label, self.val_label, self.train_delta, self.val_delta = train_test_split(self.train_list, self.train_label, self.train_delta, test_size=0.2, random_state=42) 
            
        elif self.train_test == 'test':
            self.val_list = []
            self.val_label = []
            self.test_sat_cover_dict = {}
            self.val_delta = []
            idx = 0
            for city in self.test_city_list:
                # load test panorama list
                if self.area == 'same':
                    test_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test.txt')
                if self.area == 'cross':
                    test_label_fname = os.path.join(self.root, label_root, city, 'pano_label_balanced.txt')
                with open(test_label_fname, 'r') as file:
                    for line in file.readlines():
                        data = np.array(line.split(' '))
                        label = []
                        for i in [1, 4, 7, 10]:
                            label.append(self.test_sat_index_dict[data[i]])
                        label = np.array(label).astype(np.int)
                        delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                        self.val_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                        self.val_label.append(label)
                        self.val_delta.append(delta)
                        if not label[0] in self.test_sat_cover_dict:
                            self.test_sat_cover_dict[label[0]] = [idx]
                        else:
                            self.test_sat_cover_dict[label[0]].append(idx)
                        idx += 1
                print('InputData::__init__: load ', test_label_fname, idx)
        
        self.train_label = np.array(self.train_label)
        self.train_delta = np.array(self.train_delta)
        self.val_label = np.array(self.val_label)
        self.val_delta = np.array(self.val_delta)
        self.train_data_size = len(self.train_list)
        self.val_data_size = len(self.val_list)
        self.trainIdList = [*range(0,self.train_data_size,1)]
        self.__cur_id = 0 
        self.valIdList = [*range(0,self.val_data_size,1)]
        self.__cur_test_id = 0
    
    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.val_data_size:
            self.__cur_test_id = 0
            return None, None, None
        elif self.__cur_test_id + batch_size >= self.val_data_size:
            batch_size = self.val_data_size - self.__cur_test_id
            
        batch_sat = np.zeros([batch_size, self.sat_size[0], self.sat_size[1], 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, self.grd_size[0], self.grd_size[1], 3], dtype=np.float32)
        batch_gt = np.zeros([batch_size, self.sat_size[0], self.sat_size[1], 1], dtype=np.float32)
        
        batch_idx = 0
        
        while True:
            if batch_idx >= batch_size or self.__cur_test_id >= self.val_data_size:
                break
                
            img_idx = self.__cur_test_id
            self.__cur_test_id += 1
            # ground
            img = cv2.imread(self.val_list[img_idx])

            if img is None:
                print('InputData: read fail: %s' % (self.val_list[img_idx]))
                continue
            img = img.astype(np.float32)
            img = cv2.resize(img, (self.grd_size[1], self.grd_size[0]), interpolation=cv2.INTER_AREA)
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[batch_idx, :, :, :] = img
            
            # satellite
            pos_index = 0 # we use the positive (no semi-positive) satellite images during testing
            img = cv2.imread(self.test_sat_list[self.val_label[img_idx][pos_index]])
            if img is None:
                print('InputData: read fail: %s' % (self.test_sat_list[self.val_label[img_idx][pos_index]]))
                continue
            img = img.astype(np.float32)
            img = cv2.resize(img, (self.sat_size[1], self.sat_size[0]), interpolation=cv2.INTER_AREA)
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[batch_idx, :, :, :] = img
            
            # get groundtruth location on the satellite map
            [col_offset, row_offset] = self.val_delta[img_idx, pos_index] # delta = [delta_lat, delta_lon]
            row_offset_resized = (row_offset/640*self.sat_size[0]).astype(np.int32)
            col_offset_resized = (col_offset/640*self.sat_size[0]).astype(np.int32)
            # Gaussian GT
            x, y = np.meshgrid(np.linspace(-self.sat_size[0]/2+row_offset_resized,self.sat_size[0]/2+row_offset_resized,self.sat_size[0]), np.linspace(-self.sat_size[0]/2-col_offset_resized,self.sat_size[0]/2-col_offset_resized,self.sat_size[0]))
            d = np.sqrt(x*x+y*y)
            sigma, mu = 4, 0.0
            batch_gt[batch_idx, :, :, 0] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
                    
            batch_idx += 1
            
        return batch_sat, batch_grd, batch_gt
            
    def next_pair_batch(self, batch_size):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.trainIdList)
        
        if self.__cur_id >= self.train_data_size:
            self.__cur_id = 0
            return None, None, None
        elif self.__cur_id + batch_size > self.train_data_size:
            batch_size = self.train_data_size - self.__cur_id
        
        batch_sat = np.zeros([batch_size, self.sat_size[0], self.sat_size[1], 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, self.grd_size[0], self.grd_size[1], 3], dtype=np.float32)
        batch_gt = np.zeros([batch_size, self.sat_size[0], self.sat_size[1], 1], dtype=np.float32)
        
        batch_idx = 0
        
        while True:
            if batch_idx >= batch_size or self.__cur_id >= self.train_data_size:
                break
            
            # load ground image
            img_idx = self.trainIdList[self.__cur_id]
            self.__cur_id += 1
            
            img = cv2.imread(self.train_list[img_idx])
            if img is None:
                print('InputData: read fail: %s, ' % (self.train_list[img_idx]))
                continue
            img = img.astype(np.float32)
            img = cv2.resize(img, (self.grd_size[1], self.grd_size[0]), interpolation=cv2.INTER_AREA)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[batch_idx, :, :, :] = img
            
            # load satellite image
            pos_index = random.randint(0,3) # each ground image is covered by 4 satellite images, randomly pick one
            img = cv2.imread(self.train_sat_list[self.train_label[img_idx][pos_index]])
            if img is None:
                print(
                    'InputData: read fail: %s, ' % (self.train_sat_list[self.train_label[img_idx][pos_index]]))
                continue
            img = img.astype(np.float32)
            img = cv2.resize(img, (self.sat_size[1], self.sat_size[0]), interpolation=cv2.INTER_AREA)
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[batch_idx, :, :, :] = img

            [col_offset, row_offset] = self.train_delta[img_idx, pos_index] # delta = [delta_lat, delta_lon]
            row_offset_resized = (row_offset/640*self.sat_size[0]).astype(np.int32)
            col_offset_resized = (col_offset/640*self.sat_size[0]).astype(np.int32)
            x, y = np.meshgrid(np.linspace(-self.sat_size[0]/2+row_offset_resized,self.sat_size[0]/2+row_offset_resized,self.sat_size[0]), np.linspace(-self.sat_size[0]/2-col_offset_resized,self.sat_size[0]/2-col_offset_resized,self.sat_size[0]))
            d = np.sqrt(x*x+y*y)
            sigma, mu = 4, 0.0

            # Gaussian GT
            batch_gt[batch_idx, :, :, 0] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

          
            batch_idx += 1
            
        return batch_sat, batch_grd, batch_gt
    
    def reset_scan(self):
        self.__cur_test_id = 0