import numpy as np
import cv2
import copy
import os
import random
import math


class DataLoader:
    def __init__(self, train_test):
        self.train_test = train_test
        self.grd_image_root = '/scratch/zxia/datasets/Oxford_5m_sampling/' # path to the ground images
        # read the full satellite image into memory
        self.full_satellite_map = cv2.imread('/scratch/zxia/datasets/Oxford_5m_sampling/satellite_map_new.png') # path to the satellite image

        trainlist = []
        with open(self.grd_image_root+'training.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                trainlist.append(content.split(" "))
        
        with open('Oxford_split/train_yaw.npy', 'rb') as f:
            self.train_yaw = np.load(f)
        
        vallist = []
        with open('Oxford_split/validation.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                vallist.append(content.split(" "))
        
        # 3 test traversals
        test_2015_08_14_14_54_57 = []
        with open('Oxford_split/test1_j.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                test_2015_08_14_14_54_57.append(content.split(" "))
        test_2015_08_12_15_04_18 = []
        with open('Oxford_split/test2_j.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                test_2015_08_12_15_04_18.append(content.split(" "))
        test_2015_02_10_11_58_05 = []
        with open('Oxford_split/test3_j.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                test_2015_02_10_11_58_05.append(content.split(" "))
        testlist = test_2015_08_14_14_54_57 + test_2015_08_12_15_04_18 + test_2015_02_10_11_58_05
                
        self.trainList = trainlist
        self.trainNum = len(trainlist)
        trainarray = np.array(trainlist)
        self.trainUTM = np.transpose(trainarray[:,2:].astype(np.float64))
        print('number of ground images in training set', self.trainNum)    
       
        if self.train_test == 'train': 
            self.valList = vallist
            self.valNum = len(vallist)
            valarray = np.array(vallist)
            self.valUTM = np.transpose(valarray[:,2:].astype(np.float64))    
            print('number of ground images in validation set', len(vallist)) 
            with open('Oxford_split/val_yaw.npy', 'rb') as f:
                self.val_yaw = np.load(f)
        if self.train_test == 'test': 
            self.valList = testlist
            self.valNum = len(testlist)
            valarray = np.array(testlist)
            self.valUTM = np.transpose(valarray[:,2:].astype(np.float64))
            print('number of ground images in test set', len(testlist))
            with open('Oxford_split/test_yaw.npy', 'rb') as f:
                self.val_yaw = np.load(f)
        
        # calculate the transformation from easting, northing to satellite image col, row
        # transformation for the satellite image
        primary = np.array([[619400., 5736195.],
                      [619400., 5734600.],
                      [620795., 5736195.],
                      [620795., 5734600.],
                      [620100., 5735400.]])
        secondary = np.array([[900., 900.], #tl
                    [492., 18168.], #bl
                    [15966., 1260.], #tr
                    [15553., 18528.], #br
                    [8255., 9688.]]) # c

        n = primary.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:,:-1]
        X = pad(primary)
        Y = pad(secondary)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(X, Y)

        self.transform = lambda x: unpad(np.dot(pad(x), A))

        self.trainIdList = [*range(0,self.trainNum,1)]
        self.__cur_id = 0 
        self.valIdList = [*range(0,self.valNum,1)]
        self.__cur_test_id = 0
        
   
    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.valNum:
            self.__cur_test_id = 0
            return None, None, None
        elif self.__cur_test_id + batch_size >= self.valNum:
            batch_size = self.valNum - self.__cur_test_id
        
        batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 154, 231, 3], dtype=np.float32)
        batch_gt = np.zeros([batch_size, 512, 512, 1], dtype=np.float32)
        
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # ground
            img = cv2.imread(self.grd_image_root + self.valList[img_idx][0])

            if img is None:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.valList[img_idx][0], i))
                continue
            img = cv2.resize(img, (231, 154), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img
            
            # load satellite images from the full satellite map
            image_coord = self.transform(np.array([[self.valUTM[0, img_idx], self.valUTM[1, img_idx]]]))[0] # pixel coords of the ground image. Easting, northing to image col, row
            col_split = int((image_coord[0]) // 400)
            if np.round(image_coord[0] - 400*col_split) <200:
                col_split -= 1
            col_pixel = int(np.round(image_coord[0] - 400*col_split))

            row_split = int((image_coord[1]) // 400)
            if np.round(image_coord[1] - 400*row_split) <200:
                row_split -= 1
            row_pixel = int(np.round(image_coord[1] - 400*row_split))
            
            img = self.full_satellite_map[row_split*400-200:row_split*400+800+200, col_split*400-200:col_split*400+800+200, :] # read extra 200 pixels at each side to avoid blank after rotation
            rotate_angle = self.val_yaw[img_idx]/np.pi*180-90
            rot_matrix = cv2.getRotationMatrix2D((600, 600), rotate_angle, 1) # rotate satellite image
            img = cv2.warpAffine(img, rot_matrix, (1200, 1200))
            img = img[200:1000, 200:1000, :]
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img        
            
            row_offset_resized = int(-(row_pixel/800*512-256))
            col_offset_resized = int(-(col_pixel/800*512-256))

            # Gaussian GT
            x, y = np.meshgrid(np.linspace(-256+col_offset_resized,256+col_offset_resized,512), np.linspace(-256+row_offset_resized,256+row_offset_resized,512))
            d = np.sqrt(x*x+y*y)
            sigma, mu = 4, 0.0
            img = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
            rot_matrix = cv2.getRotationMatrix2D((256, 256), rotate_angle, 1) # apply the same rotation on GT heat map
            img = cv2.warpAffine(img, rot_matrix, (512, 512))
            batch_gt[i, :, :, 0] = img
        
        self.__cur_test_id += batch_size

        return batch_sat, batch_grd, batch_gt
    
    def next_pair_batch(self, batch_size):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.trainIdList)

        if self.__cur_id + batch_size > self.trainNum:
            self.__cur_id = 0
            return None, None, None
        
        batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 154, 231, 3], dtype=np.float32)
        batch_gt = np.zeros([batch_size, 512, 512, 1], dtype=np.float32)
        batch_offset = np.zeros([batch_size, 2], dtype=np.float32)
        
        i = 0
        batch_idx = 0
        
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.trainNum:
                break
        
            img_idx = self.trainIdList[self.__cur_id + i]
            i += 1

            # load ground image
            img = cv2.imread(self.grd_image_root + self.trainList[img_idx][0])

            if img is None:
                print('InputData::next_pair_batch: read fail: %s %d' % (self.grd_image_root + self.trainList[img_idx][0], i))
                continue
            
            img = cv2.resize(img, (231, 154), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[batch_idx, :, :, :] = img
                        
            # load satellite image, when loading a satellite image that covers the ground image, the groundtruth is non-zero, when loading a nearby satellite image that does not cover the ground image, the groundtruth is zero anywhere.
            image_coord = np.round(self.transform(np.array([[self.trainUTM[0, img_idx], self.trainUTM[1, img_idx]]]))[0]) # pixel coords of the ground image. Easting, northing to image col, row
#             if_cover = random.randint(0,2)

            # generate a random offset for the ground image
            alpha = 2 * math.pi * random.random()
            r = 200 * np.sqrt(2) * random.random()
            row_offset = int(r * math.cos(alpha))
            col_offset = int(r * math.sin(alpha))

            sat_coord_row = int(image_coord[1] + row_offset) # sat center location
            sat_coord_col = int(image_coord[0] + col_offset)

            # crop a satellite patch centered at the location of the ground image offseted by a randomly generated amount
            img = self.full_satellite_map[sat_coord_row-400-200:sat_coord_row+400+200, sat_coord_col-400-200:sat_coord_col+400+200, :] # load at each side extra 200 pixels to avoid blank after rotation
            rotate_angle = self.train_yaw[img_idx]/np.pi*180-90
            rot_matrix = cv2.getRotationMatrix2D((600, 600), rotate_angle, 1) 
            img = cv2.warpAffine(img, rot_matrix, (1200, 1200))
            img = img[200:1000, 200:1000, :]

            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[batch_idx, :, :, :] = img
            row_offset_resized = int(np.round((400+row_offset)/800*512-256)) # ground location + offset = sat location
            col_offset_resized = int(np.round((400+col_offset)/800*512-256))

            # Gaussian GT
            x, y = np.meshgrid(np.linspace(-256+col_offset_resized,256+col_offset_resized,512), np.linspace(-256+row_offset_resized,256+row_offset_resized,512))
            d = np.sqrt(x*x+y*y)
            sigma, mu = 4, 0.0
            img = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
            rot_matrix = cv2.getRotationMatrix2D((256, 256), rotate_angle, 1) 
            img = cv2.warpAffine(img, rot_matrix, (512, 512))
            batch_gt[batch_idx, :, :, 0] = img
            
            batch_idx += 1
        self.__cur_id += i
        
        return batch_sat, batch_grd, batch_gt
    
    def reset_scan(self):
        self.__cur_test_id = 0
    