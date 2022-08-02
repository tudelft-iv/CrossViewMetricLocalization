import numpy as np
import os
from robotcar_dataset_sdk.python.image import load_image
import imageio

path_to_OxfordRobotCar = './OxfordRobotCar/'

print('pre-process training images')
dates = ['2015-06-26-08-09-43', '2015-08-28-09-50-22', '2015-03-03-11-31-36', '2014-12-12-10-45-15', '2015-10-30-13-52-14', 
        '2015-04-24-08-15-07', '2015-11-12-13-27-51', '2015-08-17-13-30-19', '2015-03-17-11-08-44', '2015-11-10-11-55-47', 
        '2014-11-28-12-07-13']

for date in dates:
    robotcar_filenames = []
    ground_filenames = []
    with open('Oxford_split/lookup_test.txt') as file: 
        for line in file:
            robotcar, ground = line.split(' ')
            ground, _ = ground.split('\n')
            robotcar_filenames.append(robotcar)
            ground_filenames.append(ground)
    
    if not os.path.exists(path_to_OxfordRobotCar+date+'/ground'):
        os.makedirs(path_to_OxfordRobotCar+date+'/ground')
        
    for i in range(len(robotcar_filenames)):
        img_raw = load_image(path_to_OxfordRobotCar+robotcar_filenames[i])
        
        img_cropped = img_raw[0:800,40:1240,:]
        imageio.imwrite(path_to_OxfordRobotCar+ground_filenames[i], img_cropped, '.png')
            
print('pre-process validation images')            
dates = ['2015-08-17-10-42-18']
for date in dates:
    robotcar_filenames = []
    ground_filenames = []
    with open('Oxford_split/lookup_test.txt') as file: 
        for line in file:
            robotcar, ground = line.split(' ')
            ground, _ = ground.split('\n')
            robotcar_filenames.append(robotcar)
            ground_filenames.append(ground)
    
    if not os.path.exists(path_to_OxfordRobotCar+date+'/ground'):
        os.makedirs(path_to_OxfordRobotCar+date+'/ground')
        
    for i in range(len(robotcar_filenames)):
        img_raw = load_image(path_to_OxfordRobotCar+robotcar_filenames[i])
        
        img_cropped = img_raw[0:800,40:1240,:]
        imageio.imwrite(path_to_OxfordRobotCar+ground_filenames[i], img_cropped, '.png')


print('pre-process test images')
dates = ['2015-08-14-14-54-57', '2015-08-12-15-04-18', '2015-02-10-11-58-05']
for date in dates:
    robotcar_filenames = []
    ground_filenames = []
    with open('Oxford_split/lookup_test.txt') as file: 
        for line in file:
            robotcar, ground = line.split(' ')
            ground, _ = ground.split('\n')
            robotcar_filenames.append(robotcar)
            ground_filenames.append(ground)
    
    if not os.path.exists(path_to_OxfordRobotCar+date+'/ground'):
        os.makedirs(path_to_OxfordRobotCar+date+'/ground')
        
    for i in range(len(robotcar_filenames)):
        img_raw = load_image(path_to_OxfordRobotCar+robotcar_filenames[i])
        
        img_cropped = img_raw[0:800,40:1240,:]
        imageio.imwrite(path_to_OxfordRobotCar+ground_filenames[i], img_cropped, '.png')