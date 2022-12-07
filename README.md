# ECCV2022: Visual Cross-View Metric Localization with Dense Uncertainty Estimates

![](figures/network_architecture.png)

### Abstract
This work addresses visual cross-view metric localization for outdoor robotics.
Given a ground-level color image and a satellite patch that contains the local surroundings, the task is to identify the location of the ground camera within the satellite patch.
Related work addressed this task for range-sensors (LiDAR, Radar), but for vision, only as a secondary regression step after an initial cross-view image retrieval step.
Since the local satellite patch could also be retrieved through any rough localization prior (e.g. from GPS/GNSS, temporal filtering), we drop the image retrieval objective and focus on the metric localization only.
We devise a novel network architecture with denser satellite descriptors, similarity matching at the bottleneck (rather than at the output as in image retrieval), and a dense spatial distribution as output to capture multi-modal localization ambiguities.
We compare against a state-of-the-art regression baseline that uses global image descriptors. 
Quantitative and qualitative experimental results on the recently proposed VIGOR and the Oxford RobotCar datasets validate our design.
The produced probabilities are correlated with localization accuracy,
and can even be used to roughly estimate the ground camera's heading when its orientation is unknown.
Overall, our method reduces the median metric localization error by 51\%, 37\%, and 28\% compared to the state-of-the-art when generalizing respectively in the same area, across areas, and across time. 

### Environment
We use TensorFlow 1.14. A conda environment.yml file is included for reference. 

### Models
Our trained models can be find at: https://surfdrive.surf.nl/files/index.php/s/hcv8U9TzbfpX3lk

### Datasets
VIGOR: we download the VIGOR dataset from https://github.com/Jeff-Zilence/VIGOR <br />

Oxford RobotCar cross-view matching: please download the (Bumblebee XB3, stereo, center) ground images from the official Oxford RobotCar: https://robotcar-dataset.robots.ox.ac.uk/datasets/ <br />
The original ground images are taken at a very high framerate. For training, validation, and testing, we provide `data_preprocessing_Oxford.py` to select the used ground images, remove the distorted area and vehicle bonnet around image borders, and save images with their UTM coordinates. <br />
We stitched satellite patches to build a large satellite map that covers the whole area. Our stitched image can be found at https://surfdrive.surf.nl/files/index.php/s/2U0GsLiDbWrBlwr <br />
The code for stitching satellite images can be found in `stitching_satellite_patches.ipynb` <br />
If you need the original satellite patches used for image stitching, please send an email with your name and affiliation [to me](mailto:z.xia@tudelft.nl) <br />
In `readdata_Oxford.py`, we provide code to convert the pixel coordinates of the stitched satellite image to UTM coordinates.


### Training and evaluation
Training on VIGOR dataset: <br />
samearea split: `python train_VIGOR.py -a same` <br />
crossarea split: `python train_VIGOR.py -a cross` <br />
Testing on VIGOR dataset: <br />
samearea split: `python test_VIGOR.py -a same` <br />
crossarea split: `python test_VIGOR.py -a cross`<br />

Training on Oxford RobotCar dataset: <br />
`python train_Oxford.py` <br />
Testing on Oxford RobotCar dataset: <br />
`python test_Oxford.py` <br />


### Citations
```
@inproceedings{xia2022visual,
  title={Visual Cross-View Metric Localization with Dense Uncertainty Estimates},
  author={Xia, Zimin and Booij, Olaf and Manfredi, Marco and Kooij, Julian FP},
  booktitle={European Conference on Computer Vision},
  pages={90--106},
  year={2022},
  organization={Springer}
}
```
