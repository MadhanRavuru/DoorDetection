# YOLO with Custom Dataset in PyTorch

This article shows how to train on custom dataset
https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9

# Training for Door Detection in floor plans

Detecting single class (door) using YOLO Object detection framework. The project uses yolov3 for detection

High quaity architectural images from [Cubicasa5k](https://zenodo.org/record/2613548#.XtDCHMYzZuQ) dataset is considered for training. Further challenging images were selected by googling to generate more robust model for challenging plans.

In the *data/DoorDetection* folder, we have the training images along with annotated labels generated using *bbox.py*. As the dataset size is small, we have augmented the images. Even after augmentation, training set has around only 1374 images.

The initial weights for model are the weights from coco dataset with 80 classes.

Train using the *Train.ipynb* notebook to get weights for our model. *checkpoints* folder will be created with weights stored for certain epochs. Place them in *config* folder with name *yolov3.weights*.

I had to train for with 1374 images for 100 epochs, batch size of 8, and learning rate as 0.0001
Then, using 90.weights from checkpoints folder, the model is further trained with different setting. Now, learning rate is 0.0005

**Observation**
The model detects door in simple floor plans effectively. For more challenging plans, we split image intosmaller merges, perform detection and merge as in staircase detection.

The model basically learns to detect arcs in images, which are nothing but doors. For complex floor plans, we get lot of false positives and false negatives. This means the model is not robust. It is hard for model to identify doors in challenging plans
