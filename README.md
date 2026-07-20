## Introduction
The fundus of the eye is the interior surface of the eye opposite the lens and includes the retina, optic disc, macula, fovea, and posterior pole. The fundus can be examined by fundus photography. Medical signs that can be detected from observation of eye fundus for any lesions. It would be beneficial for us if we could have a machine-learning model that can inform us about possible conditions of the eyes when supplying it with fundus images. For this project, we have obtained fundus images from [Kaggle]( https://www.kaggle.com/datasets/manan1717/ocular-disease-dataset) to work with. We will leverage Convolutional Neural Network (CNN) to build our model because it is very efficient in extracting features of increasing complexity and abstraction and has become the backbone of most modern image recognition systems. In addition, CBAM (Convolutional Block Attention Module) will be integrated into our CNN to enhance the representational power of the network by focusing on important features along both the spatial and channel dimensions.  
<br>

## Objective
Our objective is to build a Convolutional Neural Network (CNN) model that can classify out of the given six common ocular diseases and normal fundus.  
<br>

## Labels and Class Distribution
Disease Class || Image Count  
Age-related Macular Degeneration (A) || 511  
Cataract (C) || 1,008  
Diabetic Retinopathy (D) || 3,128  
Glaucoma (G) || 1,646  
Hypertensive Retinopathy (H) || 420  
Myopia (M) || 739  
Normal (N) || 2,997  
Total || 10,449  
<br>

## Platform and Software Versions
Python Version: 3.12.13 | packaged by Anaconda, Inc.  
PyTorch Version:  2.13.0+cu130  
GPU: NVIDIA GeForce RTX 4070  
CUDA UMD Version: 13.3  
<br>


## CNN Model Summary
<br>

````markdown
===============================================================================================  
Layer (type:depth-idx)                        Output Shape              Param #  
===============================================================================================  
CNNet                                         [1, 7]                    --  
├─Sequential: 1-1                             [1, 7]                    --  
│    └─Conv2d: 2-1                            [1, 32, 256, 256]         896  
│    └─BatchNorm2d: 2-2                       [1, 32, 256, 256]         64  
│    └─ReLU: 2-3                              [1, 32, 256, 256]         --  
│    └─MaxPool2d: 2-4                         [1, 32, 128, 128]         --  
│    └─Conv2d: 2-5                            [1, 64, 128, 128]         18,496  
│    └─BatchNorm2d: 2-6                       [1, 64, 128, 128]         128  
│    └─ReLU: 2-7                              [1, 64, 128, 128]         --  
│    └─MaxPool2d: 2-8                         [1, 64, 64, 64]           --  
│    └─Conv2d: 2-9                            [1, 128, 64, 64]          73,856  
│    └─BatchNorm2d: 2-10                      [1, 128, 64, 64]          256  
│    └─ReLU: 2-11                             [1, 128, 64, 64]          --  
│    └─MaxPool2d: 2-12                        [1, 128, 32, 32]          --  
│    └─Conv2d: 2-13                           [1, 256, 32, 32]          295,168  
│    └─BatchNorm2d: 2-14                      [1, 256, 32, 32]          512  
│    └─ReLU: 2-15                             [1, 256, 32, 32]          --  
│    └─MaxPool2d: 2-16                        [1, 256, 16, 16]          --  
│    └─Conv2d: 2-17                           [1, 512, 16, 16]          1,180,160  
│    └─BatchNorm2d: 2-18                      [1, 512, 16, 16]          1,024  
│    └─ReLU: 2-19                             [1, 512, 16, 16]          --  
│    └─CBAM: 2-20                             [1, 512, 16, 16]          --  
│    │    └─ChannelAttention: 3-1             [1, 512, 1, 1]            32,768  
│    │    └─SpatialAttention: 3-2             [1, 1, 16, 16]            98  
│    └─MaxPool2d: 2-21                        [1, 512, 8, 8]            --  
│    └─Conv2d: 2-22                           [1, 512, 8, 8]            2,359,808  
│    └─BatchNorm2d: 2-23                      [1, 512, 8, 8]            1,024  
│    └─ReLU: 2-24                             [1, 512, 8, 8]            --  
│    └─CBAM: 2-25                             [1, 512, 8, 8]            --  
│    │    └─ChannelAttention: 3-3             [1, 512, 1, 1]            32,768  
│    │    └─SpatialAttention: 3-4             [1, 1, 8, 8]              98  
│    └─AdaptiveAvgPool2d: 2-26                [1, 512, 2, 2]            --  
│    └─Flatten: 2-27                          [1, 2048]                 --  
│    └─Linear: 2-28                           [1, 256]                  524,544  
│    └─ReLU: 2-29                             [1, 256]                  --  
│    └─Dropout: 2-30                          [1, 256]                  --  
│    └─Linear: 2-31                           [1, 128]                  32,896  
│    └─ReLU: 2-32                             [1, 128]                  --  
│    └─Dropout: 2-33                          [1, 128]                  --  
│    └─Linear: 2-34                           [1, 7]                    903  
===============================================================================================  
| Metric | Value |  
|--------|------:|  
| Total Parameters | 4,555,467 |  
| Trainable Parameters | 4,555,467 |  
| Non-trainable Parameters | 0 |  
| Total Mult-Adds | 1.42 G |  
| Input Size | 0.79 MB |  
| Forward/Backward Pass Size | 65.56 MB |  
| Parameters Size | 18.22 MB |  
| Estimated Total Size | 84.57 MB |  
===============================================================================================  
````

<br>

## Discussion

Our proposed CNN model is able to produce an overall accuracy of 77%, overall precision of 0.7803, overall recall of 0.7742, and overall F1 score of 0.7749. Meanwhile, the per-class accuracies are ranging from 63% to 99%. The highest being Class 1, Cataract (C), being 99%. Taking a look at the raw images of Class 1, we can observe that the images of this class are very distinctive visually and are easy to identify. This means the features of Class 1 were distinctive enough that our model can easily learnt. The worse class is Class 4, Hypertensive Retinopathy (H), has an accuracy of only 63%. This is due to class 4 has the smallest number of images (420, 4% of total) for the model to train on, a class imbalance issue. In this case, the neural network cannot learn too much about the features of class 4. For Class 2, Diabetic Retinopathy (D), while it has highest number of images (3,128 records), its accuracy is not among the highest ones. From the confusion matrix, we can observe that Class 2 was often mistaken as Class 6 Normal (N), or vice versa. By observing the images of Class 2 being mistaken as Class 6, we found that Class 2 images genuinely look like normal eyes. This means our model cannot efficiently differentiate Class 2 and Class 6 due to their visual similarity. 

Recap  
Per-class Accuracies  
Class 0: 0.8421  
Class 1: 0.9934  
Class 2: 0.7191  
Class 3: 0.7976  
Class 4: 0.6349  
Class 5: 0.7387  
Class 6: 0.7622  

Performance Metrics  
Overall Accuracy:&nbsp;&nbsp;&nbsp;&nbsp;0.7742  
Overall Precision:&nbsp;&nbsp;&nbsp;&nbsp;0.7803  
Overall Recall:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.7742  
Overall F1 Score:&nbsp;&nbsp;&nbsp;&nbsp;0.7749  

<br>

## Future Work

For future work, we can consider the following.

1. We can find more images for Class 4, Hypertensive Retinopathy (H), so we can solve the class imbalance issue.
2. To be able to handle Class 2 more efficiently, we can try other preprocessing methods. For example, we can try identifying the location of the lesion, then crop that region off. Next we can use the lesion only image, without resizing down too much of the resolution, to do the training. 
To sum up, we have successfully produced an effective classification model that can identify the specific ocular disease of eyes, out of the provided 6 conditions plus 1 normal, when providing the fundus images. Our model has good performance metrics where overall accuracy, precision, recall, and F1 score are all being almost 80%. There may be some minor issues when identifying certain conditions, but the overall architecture itself is good. 

