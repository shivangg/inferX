# Abstract

<!-- In the abstract, student gives a high level overview of what is being attempted in the report. Abstracts are typically 5-10 sentences that provide just enough context to understand the gist of the report. -->

In this project, a network will be trained for classifying real world objects into predefined classes.
In addition to training a network on the supplied dataset, a different network will be chosen and trained using self acquired data. The quality and quantity of data acquired will be discussed. 

The network would be able to successfully classify the objects in the self collected data into 4 classes with reasonable accuracy.

# Introduction

<!-- The introduction should provide some material regarding the history of the problem, why it is important and what is intended to be achieved. If there exists any previous attempts to solve this problem, this is a great place to note these while conveying the differences in your approach (if any). The intent is to provide enough information for the reader to understand why this problem is interesting and set up the conversation for the solution you have provided. Use this space to introduce your robotic inference idea and how you wish to apply it. If you have any papers sites you have referenced for your idea, please make sure to cite them.
 -->

The classification of objects in the 2D image had been tried to be solved by a lot of classical methods but deep learning was able to become the new State of the Art technology. Deep learning shines when there is a lot of data.

The advantages of deep learning :

1. Scales well with increase in data.
2. No in-depth feature detection rules of the data needs to be hard coded.
3. Adaptable and transferable to the new problem datasets.

In the defense of the classical algorithms[[1]], classical algorithms are better than deep learning in some cases:

1. Needs no training time.
2. Financially and computationally cheap to run.
3. Easier to implement: The parameter tuning in the case of the classical algorithms is more straight-forward because of the thorough understanding of the inner working of them. In case of the deep learning, even the researchers do not understand the inner working of the DL network(_almost a blackbox_).

|![Deep Learning Vs Classical ML Algorithms]|
|:-----------------------------------------:|
|_Image Source: https://towardsdatascience.com/deep-learning-vs-classical-machine-learning-9a42c6d48aa_ |


To investigate the effectiveness of the inference models with random everyday objects that are used extensively by the humans, it was decided to classify the everyday objects on the work desk for this inference task.

Classes - **Wallet**, **Mouse**, **NailCutter**, **Nothing**

All the inference tasks essentially have similar steps that need to be followed. **Nvidia DIGITS is a system to smoothen this workflow.** This project will employ the Nvidia DIGITS workflow for developing the inference system.

Nvidia DIGITS is a way to quickly:

* **Build a dataset** that the network can take as input .
* **Define, customize and visualize** a network architecture.
* **Train the network** with the dataset.

Nvidia DIGITS provides a cleaner interface to do all of the above, through the web browser. 

The common inference tasks are: 

1. Classification: Answers the _what_ in the image
2. Detection: Answers the _where_ in the image
3. Segmentation: Pixel level labeling of the images.


# Formulation

<!-- At the background / formulation stage,
you should begin diving into the technical details of your approach by explaining to the reader 
how hyperparameters for training the network were defined,
what type of network was chosen,
and the reasons these items were performed.
This should be factual and authoritative, meaning you should not use language such as I think this will work or Maybe a network with this architecture is better... Instead, focus on items similar to, A 3-layer network architecture was chosen with X, Y, and Z parameters Explain why you chose the network you did for the supplied data set and then why you chose the network used for your robotic inference project. -->

### Transfer Learning 

In practice, it is very rare that a CovNet is trained from scratch (with random initialization) because of the unavailability of the vast amount of dataset and compute power needed.

> Andrew Ng once explained neural network as it being like a rocket ship. Rocket ship needs a powerful engine and lot of fuel. Powerful engine but less fuel fails to make it to the orbit. Weak engine but lot of fuel fails to even liftoff. Analogously, DL models are engines and dataset is the fuel.


Transfer learning is the finetuning of a network  pretrained on a large dataset of the problem similar to the original problem.

In the project transfer learning was leveraged. A GoogLeNet that was pretrained on the ImageNet(1000 Image classes) was used for classifying the everyday objects in the self acquired data.

The [GoogLeNet] was used because it has 22 layers and 12 times less parameters than the AlexNet because of:

1. The absence of the fully connected later at the end. 
2. Use of the concept of a **local mini network** inside a bigger network called the **Inference module**. The inference module played the major role in the parameter reduction. This solved the problem of deciding which convolution to use - 1x1, 3x3, 5x5, 7x7. It enabled the network to use **all of the convolutions concatenated depth wise**.

The naive **Inference module** used the depth wise concatenation of the `1x1`, `3x3`, `5x5 convolutional layers` along with a `3x3 max pooling`. The max pooling layer led to the depth staying the same or increasing in the output layer of the Inference module. This problem was solved by 1x1 convolutional layer with the 3x3, 5x5 and more importantly the max pooling layer. The number of 1x1 convolutional layer **helped reduce the dimensionality of the output layer** because the depth of output layer will just be the `number` of 1x1 convolutional layers.

|![Inception module] |
|:------------------:|
|_Image Source: Inception module from “Going Deeper with Convolutions.”_ |



GoogLeNet was faster and more accurate than the AlexNet. **The idea around the development of GoogLeNet was that it should be able to run even in the smartphone.**(_calculation budget around 1.5 billion multiply-adds on prediction_)

Also, SGD performed better than the RMSProp and was used in the finetuning of the network.



# Data Acquisition

<!-- The Data Acquisition section should discuss the data set. Items to include are 
the number of images,
size of the images,
the types of images (RGB, Grayscale, etc.),
how these images were collected (including the method). 

Providing this information is critical if anyone would like to replicate your results. After all, the intent of reports such as these are to convey information and build upon ideas so you want to ensure others can validate your process. 

Justifying why you gathered data in this way is a helpful point, but sometimes this may be omitted here if the problem has been stated clearly in the introduction. It is a great idea here to have at least one or two images showing what your data looks like for the reader to visualize. -->

### Number of images
#### Number of Training Images

Mouse |Wallet	| Nail Cutter |Nothing
:----:|:-----:|:-----------:|:-------:
317 	|   216 |64				    |25

**Training Images Distribution**: This distribution of images for each class was used for training the network.

|![number of Training images]|
|:--------------------------:|
| Number of Training Image in Each Class  |



#### Number of Test Images

Mouse |Wallet	| Nail Cutter |Nothing
:----:|:-----:|:-----------:|:-------:
105 	|   72  |21				    |8

**Test Images Distribution**: This distribution of images was used for the validation of the network. The network validates its accuracy after each run from these images.

|![number of test images]|
|:--------------------------:|
| Number of Training Image in Each Class  |

The collected image data was augmented by using the rotation of:
* 90 degrees clockwise
* 90 degrees counter- clockwise
* 180 degrees

As a result, the number of images in the dataset quadrupled.

As it can be seen here, the number of images for the `Wallet` and `Mouse` class is much more than the `Nail Cutter`. This will be explained in detail under the [Discussion](#Discussion) section.

### Size of images

The images were captured after setting **image aspect ratio	to 1:1** so that the images are square in shape. This is because the network that we will be training on accepts `256x256 images`. If the images are square, they can be easily resized to 256 pixels maintaining the aspect ratio. Had the aspect ratio been different from 1:1, it would lead to _the unnecessary cropping or squashing_ of the image data causing loss of important data.

Also, each such square image had the file size of `7 - 13 kB`. 

### Image Color Type

Real World objects are in color and **the grayscale conversions of them will conflict with each other leading to 2 objects looking the same**. Thus, the RGB images of the objects was captured and a network that can handle the RGB(_GoogLeNet_) was used to successfully classify them.


### Image Collection Method

The smartphone camera was used for the collection of the images because:

1. The same camera would be used for the test of the classification accuracy.
2. Lending to the ubiquitous nature of the smartphone cameras, the trained network **will have better probability of successfully classifying the objects** in a different condition. This is because atleast the testing and training data capture device stayed the same, i.e, smartphone camera.

It should be duly noted that the **environment conditions** in which the classification will need to successfully work **should overlap** with the environment in which the training image would be captured.

For example, if the classification needs to work _in a lighted room with diffused sunlight from the window_ then the data acquisition should be done in the _same environment_.


Mouse  					|  Wallet			   |   Nail Cutter  					 |			 Nothing
:--------------:|:--------------:|:-------------------------:|:------------------:
![Mouse Image]  |![wallet Image] | ![Nail Cutter Image]			 |  ![Nothing Image]  

As evident, the images:

1. Have an aspect ratio of 1:1
2. Diffused light that lead to shiny surface

The test conditions will be similar to the above features.

In the supplied data, the images were also square.

|      Supplied Data 		|
|-----------------------|
| ![Supplied Data View] |





# Results 

<!-- Results part is typically the hardest part of the report for many. You want to convey your results in an unbiased fashion. If you results are good, you can objectively note this. Similarly, you may do this if they are bad as well. You do not want to justify your results here with discussion; this is a topic for the next session. Present the results of your robotics project model and the model you used for the supplied data with the appropriate accuracy and inference time For demonstrating your results, it is incredibly useful to have some charts, tables, and/or graphs for the reader to review. This makes ingesting the information quicker and easier.
 -->

#### Model for classifying supplied dataset

Used GoogLeNet with SGD with the supplied data that achieved :

1. The accuracy of 75.4098%
2. Inference time of 5.11062 ms using TensorRT


| ![Supplied data Accuracy] | 
|:--------------------------:|
| **The accuracy and inference time of the network trained on the supplied data** |

The caffe model trained on the supplied data can be found [here](suppliedDataModel).

The five millisecond inference time is good enough to be deployed for real time classification for use in the robotics workbench.

Candy Box       |  Bottle
:--------------:|:--------------:
![Candy Image]  |![Bottle Image] 


It should be noted that when training the GoogLeNet with the ADAM, the accuracy dropped to 67.213%.

| ![Supplied data less Accuracy ADAM] | 
|:--------------------------:|
| **The accuracy of the network decreased when trained with ADAM instead of SGD** |

#### Model for classifying self acquired dataset
In the [custom model], the following important features can be seen as a function of epochs:

* Validation Loss
* Validation Accuracy
* Training Loss
* Training Accuracy

| ![Custom Model Accuracy Graph ] | 
|:--------------------------:|
| **Loss and Accuracy of Train and Validation data with with SGD** |

The custom model was also tested on separate images the result which the network was able to successfully classify with good accuracy.

| **Separately Tested on Different Images** |
|:-----------------------------------:|
| ![Custom Model Test Mouse ] | 
| ![Custom Model Test NailCutter ] | 
| ![Custom Model Test Nothing ] | 
| ![Custom Model Test Wallet ] | 


As evident from the separate tested images, the finetuned GoogLeNet was able to successfully classify the objects in the problem set with good accuracy.

### Visualizations

As an advantage of using **Nvidia DIGITS**, the network can be easily visualized. The  weights and activations from the pooling layers, convolutional layers, Inception layer can be easily visualized.

| ![Input nailcutter] |  ![Inception layer Output Activations]     |
|:--------------------------:|:-----------------------:|
| **Input image of nail cutter** | **Inception layer(_near final layers_) Output Activations** | 

| ![Conv layer activations] | 
|:--------------------------:|
| **Output Activation from Convolutional layer near input layer** |



# Discussion 

<!-- Discussion is the only section of the report where you may include your opinion. Make sure your opinion is based on facts, in this case, your results.
If your results are poor, make mention of what may be the underlying issues. 
If the results are good, why do you think this is the case?
Also, reflect on which is more important, inference time or accuracy, in regards to your robotic inference project.
Again, avoid writing in the first person (i.e. Do not use words like I or me).If you really find yourself struggling to avoid the word I or me; sometimes, this can be avoid with the use of the word one. As an example: instead of, "I think the accuracy on my dataset is low because the images are too small to show the necessary detail" try, "one may believe the accuracy on the dataset is low because the images are too small to show the necessary detail". They say the same thing, but the second avoids the first person. -->


It was interesting to see the classification accuracy improve over the epochs as the network was finetuned by SGD.

| ![Difficult Classification] | 
|:--------------------------:|
| **performance of a snapshotted earlier epoch less tuned network having a hard time classifying** |

When training was started with only 25 images in each class to test the effect of less data on the network, the network was able to classify the **Nothing** class but could not differentiate between the `Wallet` and `Mouse`. This was because the objects `Wallet` and `Mouse` were almost identical in terms of:

* Color
* Surface shine

This is the reason more data had to be collected for these 2 classes to 
As it can be seen from the earlier epochs snapshot of the model, there is more problem classifying between `Wallet` and `Mouse` than the `nailCutter`. As the epochs increased, this problem was minimized as the network got better at classification by learning the underlying patterns in the image.

### Accuracy vs Inference Time
 As the accuracy increase,

1. The number parameters required increases leading to a bigger sized network.
2. More parameters mean more time to train the network.
3. Bigger network will be difficult to deploy on the embedded system which comes with less compute power.

But the higher accuracy means better classification. This can be a dependable and reliable solution solution that can handles wider spectrum of data.

To reduce the inference time, the network needs to be smaller, i.e, less number of network parameters.

1. This narrows the learning capability of the network.

Therefore, the trade-off between accuracy and inference time needs to be made. This should be highly dependent on the task at hand. 

If a general solution is needed that can infer for the wide range of data inputs with the network running with good compute power on a work station, accuracy can be increased.

If a very task specific solution is needed, that needs to work fast on the embedded system with limited compute power in real time, the inference time should be reduced to be in order of milliseconds. 


# Conclusion

<!-- The Conclusion / Future Work section is intended to summarize your report. Your summary should include a recap of the results, did this project achieve what you attempted, and is this a commercially viable product? For Future work, address areas of work that you may not have addressed in your report as possible next steps. For future work, this could be due to time constraints, lack of currently developed methods / technology, and areas of application outside of your current implementation. Again, avoid the use of the first-person. -->

The project good results in the supplied as well as the custom self acquired data. This was essentially a result of:

* Quality data collection of the objects.
* Smooth and fast development of ideas using the Nvidia DIGITS workflow.
* And most importantly, **transfer learning** which enabled one with limited data and compute power to build a good classification network. 

This inference model can be deployed on a small robotic arm that can manipulate everyday objects on the table. The problems with this approach is that:

1. Since no depth information is available, it will be difficult to manipulate the table objects as the trajectory calculations depend on the final pose to be reached.
2. Finer manipulations will require the computation the grasping pose that is still a open area of research.


### Future Work


In the future, rather than using a classification network, a detection network can be used to output the object coordinates in the image. Although this would require each image in the dataset to be annotated with a bounding box.

Also, it will be interesting to see the performance of such networks with the point cloud data to LIDARs, stereographic and RGB-D camera.


<!-- LINKS -->

[Custom Model Accuracy Graph ]: images/customData/4ClassesNetwork/TrainingGraph.png
[number of test images]: images/customData/4ClassesNetwork/TestM105W72N21Cutter8.png
[number of Training images]: images/customData/4ClassesNetwork/TrainM317W216N64Cutter25.png
[Custom Model Test Mouse ]: images/customData/4ClassesNetwork/mouse@50.png
[Custom Model Test NailCutter ]: images/customData/4ClassesNetwork/nailCutter@50.png
[Custom Model Test Nothing ]: images/customData/4ClassesNetwork/nothing@50.png
[Custom Model Test Wallet ]: images/customData/4ClassesNetwork/wallet@50.png
[Difficult Classification]: images/customData/4ClassesNetwork/DiscussWallet.png

[Candy Image]: images/suppliedData/SuppliedDataClassificationCandy.png
[Bottle Image]: images/suppliedData/SuppliedDataClassifiedBottle.png
[Supplied Data View]: images/suppliedData/suppliedExploreData.png
[Supplied data Accuracy]: images/suppliedData/[IMP]suppliedAccuracyShot.png
[Supplied data less Accuracy ADAM]: images/suppliedData/[imp]CouldNotsolveWithADAM.png

[Mouse Image]: images/customData/egImage.jpg
[wallet Image]: images/customData/egImage2.jpg
[Nail Cutter Image]: images/customData/egImage3.jpg
[Nothing Image]: images/customData/egImage4.jpg
[Deep Learning Vs Classical ML Algorithms]: images/customData/dlVsClassical.png
[Inception module]: images/customData/inception.png

[Input nailcutter]: images/customData/visualDIGITS/nailCutterInput.png
[Inception layer Output Activations]: images/customData/visualDIGITS/nailCutterInceptionVisualActivations.png
[Conv layer activations]: images/customData/visualDIGITS/convolutonActivationsNearInput.png

[custom model]: customDataModel
[1]: https://arxiv.org/pdf/1802.00036
[GoogLeNet]: https://arxiv.org/pdf/1409.4842.pdf