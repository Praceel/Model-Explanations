

## Report

### Model & Data

* Which model are you going to explain? What does it do? On which data is it used?

We are explaining a pretrained ResNet50 model, which is capable of classifying images. When provided with an input image, the model will output a probability distribution over the different classes of the image dataset, allowing it to make predictions about the content of the image. 

The original pretrained model has been trained on ImageNet data. For the purpose of this study, we retrained it on the Intel Image Classification dataset, which contains RGB images of size 150x150 showing different landscapes belonging to one of the following classes: buildings, forest, glacier, mountain, sea, street. The size of the training dataset was 13986 and the size of the test dataset was 2993. 

We retrained the model for a total of 50 epochs. However, since no improvement was observed during the last 4 epochs, we used the state of the model after 46 epochs for classification. For more details on the training process, see [training/image_classification_training_colab.ipynb]( training/image_classification_training_colab.ipynb).

* From where did you get the model and the data used?

We got the pretrained model from Pytorch's Torchvision module (http://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) and the data was obtained from Kaggle (https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

* Describe the model.

ResNet50 is a deep convolutional neural network that is trained on the ImageNet dataset. It is a 50-layer network, consisting of several convolutional layers, activation layers, and pooling layers with "residual connections". The residual connections, which allow the model to bypass some of the layers and skip certain transformations, are an important part of the network architecture and make it possible for the model to learn more complex and abstract features of the input data.

At a high level, the architecture of ResNet50 can be thought of as consisting of four main parts: the initial convolutional layers, the bottleneck layers, the residual layers, and the final classifier layers.

The initial convolutional layers of ResNet50 are similar to those found in many other convolutional neural networks. These layers use small filters to detect simple features in the input data, such as edges and textures. These features are then passed on to the next layers in the network, where more complex features are detected.

The bottleneck layers in ResNet50 are made up of one convolutional layer and one max-pooling layer. These layers reduce the dimensions of the input data, allowing the network to focus on the most important features.

The residual layers in ResNet50 are the key to the network's architecture. In these layers, the output of the previous layer is added to the output of the current layer, using a technique known as "shortcut connections" or "skip connections". This allows the network to learn the residual, or difference, between the previous layer's output and the desired output, making it easier for the network to learn and improve its performance.

Finally, the classifier layers of ResNet50 are fully-connected layers that use the output of the residual layers to make predictions about the input data. These predictions are made using a softmax activation function, which produces a probability distribution over the possible classes for the input data.

Overall, the architecture of ResNet50 is designed to be very deep, with many layers and residual connections, which allows the model to learn a rich set of features from the input data and achieve state-of-the-art performance on a variety of image recognition tasks.


### Explainability Approaches
Find explainability approaches for the selected models. Repeat the section below for each approach to describe it.

#### Approach 1 - Confusion Evolution

With this first approach we want to visualize the model training by depicting the class predictions for each image in the test set at each step of training, along with some metrics describing the overall difficulty of classifying a certain image. For the sake of completeness, we added plots of loss and accuracy throughout training as well as a confusion matrix.

![instance_flow.png](https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/confusion_evolution.png)

##### Confusion Evolution in Hohman et al. Classification

WHY?

By visualizing the evolution of image classification throughout training, we can get an idea about which images are easier to classify and which are particularly hard for the model to learn. This can be a useful information for improving the model. Moreover, we can get an idea about the stability of the model, which can be useful information for selecting a model. If the visualization is updated already during training, it could also be used as a decision criterion for early stopping.

WHAT?

The visualisation shows the predictions for each image in the test set at each epoch of training. Moreover, three metrics were calculated for each image based on the predictions – frequency, variability and misclassification rate. Frequency is the proportion of epochs where the prediction changed compared to the previous epoch and the variability describes how many of the 6 classes were predicted for a given image throughout the training process. For a more concise representation of the model’s behaviour during and after training, training and validations loss as well as training and test accuracy are plotted in addition to a confusion matrix of the final model.

WHEN?

While the visualisation was created after training from the predictions, loss values and accuracies recorded during training, it has the potential to be regularly updated and already used during training.

WHO?

This approach is probably most interesting for model developers, who want to improve or select the model. For a model user the training phase is likely less interesting, but the confusion matrix can give valuable insights on the model’s performance.

HOW?

During training of the model, predictions on the test dataset as well as training and validation loss and accuracy were logged. The model predictions over time are depicted as a heatmap with each row corresponding to an image, each column corresponding to a training epoch and colours representing the 6 classes. Frequencies, variabilities and misclassification rates for each image are shown as horizontal bar charts aligned to the respective rows in the heatmap. Loss and accuracy curves are illustrated as line charts and aligned with the corresponding columns in the heatmap. Finally, the confusion matrix is also visualized as a heatmap with the real labels as rows and the predictions of the final model as columns.

WHERE?

The upper part of the visualization is based on one view of the InstanceFlow tool (https://jku-vds-lab.at/publications/2020_visshort_instanceflow/).

Interpretation:

From the visualization, we can infer that up to epoch 12, while improving in terms of training loss and accuracy, the model was not generalizing well. This is reflected both in the evolution of predictions as well as in the validation loss and test accuracy. Interestingly, at first all images are classified as “buildings”. Between epochs 13 and 39 the model’s performance was highly fluctuating in terms of prediction accuracy, before it stabilized after 40 epochs. 

According to the final model’s predictions and the confusion matrix, most confusion happens between the labels “buildings” and “street” as well as “mountain” and “glacier”, which is what would intuitively be expected. While the label “buildings” has the lowest fluctuation frequency, variability and misclassification rate when looking at the entire training phase, “forest” shows the highest classification accuracy in the final model.


#### Approach 2 - Local Interpretable Model-agnostic Explanations (LIME)

Local Interpretable Model-agnostic Explanations (LIME) is model-agnostic tool for explainable AI. The algorithm highlights the superpixels in images that contribute positively or negatively to the model’s decision-making process. LIME helps us understand how complex models work by understanding how they behave on a local level. Underlying for this is the assumption that the complex model behaves linearly on a local level. More information and sources can be found in the corresponding jupyter notebook. 

In our case, the use of Lime is helpful in image classification because it can help to understand why a model made a certain prediction, which can be useful for debugging and improving the model. Another advantage of Lime is that it can be used to generate explanations that are easier for humans to comprehend.
For example, in our misclassification example (building was detected as sea), it seems to have played a role that the lower part of the image looks rather homogeneous, comparable to water. In addition, the image focus is aligned with the horizon, so that the edge from the sky to the building was probably interpreted as a coast. The marked pillar could also possibly have been recognized as a coastal component. Finally, according to the highlighting, the ceiling of the building also played a role; this is again homogeneous, so that it may have been interpreted as the sky.

![LimeWrongPred](https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/dde60e25038927746978b91f26a632e83986d76e/LimeWrongPred.png
)
##### LIME in Hohman et al. Classification

WHY?

to understand which areas of an image were crucial for a certain prediction

WHO?

model users, model developers & builders 

WHAT?

highlight superpixels in images that contribute positively or negatively to the model’s decision-making process

HOW?

Instance-based analysis & exploration with LIME

WHEN?

After training 

WHERE?

This approach has been adressed in the following paper: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).




#### Approach 3 - Grad-CAM approach

Gradient-weighted Class Activation Mapping



Grad-CAM method provides us with a way to look into what particular parts of the image influenced the whole model’s decision for a specifically assigned label. It starts with finding the gradient of the most dominant logit with respect to the latest activation map in the model. We can interpret this as some encoded features that ended up activated in the final activation map persuaded the model as a whole to choose that particular logit (subsequently the corresponding class). The gradients are then pooled channel-wise, and the activation channels are weighted with the corresponding gradients, yielding the collection of weighted activation channels. By inspecting these channels, we can tell which ones played the most significant role in the decision of the class.

##### Grad-Cam in Hohman et al. Classification

WHY?

To understand what model has learned as significant features in images for each class classification.

WHO?

Mostly beneficial for model users and developers

WHAT?

The output and gradient of last activation layer of convolutional layers with respect to input image.

HOW?

Grad-Cam approach. Weighting the activations of last layer by its pooled gradients, creating a heatmap from these weighted activations and superimposing the heatmap on the input image.

WHEN?
    
After training in a trained model.     
   
WHERE? 

This approach is based on paper  Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra; Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618-626
“Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization ” (https://arxiv.org/pdf/1610.02391.pdf) and was inspired from blogpost https://towardsdatascience.com/understand-your-algorithm-with-grad-cam-d3b62fce353 .

To analyze which portion of an image is used by model classify images into six labels (buildings, forest, glacier, mountain, sea, street) 

From the superimposed images of model weighted activations heatmap in input images we can have the idea of which pixels or patterns in images were observed by the model to label them in a specific class with high probability. It is more intituitive when we look at misclassified and correctly classified images. For example: Looking some examples of sea class we can observe that model looks for blue sky to label it as sea. In one of the image of building it classify the image as sea cause the image has a huge portion of blue sky which model sees and determine the image to be sea

###### Some examples of correct predictions and their heatmap 


<img src="https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/g1.JPG " height="450" width="450"/>

<img src="https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/G2.JPG " height="225" width="450"/>

<img src="https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/G3.JPG " height="225" width="450"/>


###### Some examples of wrong predictions and their heatmap 


<img src="https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/g4.JPG " height="450" width="450"/>

<img src="https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/g5.JPG " height="225" width="450"/>


#### Approach 4 - Saliency Maps (Vanilla)

A saliency map is a visual explanation of a neural network's prediction. It shows which parts of the input image were most influential for the network in making its prediction. This can be useful for understanding how a neural network works and for identifying potential problems or errors in its predictions. This approach is based on the idea that the network is looking for certain patterns in the input image, and the saliency map shows which parts of the image contain those patterns.

Saliency maps use a technique called backpropagation to compute the gradient of the target class with respect to the input image, which gives a measure of how sensitive the model's prediction is to changes in each pixel of the input. This gradient is then used to generate a heatmap highlighting the most important regions of the image.

To create a saliency map for ResNet50, we would first feed an input image into the network and record the network's prediction. Then, we would compute the gradient of the network's prediction with respect to the input image. This gradient tells us how much the prediction changes when we make small changes to the input image. By visualizing the gradient, we can see which parts of the input image had the greatest influence on the network's prediction.

Saliency maps are useful because they can help us understand how a neural network works and why it makes the predictions it does. They can also help us identify potential errors or problems in the network's behavior. For example, if the saliency map shows that the network is only paying attention to a small part of the input image, it could indicate that the network is not considering all of the relevant information when making its prediction.

##### Saliency Maps in Hohman et al. Classification

WHY?

To interpret representations learned by a model (ResNet50) and to get better insighst about the model

WHO?

Model Users and model developers

WHAT?

Visualize the pixels in the input image that leads to the model's prediction.

HOW?

Using Saliency Maps, which calcualtes the gradient of the traget class with respect to the input image

WHEN?

After Training phase

WHERE?

To analyze which parts of the input contribute to the correct predictions
This approach is based on the paper
Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, Simonyan et.al., 2013)
https://arxiv.org/abs/1312.6034

For instance, the Saliency Map for the correct predictions for two sample classes are

![forest-forest](https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/forest-forest.png)

![mountain-mountain](https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/mountain-mountain.png)

And the Saliency Map for the incorrect predictions for the two sample classes are

![buildings-street](https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/buildings-street.png)

![mountain-glacier](https://github.com/jku-icg-classroom/assignment-2-model-explanations-hakuna-matata/blob/main/figures/mountain-glacier.png))

From the results, we can argue that when the model makes a correct prediction, the saliency map will show which parts of the image were most important for the model in making that prediction. In case of Mountain as a correct prediction, the landscape elevation and most importantly the peak seems to play an important role. This can help us understand which features of the image the model is focusing on, and how it is using those features to make its decision.

And when the model makes an incorrect prediction, the saliency map can help us understand why the model may have made that mistake. By looking at the areas of the image that the model is focusing on, we can potentially identify any biases or problems with the model that may have led to the incorrect prediction. In case of Mountain wrongly predicted as Glacier, the slopy part of the mountain covered with snow seems to be confusing for the model with the similar features of the Glacier. This can help us improve the model and reduce the number of incorrect predictions it makes in the future.


### Summary of Approaches
Write a brief summary reflecting on all approaches.

Saliency maps, Grad CAM, LIME, and Confusion Evolution are all explainability approaches that are used to understand the behavior of machine learning models. These methods are all designed to provide insights into the decision-making process of the model, and can be useful for interpreting the predictions made by the model.

One important difference between these approaches is the level of interpretability they provide. Saliency maps and Grad CAM are visualization-based methods, and provide interpretable explanations in the form of heatmaps or saliency maps that highlight the importance of different input features. LIME, on the other hand, provides interpretable explanations in the form of simple interpretable models that approximate the behavior of the complex model locally. Confusion Evoluation provides a more detailed level of interpretability, by allowing users to study the flow of individual data points through the model training.

Another difference between the approaches shown is their scope of application. Saliency Maps and Grad-CAM are suitable for convolutional neural networks (CNNs). LIME, on the other hand, is a model-independent method, i.e. it can be combined with any type of model. Similarly, Confusion Flow is also model-agnostic and provides a detailed level of interpretability.
