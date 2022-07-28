# Fashion-Products-Recommendation-System

Online shopping websites always suggest items similar to the ones you have browsed and sometimes even items that can go with it. This notebook is a basic replication of the product recommendations using image similarity. This topic falls majorly in the domain of Recommendation sytems, and I have a whole notebook dedicated to the topic.

![](https://1030z2bnst92zo6j523feq9e-wpengine.netdna-ssl.com/wp-content/uploads/2020/02/product-recommendations-hero.png)

## About the VGG16 Model

![image](https://user-images.githubusercontent.com/64821137/181394116-1c5aa645-262e-48d1-a561-bac849cf86fc.png)

VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes.VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.It always uses 3 x 3 filters with stride of 1 in convolution layer and uses SAME padding in pooling layers 2 x 2 with stride of 2.
Here, the last softmax layer is not required as classification is not performed. Therefore, for feature extraction the outputs are from the fc2 layer.

## What is image similarity?
Image similarity is the measure of how similar images are. In other words, it quantifies the degree of similarity between intensity patterns in images.

Input an image

![image](https://user-images.githubusercontent.com/64821137/181394505-d1054c61-e4c4-488e-adf9-6ba0eb60fbe4.png)

Return the top 5 most similar products

![image](https://user-images.githubusercontent.com/64821137/181394586-7b8b05d8-24a7-40b2-8765-49f12a5a6d1e.png)
