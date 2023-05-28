# Brain-Tumor-Detection

A brain tumor is a very critical disease which causes deaths of many individuals. Currently, doctors locate the position and the area of brain tumor by looking at the MR Images of the brain of the patient manually. This results in inaccurate detection of the tumor and is considered very time consuming.
This project deals with such a system, which uses computer, based procedures to detect tumor blocks using Convolution Neural Network Algorithm for MRI images of different patients.

The complete project report can be found [here](https://drive.google.com/file/d/1EqwRxBz-ELX_4tndb2sMG8GypmbpzqxD/view?usp=drivesdk).

## Objective

- The main aim of the application is tumor identification.
- The main reason behind the development of this application is to provide proper treatment as soon as possible and protect the human life which is in danger.
- This application is helpful to doctors as well as patient.
- The manual identification is not so fast, more accurate and efficient for user. To overcome those problems this application is designed.

## Dataset

The dataset selected for our model is acquired from the internet. In this dataset, there were different folders with a different set of images, one with a healthy brain image and the other with a brain tumor image. The MRI images present in the dataset are of different dimensions. This dataset is selected because acquiring data sets from hospitals is not a very simple task. The data set used is available on the internet website Kaggle and the link to reach there is provided below.

https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection.

Using these datasets, we have trained our model to predict brain tumor in MR images of any brain.

## Software Requirements

To run this project locally, you need to install the following :

- Python
- PIP
- VS Code
- Tensor Flow
- Keras
- Matplotlib
- NumPy
- SciKit Learn
- Flask

## Experimental Results

When the model is applied to the testing data set for 32 epochs, a validation accuracy of upto 85.19% is obtained and the validation loss is also less.

![lossed_plot](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/63eeee25-aebd-46f3-b088-13c772304f55)

fig. Model Loss

As seen in the above figure, when the model is applied to the validation and testing set, the loss 
gradually decreases with the increasing number of epochs.

![accuracy_plot](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/9ec6dd25-4699-4781-8f6f-520cb4a89a47)

fig. Model Accuracy

The accuracy of the convolutional neural network model achieved after applying it to the testing set was 97.79%. with a very minimal loss with increasing epochs.

![image](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/a5fb9d56-b68f-460e-ae62-a5f5465a03ce)
![image](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/6948b2aa-2d91-442f-b5d1-a5d8b050b010)

fig. Model History

From above figures, it can be confirmed that the accuracy increases with the increase in the number of epochs and there is a decrease in loss of the testing set.

![image](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/5114e2bd-0f2b-400f-832f-e129a1dcda0d)

fig. Model Summary

## Demonstration

The main model is created using python. But we have also created a simple web application using flask and html to manually test the results by uploading images with a user-friendly interface.

### Sample Input 1

![0 (no)](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/258dd13d-5a23-4757-8c2e-a3d22f5292ee)

fig. Non-tumor Brain MR Image

![image](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/e201a30c-7aab-4e65-8db3-675cb1fd7c85)

fig. Inititial Interface

![image](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/cac9586e-93a7-4469-b8e9-dccc9fbde3cd)

fig. Uploading MRI Image of Non-tumor Brain

![image](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/403e4b5c-3570-4301-993e-fd0aa290074c)

fig. Predicted Output for Sample Input 1


### Sample input 2

![3 (yes)](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/77523b01-f277-4953-bd65-715396b0ca22)

fig. Tumor Brain MR Image

![image](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/3160cb74-0f8a-4d74-8d47-925c0994c76b)

fig. Uploading MRI Image of Tumor Brain

![image](https://github.com/Riddhi9570/Brain-Tumor-Detection/assets/72887868/c1ad6169-cc62-4fec-8ddb-255f8f8c11ae)

fig. Predicted Output for Sample Input 2

Hence, the predicted output identifies the tumor and non-tumor brain MRI images.

## Conclusion and Future Scope

The created model achieves an accuracy of 97.79% when applied to the training set and an accuracy of upto 85.19% when applied to the validation set. It is observed on extermination that the proposed approach needs a vast training set for better accurate results. In future, different datasets would be applied to this model, to further increase the overall accuracy. The proposed approach can be further improvised through in cooperating weakly trained algorithms that can identify the abnormalities with a minimum training data and also self-learning algorithms would aid in enhancing the accuracy of the algorithm and reduce the computational tim

## Team Members

- [Kumar Yash](https://github.com/kumaryash18)
- [Aryan](https://github.com/AryanYuva)
- [Riddhi](https://github.com/riddhi9570)
