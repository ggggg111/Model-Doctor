# Model-Doctor

## Overview

This is an unofficial implementation of the [Model Doctor: A Simple Gradient Aggregation Strategy for Diagnosing and
Treating CNN Classifiers](https://arxiv.org/abs/2112.04934).

## Requirements and Installation

This program has been developed and tested using Python 3.9.8 and CUDA 11.3 on Windows 10.

It is highly recommended to create and activate a conda environment first by running the commands below.

```
conda create -n modeldoctor python=3.9
conda activate modeldoctor
```

Next, run the [requirements.txt](./requirements.txt) file to install the necessary third-party packages with their respective versions.

```
pip install -r requirements.txt
```

## Instructions

The process of using Model Doctor is straightforward and the files to run the program have to be executed in a certain order. The program is written in a way so that both the diagnosing and treating stages are easily identified and related to the paper itself.

### Train a model

In the first place, a model has to be trained from zero. Make sure from now onwards, which model and dataset are being used, as well as the dataset path.

```
python src\train.py
    -- data_path [data_path]
    -- dataset [dataset]
    -- device [device]
    -- model_name [model_name]
    -- checkpoints_path [checkpoints_path]
    -- batch_size [batch_size]
    -- learning_rate [learning_rate]
    -- epochs [epochs]
```

### Test a model

When a model is trained, it can be tested on the selected dataset. Note that you can test the model not only after the training phase explained above, but also after the Model Doctor training phase.

```
python src\test.py
    -- data_path [data_path]
    -- dataset [dataset]
    -- device [device]
    -- model_name [model_name]
    -- checkpoints_path [checkpoints_path]
    -- checkpoint_file [checkpoint_file]
    -- batch_size [batch_size]
```

### Extract high-confidence samples

The next step is to extract high-confidence samples. By default, a number of 100 high-confidence samples are extracted for each class of the selected dataset. Furthermore, a high-confidence sample is one where the model has a 90% (inclusive) or more of confidence that it is from the correct class. Note that the high-confidence samples are used for the channel loss. The samples must be from the correct class that the model has predicted.

```
python src\generate_hc_images.py
    -- data_path [data_path]
    -- dataset [dataset]
    -- high_confidence_path [high_confidence_path]
    -- device [device]
    -- model_name [model_name]
    -- checkpoints_path [checkpoints_path]
    -- checkpoint_file [checkpoint_file]
    -- batch_size [batch_size]
    -- num_images [num_images]
    -- min_confidence [min_confidence]
```

### Extract low-confidence samples

Low-confidence samples are also required. By default, a number of 20 low-confidence samples are extracted for each class of the selected dataset. Furthermore, a low-confidence sample is one where the model has a maximum of 80% (non-inclusive) and a minimum of 60% (inclusive) of confidence that it is from the correct class. Note that the low-confidence samples are used for the spatial loss. The samples don't need to be from the correct class that the model has predicted because, otherwise, due to the high-accuracy of the already trained model, there are not enough samples to extract.

```
python src\generate_lc_images.py
    -- data_path [data_path]
    -- dataset [dataset]
    -- low_confidence_path [low_confidence_path]
    -- device [device]
    -- model_name [model_name]
    -- checkpoints_path [checkpoints_path]
    -- checkpoint_file [checkpoint_file]
    -- batch_size [batch_size]
    -- num_images [num_images]
    -- max_confidence [max_confidence]
    -- min_confidence [min_confidence]
```

### Generate low-confidence images masks

In this part, the idea is to mask the low-confidence images so that the background has a value of 1 and the rest has a value of 0. The low-confidence images have to be annotated manually, using a polygonal annotation tool. In this case, LabelMe has been used. When the low-confidence images have been labelled, they need to be converted into a mask. Note that at this point, only one polygonal shape is supported, so the drawing of the polygon should correspond to the outline of the object.

```
python src\annotations_to_mask.py
    -- low_confidence_annotations_path [low_confidence_annotations_path]
    -- low_confidence_masks_path [low_confidence_masks_path]
    -- dataset [dataset]
```

### Generate the gradients

The average correlation distribution of the target category and the 2D convolutional kernels is calculated using the high-confidence samples explained previously. Thus, the gradient of the target class w.r.t. the feature maps of the last layer are computed and extracted. This is known as the diagnosing stage.

```
python src\diagnosing.py
    -- high_confidence_samples_path [high_confidence_samples_path]
    -- dataset [dataset]
    -- device [device]
    -- model_name [model_name]
    -- checkpoints_path [checkpoints_path]
    -- checkpoint_file [checkpoint_file]
    -- gradients_path [gradients_path]
    -- delta [delta]
```

### View correlation distribution

The average correlation distribution of the feature maps of the last 2D convolutional layer of the model can be viewed as a heatmap. The sample image and its class have to be given in the command.

```
python src\statistical_correlation.py
    -- data_path [data_path]
    -- dataset [dataset]
    -- device [device]
    -- model_name [model_name]
    -- checkpoints_path [checkpoints_path]
    -- checkpoint_file [checkpoint_file]
    -- image_path [image_path]
    -- image_class [image_class]
```

### Train on Model Doctor

Having obtained all the previous data, now we can train the model on the Model Doctor. The new loss function can be now used, which is the sum of the original loss, the channel loss, and the spatial loss. When the training finishes, the saved model doesn't override the original one, but rather saves a new model in the original path but with a different file name.

```
python src\treating.py
    -- data_path [data_path]
    -- dataset [dataset]
    -- device [device]
    -- model_name [model_name]
    -- checkpoints_path [checkpoints_path]
    -- checkpoint_file [checkpoint_file]
    -- batch_size [batch_size]
    -- learning_rate [learning_rate]
    -- epochs [epochs]
    -- gradients_path [gradients_path]
    -- delta [delta]
```

## Models

The following models are currently supported:

- AlexNet.
- VGG16.
- Resnet50.
- WideResNet50.
- ResNeXt50.
- DenseNet121.
- EfficientNetV2.
- GoogLeNet.
- MobileNetV2.
- InceptionV3.
- ShuffleNetV2.
- SqueezeNet.
- MnasNet.

## Datasets

The following datasets are currently supported:

- MNIST.
- Fashion MNIST.
- CIFAR-10.
- CIFAR-100.
- SVHN.
- STL-10.