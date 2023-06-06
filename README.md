# Persian Handwritten Recognition

[Download Pretrained Models](https://huggingface.co/Azadeh/PersianHandwrittenDigits/tree/main)

## Abstract

This technical report describes the implementation of a Persian digit detector using a GPU or CPU. The detector is trained on the Hoda dataset, which contains handwritten Persian digit images. The ResNet-50 model is used for training, and the detector achieves a high accuracy (more than 99%) on the test images.

## Dataset

The Hoda dataset is used for training and testing the Persian digit detector. The dataset consists of two parts: the training set with 60,000 images and the test set with 20,000 images. The dataset is downloaded from the GitHub repository [aztofighi/Persian_Handwritten_Recognition](https://github.com/aztofighi/Persian_Handwritten_Recognition).

## Preprocessing

Before training the model, the images are preprocessed using the following steps:

1. Resize: The images are resized to a fixed size of 32x32 pixels using the `transforms.Resize` function from the `torchvision.transforms` module.
2. Normalize: The pixel values of the resized images are normalized using the `transforms.Normalize` function, with a mean of (0.5, 0.5, 0.5) and a standard deviation of (0.5, 0.5, 0.5).

## Model Architecture

The ResNet-50 model is used as the backbone for the Persian digit detector. The model is pretrained on the ImageNet dataset and then fine-tuned on the Hoda dataset. The fully connected layer of the model is replaced with a new linear layer that has the same number of output units as the number of classes in the Hoda dataset.

## Training

The model is trained using the following hyperparameters:

- Number of epochs: 2
- Batch size: 100
- Learning rate: 0.001

The training loop iterates over the training dataset for the specified number of epochs. The loss function used is the cross-entropy loss, and the optimizer used is Adam. The training loss for each epoch is stored for plotting.

## Testing

After training, the model is evaluated on the test dataset to measure its accuracy. The model is set to evaluation mode using `model.eval()` and the test images are passed through the model. The predicted labels are compared with the ground truth labels, and the accuracy is calculated.

## Saving and Loading the Model

The trained model is saved to a checkpoint file using the `torch.save` function. The saved model can be loaded later using the `torch.load` function. The model can then be used for inference or further training.

## Inference

To perform inference with the trained model, an image can be uploaded and processed. The uploaded image is first converted to grayscale, then thresholded to obtain a binary image. The outmost non-zero pixels are found and the image is cropped to remove zero padding. The cropped image is resized to 32x32 pixels and converted to RGB format. The resized image is normalized and passed through the model for prediction. The predicted label is displayed along with the original, binary, centered binary, and padded images.

## Conclusion

The Persian digit detector implemented in this project demonstrates the effectiveness of using a GPU for training deep learning models. The ResNet-50 model achieves high accuracy on the Hoda dataset, showcasing its ability to detect handwritten Persian digit images. This detector can be used in various applications, such as optical character recognition for Persian text.

