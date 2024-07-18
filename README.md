# Brain-Tumor-Classification

Project Overview
This project aims to develop an automated system for detecting and classifying brain tumors from MRI images using deep learning techniques. The dataset includes MRI images labeled with three types of brain tumors: Meningioma, Glioma, and Pituitary Tumor. A convolutional neural network (CNN) based on the VGG16 architecture, pre-trained on the ImageNet dataset, was fine-tuned for this task.

Abstract

The project focuses on creating a reliable and efficient model for brain tumor detection and classification. By leveraging a VGG16-based CNN model, the system demonstrates high accuracy in identifying different types of brain tumors, highlighting its potential to aid medical diagnosis.

Introduction
Brain tumors are severe and life-threatening conditions that require accurate and timely diagnosis. MRI imaging is a critical tool in detecting and classifying brain tumors, providing detailed insights into the brain's structure. This project leverages deep learning, specifically convolutional neural networks (CNNs), to automate the process of brain tumor detection and classification.

Dataset

The dataset for this project was gathered from Figshare and contains MRI images with corresponding labels indicating the type of tumor present. The dataset comprises 3064 T1-weighted contrast-enhanced MRI images from 233 patients. These images represent three types of brain tumors:

Meningioma: 708 slices
Glioma: 1426 slices
Pituitary Tumor: 930 slices
Methodology
Data Collection and Preprocessing
Normalization: Pixel values of the images were normalized to a range of [0, 1].
Contrast Stretching: Applied to enhance image contrast.
Gaussian Blurring: Used to reduce noise and smooth images.
Edge Enhancement: The Sobel operator was employed to enhance edges.
Morphological Operations: Morphological closing was performed to close small holes in the images.
Standardization: Feature-wise standardization was applied to the images.
Model Architecture

The project utilizes a CNN based on the VGG16 architecture. The model was fine-tuned to classify the MRI images into the three tumor types. The VGG16 model includes several convolutional and pooling layers, followed by fully connected layers, which were customized for this specific task.

Training and Evaluation
The model was trained using the preprocessed dataset and evaluated using various metrics to assess its performance. The evaluation included accuracy, precision, recall, and F1-score to ensure the model's robustness and reliability in real-world applications.

Results

The findings demonstrate the model's high accuracy in classifying brain tumors. The results highlight the system's potential for aiding medical diagnosis by providing reliable and accurate classification of brain tumors from MRI images.

Future Work
Future work will focus on expanding the dataset, exploring advanced architectures, incorporating 3D data, conducting clinical trials, and developing user-friendly interfaces for seamless integration into medical systems. These efforts aim to further enhance the model's accuracy, robustness, and applicability, ultimately improving patient outcomes and supporting medical professionals in their critical work.

References
LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
Litjens, G., et al. (2017). A survey on deep learning in medical image analysis. Medical image analysis, 42, 60-88.
Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115-118.
Ronneberger, O., et al. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
Cheng, J., et al. (2016). Enhanced performance of brain tumor classification via tumor region augmentation and partition. PloS one, 10(10), e0140381.
Zhang, Y., et al. (2018). Binary PSO with mutation operator for feature selection using decision tree applied to spam detection. Knowledge-Based Systems, 64, 22-31.
Reza, S. M. S., et al. (2019). Deep convolutional neural networks for brain tumor detection. IEEE Access, 7, 124677-124690.
For more detailed information, please refer to the project documentation.
